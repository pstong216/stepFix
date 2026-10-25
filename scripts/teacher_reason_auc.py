"""Does extra reasoning / an attributor note make the SELF-teacher's preference informative?  No training: for the two
candidates of every pointer state (rollout labels known), re-score them under teacher prompt variants and report the AUC of the
teacher's candidate probability vs rollout success (baselines: plain 0.536, raw hindsight 0.526; env one-step 0.74).
  plain        policy prompt (student as is)
  hind         + raw hindsight block (what happened after t)                     [= the existing hindsight teacher]
  reason_plain + "this attempt FAILED" + the model's own 3-sentence analysis (generated first, no future shown)
  reason_hind  hindsight block + the model's own analysis of the failure
  modeA_hind   hindsight block + Model A's diagnosis (first decisive error step + Who&When error-mode code)
Writes per-state rows (analysis text, candidate probs, teacher top-1) to <out>.jsonl and a summary table.
    SE_DIR=reports/alfworld/self_evolve3b python scripts/teacher_reason_auc.py --arm attr --out reports/alfworld/self_evolve3b/proxy/teacher_reason_attr.jsonl
"""
import argparse, json, math, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch
from alfworld_attr_opd import history_at, load_jsonl
from alfworld_opd_plus import load_policy
from self_evolve import HindsightHistory, softmax
from env_evidence_filter import auc

ANALYSIS_Q = ("Before acting: this attempt FAILED. In at most 3 sentences, say why it failed and what the agent should do at the "
              "CURRENT step (do not name a specific action yet).")


class NoteHistory:
    """Policy history + optional hindsight block + a free-text note (analysis / attributor diagnosis) before the admissible list."""
    def __init__(self, h, note, hind=None):
        self.steps = h.steps; self._h = hind or h; self._note = note
    def render_prompt(self, max_history_steps=10):
        p = self._h.render_prompt(max_history_steps)
        return p.replace("Admissible actions:", self._note.rstrip() + "\nAdmissible actions:", 1)


@torch.no_grad()
def generate(pol, tok, model, history, question, max_new_tokens=96):
    user = history.render_prompt().replace("Next action:", question)
    msg = tok.apply_chat_template([{"role": "system", "content": "You are an embodied household agent reviewing a failed attempt."},
                                   {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    x = tok(msg, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**x, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, x.input_ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="attr"); ap.add_argument("--out", required=True)
    ap.add_argument("--traj", default="reports/alfworld/student3b_low100_train_b50.jsonl"); ap.add_argument("--student", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default=str(ROOT / "reports/alfworld/student3b_low100/final")); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--modeA", default="reports/alfworld/attr7bnoobs_student3b_low100_train_b50.jsonl")
    a = ap.parse_args(); SE = ROOT / os.environ.get("SE_DIR", "reports/alfworld/self_evolve3b")
    eps = {e["id"]: e for e in load_jsonl(ROOT / a.traj)}
    tg = load_jsonl(SE / a.arm / "targets.jsonl"); ex = {(r["id"], r["t"]): r for r in load_jsonl(SE / a.arm / "round1/examples.jsonl")}
    modeA = {r["id"]: r for r in load_jsonl(ROOT / a.modeA)} if (ROOT / a.modeA).exists() else {}
    out = ROOT / a.out; done = {(json.loads(l)["id"], json.loads(l)["t"]) for l in open(out)} if out.exists() else set()
    tok, model, pol = load_policy(a.student, a.adapter); model.eval()
    todo = [r for r in tg if (r["id"], r["t"]) not in done][: a.limit or None]; print(f"{len(todo)} states", flush=True)
    with open(out, "a") as f:
        for k, r in enumerate(todo):
            e = eps[r["id"]]; t = r["t"]; lp = r["teacher_logprobs"]
            cands = [c for c in sorted(lp, key=lambda c: -lp[c]) if c != r["a_minus"]][:2]
            won = {c: bool(ex[(r["id"], t)]["alternatives"].get(c, False)) if (r["id"], t) in ex else False for c in cands}
            h = history_at(e, t); hh = HindsightHistory(h, e, t)
            row = {"id": r["id"], "t": t, "cands": cands, "won": won, "a_minus": r["a_minus"], "variants": {}}
            an_plain = generate(pol, tok, model, NoteHistory(h, "Note: this attempt FAILED."), ANALYSIS_Q)
            an_hind = generate(pol, tok, model, hh, ANALYSIS_Q)
            row["analysis_plain"] = an_plain; row["analysis_hind"] = an_hind
            variants = {"plain": h, "hind": hh,
                        "reason_plain": NoteHistory(h, "Note: this attempt FAILED.\nAnalysis: " + an_plain),
                        "reason_hind": NoteHistory(h, "Analysis: " + an_hind, hind=hh)}
            if r["id"] in modeA:
                m = modeA[r["id"]]; variants["modeA_hind"] = NoteHistory(h, f"An external error attributor says the first decisive error was at step {m['t_hat']} "
                                                                           f"(error mode {m['pred'].get('error_mode')}).", hind=hh)
            for name, hist in variants.items():
                d = pol.score_admissible_commands_cached(hist); p = softmax(d)
                row["variants"][name] = {"p": {c: p.get(c, 0.0) for c in cands}, "top1": max(p, key=p.get), "H": -sum(q * math.log(q + 1e-12) for q in p.values())}
            f.write(json.dumps(row) + "\n"); f.flush()
            if k % 20 == 0: print(f"[{k + 1}/{len(todo)}]", flush=True)
    rows = load_jsonl(out); names = sorted({n for r in rows for n in r["variants"]})
    print(f"\n{len(rows)} states, {sum(sum(r['won'].values()) for r in rows)} winning candidates of {sum(len(r['cands']) for r in rows)}")
    print(f"{'variant':14s} {'AUC':>6s} {'within-state':>12s} {'top1 wins':>10s} {'top1 = a-':>10s} {'H':>5s}")
    for n in names:
        y, s, ws, wn, t1w, t1a = [], [], 0, 0, 0, 0
        for r in rows:
            v = r["variants"].get(n)
            if not v: continue
            for c in r["cands"]: y.append(r["won"][c]); s.append(v["p"][c])
            if len(r["cands"]) == 2 and r["won"][r["cands"][0]] != r["won"][r["cands"][1]]:
                w = r["cands"][0] if r["won"][r["cands"][0]] else r["cands"][1]; l = r["cands"][1] if w == r["cands"][0] else r["cands"][0]
                wn += 1; ws += (v["p"][w] > v["p"][l]) + 0.5 * (v["p"][w] == v["p"][l])
            t1w += bool(r["won"].get(v["top1"], False)); t1a += v["top1"] == r["a_minus"]
        print(f"{n:14s} {auc(y, s):6.3f} {ws / max(1, wn):8.3f} ({wn:3d}) {t1w:10d} {t1a:10d} {sum(r['variants'][n]['H'] for r in rows if n in r['variants']) / len(rows):5.2f}")
    print("DONE teacher_reason", flush=True)


if __name__ == "__main__":
    main()
