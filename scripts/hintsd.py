"""HINT-SD (Yeo et al., EMNLP Findings 2026, arXiv 2605.17873) re-implemented in our ALFWorld framework as a baseline (Oct 7).
Faithful to the official code (github.com/wgcyeo/HINT-SD, src/train/appworld_feedback.py + self_distillation.py):
  analyze -- the CURRENT student (no stronger model) reads its full failed trajectory and returns up to 3 problematic steps, each with a
             correction of < 3 sentences, as JSON (the official multi-step prompt, 'AppWorld tool-use' -> 'ALFWorld household'); sampling
             temperature 0.6, <= 512 new tokens, trajectory text clipped to its last 12000 characters.  Unparseable outputs are skipped.
  teacher -- the same student conditioned on the original prefix + the feedback placed at the target turn scores the admissible commands;
             the teacher distribution p_T becomes an 'rkl' row (reverse KL KL(q_student || p_T) on that step, as in HINT-SD).
No environment verification, no replay, no budget: every parsed (failure, step) is used.
"""
import argparse, json, re, sys, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch
from alfworld_attr_opd import history_at, load_jsonl
from alfworld_opd_plus import load_policy
from teacher_reason_auc import NoteHistory

SYSTEM = ("You analyze failed ALFWorld household-task trajectories. Identify up to {k} problematic steps where the agent made mistakes "
          "(between 1 and {k} steps total, ordered earliest first). For EACH problematic step, write a distinct correction in less than "
          "three sentences that targets only that step's mistake. Output valid JSON only, no other text.")
USER = ("Review this failed ALFWorld attempt and identify the problematic steps. Return one (step, feedback) pair per problematic step.\n\n"
        "{traj}\n\nOutput format (JSON only, no other text):\n"
        '{{"failures": [{{"step": <1-indexed step number>, "feedback": "<correction for this step>"}}, ...]}}')


def render(e, max_chars=12000):
    lines = [f"Task: {e['goal']}", f"Initial observation: {e['initial_obs']}"]
    for k, s in enumerate(e["steps"]): lines += [f"Step {k + 1}: Action: {s['action']}", f"Observation: {s['next_obs']}"]
    lines.append("Outcome: FAILED")
    t = "\n".join(lines)
    return t if len(t) <= max_chars else "..." + t[-max_chars:]


def parse(raw, n, k=3):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m: return None
    try: obj = json.loads(m.group(0))
    except Exception: return None
    out = []
    for f in (obj.get("failures") or [])[:k]:
        try: st = int(f["step"]); fb = str(f["feedback"]).strip()
        except Exception: continue
        if 1 <= st <= n and fb and (st - 1) not in [x[0] for x in out]: out.append((st - 1, fb))
    return out or None


def cmd_analyze(a):
    eps = {e["id"]: e for e in load_jsonl(ROOT / a.traj)}; ids = [r["id"] for r in load_jsonl(ROOT / a.ids)]
    if a.shard: ids = ids[a.shard[0]::a.shard[1]]
    out = ROOT / a.out; done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    tok, model, pol = load_policy(a.model, str(ROOT / a.adapter)); model.eval(); torch.manual_seed(a.seed)
    print(f"{len(ids) - len(done & set(ids))} failures to analyze", flush=True)
    with open(out, "a") as f, torch.no_grad():
        for n, i in enumerate(ids):
            if i in done or i not in eps: continue
            e = eps[i]; msgs = [{"role": "system", "content": SYSTEM.format(k=a.max_steps)}, {"role": "user", "content": USER.format(traj=render(e))}]
            x = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), return_tensors="pt", add_special_tokens=False).to(model.device)
            o = model.generate(**x, max_new_tokens=512, do_sample=True, temperature=a.temperature, top_p=1.0, pad_token_id=tok.pad_token_id)
            raw = tok.decode(o[0, x.input_ids.shape[1]:], skip_special_tokens=True).strip(); pr = parse(raw, len(e["steps"]), a.max_steps)
            f.write(json.dumps({"id": i, "targets": [{"t": t, "feedback": fb} for t, fb in pr] if pr else [], "parsed": bool(pr), "raw": raw}) + "\n"); f.flush()
            if n % 25 == 0: print(f"[{n + 1}/{len(ids)}]", flush=True)
    print("DONE analyze", flush=True)


def cmd_teacher(a):
    eps = {e["id"]: e for e in load_jsonl(ROOT / a.traj)}; rows = [r for r in load_jsonl(ROOT / a.analysis) if r["parsed"]]
    items = [(r["id"], x["t"], x["feedback"]) for r in rows for x in r["targets"]]
    if a.shard: items = items[a.shard[0]::a.shard[1]]
    out = ROOT / a.out; done = {(json.loads(l)["id"], json.loads(l)["t"]) for l in open(out)} if out.exists() else set()
    tok, model, pol = load_policy(a.model, str(ROOT / a.adapter)); model.eval()
    with open(out, "a") as f, torch.no_grad():
        for n, (i, t, fb) in enumerate(items):
            if (i, t) in done: continue
            h = history_at(eps[i], t); lp = pol.score_admissible_commands_cached(NoteHistory(h, "Feedback on this step (from a review of the failed attempt): " + fb))
            cmds = list(lp); p = torch.softmax(torch.tensor([lp[c] for c in cmds]), 0).tolist()
            f.write(json.dumps({"kind": "rkl", "src": a.traj, "id": i, "t": t, "cmds": cmds, "target": p, "feedback": fb}) + "\n"); f.flush()
            if n % 50 == 0: print(f"[{n + 1}/{len(items)}]", flush=True)
    print("DONE hintsd_teacher", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("analyze", "teacher"):
        s = sub.add_parser(name); s.add_argument("--traj", required=True); s.add_argument("--adapter", required=True); s.add_argument("--out", required=True)
        s.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); s.add_argument("--shard", type=int, nargs=2, default=None)
        if name == "analyze": s.add_argument("--ids", required=True); s.add_argument("--max_steps", type=int, default=3); s.add_argument("--temperature", type=float, default=0.6); s.add_argument("--seed", type=int, default=0)
        else: s.add_argument("--analysis", required=True)
    a = ap.parse_args(); cmd_analyze(a) if a.cmd == "analyze" else cmd_teacher(a)


if __name__ == "__main__":
    main()
