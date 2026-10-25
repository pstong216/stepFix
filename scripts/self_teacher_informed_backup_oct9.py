"""Informed self-teacher: the SAME student model, given what the student could not see when it acted -- the failed episode's
future (hindsight), the self-trained attributor's diagnosis, and its own written analysis -- scores every admissible command at
the pointer step.  Two calls per state: generate the analysis, then score with the analysis filled in.
Writes targets.jsonl in the hindsight_steps schema (teacher_logprobs = informed, plain_logprobs = plain student) + the analysis.
Ablations: --no_hind / --no_diag / --no_analysis.  Resumable.
    python scripts/self_teacher_informed.py --traj T --steps D/self/steps.jsonl --diag D/attr/pointer.raw.jsonl --adapter S --out D/self/targets.jsonl
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch
from alfworld_attr_opd import history_at, load_jsonl
from alfworld_opd_plus import load_policy
from self_evolve import HindsightHistory
from teacher_reason_auc import NoteHistory, generate, ANALYSIS_Q

TYPE_WORDS = {"wrong_location": "went to the wrong place", "wrong_object": "took the wrong object", "wrong_target": "acted on the wrong place or object",
              "misorder": "did a later step too early (e.g. placed before cleaning/heating)", "misplace": "put the object in the wrong place",
              "wrong_value": "put the object in the wrong place", "wrong_op": "used the wrong action", "skip": "skipped a required step", "random": "an unproductive action"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--traj", required=True); ap.add_argument("--steps", required=True); ap.add_argument("--diag", default=None)
    ap.add_argument("--adapter", required=True); ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, nargs=2, default=None); ap.add_argument("--no_hind", action="store_true"); ap.add_argument("--no_diag", action="store_true"); ap.add_argument("--no_analysis", action="store_true")
    a = ap.parse_args()
    eps = {e["id"]: e for e in load_jsonl(ROOT / a.traj)}; steps = [(r["id"], r["steps"][0]) for r in load_jsonl(ROOT / a.steps)]
    diag = {r["id"]: r for r in load_jsonl(ROOT / a.diag)} if a.diag and (ROOT / a.diag).exists() else {}
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    steps = [s for s in steps if s[0] in eps]
    if a.shard: steps = steps[a.shard[0]::a.shard[1]]   # shards write their own --out (Oct 6 14:40)
    # Oct 8 fix: shard BEFORE the resume filter. Before, a restarted shard re-sliced the shrunken todo list, so it redid other shards'
    # states and skipped some of its own. On (re)start, keep only this shard's own rows in its --out (first copy of each state).
    done = set()
    if out.exists():
        own, keep = set(steps), []
        for l in open(out):
            try: r = json.loads(l)
            except Exception: continue
            k = (r["id"], r["t"])
            if k in own and k not in done: done.add(k); keep.append(l if l.endswith("\n") else l + "\n")
        open(out, "w").writelines(keep)
    todo = [s for s in steps if s not in done]
    print(f"{len(todo)} states (hind={not a.no_hind} diag={not a.no_diag and bool(diag)} analysis={not a.no_analysis})", flush=True)
    tok, model, pol = load_policy(a.model, str(ROOT / a.adapter)); model.eval()
    with open(out, "a") as f, torch.no_grad():
        for k, (i, t) in enumerate(todo):
            e = eps[i]; h = history_at(e, t); lp_plain = pol.score_admissible_commands_cached(h)
            hh = None if a.no_hind else HindsightHistory(h, e, t)
            parts = []; d = diag.get(i)
            if d and not a.no_diag:
                typ = d.get("type"); parts.append(f"Diagnosis (an error attributor trained on this agent's own trajectories): the first decisive error was at step {d['t_hat']}"
                                                  + (f", type: {typ} ({TYPE_WORDS.get(typ, typ)})" if typ else "") + ".")
            an = None
            if not a.no_analysis:
                an = generate(pol, tok, model, NoteHistory(h, "\n".join(parts) if parts else "Note: this attempt FAILED.", hind=hh), ANALYSIS_Q)
                parts.append("Analysis: " + an)
            hist = NoteHistory(h, "\n".join(parts), hind=hh) if parts else (hh or h)
            lp = pol.score_admissible_commands_cached(hist)
            f.write(json.dumps({"id": i, "t": t, "a_minus": e["steps"][t]["action"], "admissible": list(lp), "teacher_logprobs": lp, "plain_logprobs": lp_plain,
                                "mode": "informed", "analysis": an, "diag": d}) + "\n"); f.flush()
            if k % 20 == 0: print(f"[{k + 1}/{len(todo)}]", flush=True)
    print("DONE informed_teacher", flush=True)


if __name__ == "__main__":
    main()
