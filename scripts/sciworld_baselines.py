"""ScienceWorld RFT baselines (Oct 10 2026): data builders for the two rejection-sampling fine-tuning baselines that run with the SAME
collection / trainer / 848 eval as StepFix (scripts/run_sciworld_stepfix_round.sh).  The trainer is scripts/sciworld_stepfix.py train (used
read-only); rows written here use its schema {"kind", "src", "id", "t", "action"} and are rendered with sciworld_agent.render (10-step history).
  merge  collection shards (dedupe by id, sorted by (task, variation) as sciworld_stepfix.py prep) -> DIR/traj.jsonl
  build  --mode rftm    : N steps sampled uniformly at random (seed) from all steps of the round's successful episodes, skipping
                          'look around' / 'inventory' / 'task' (N = the row count StepFix trained on in that round)
         --mode rftfull : every step of every successful episode; only exact duplicates (same rendered student prompt AND same action) dropped
"""
import argparse, glob, json, os, random, sys
import os as _os_umask; _os_umask.umask(0o027)   # Oct 12: some srun steps inherit umask 0117 (dirs created without x); force 0027
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("ERROR_ATTR_ROOT", str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from sciworld_agent import render
SKIP = ("look around", "inventory", "task")   # same skip list as the StepFix replay rows


def L(p):
    out = []
    for l in open(p):
        try: out.append(json.loads(l))
        except Exception: pass
    return out


def cmd_merge(a):
    D = ROOT / a.dir; rows = {}
    for p in sorted(glob.glob(str(ROOT / a.inputs))):
        for e in L(p): rows.setdefault(e["id"], e)
    eps = sorted(rows.values(), key=lambda e: (e["task"], e["variation"]))
    open(D / "traj.jsonl.tmp", "w").writelines(json.dumps(e) + "\n" for e in eps); os.replace(D / "traj.jsonl.tmp", D / "traj.jsonl")
    won = sum(e["won"] for e in eps)
    print(f"merge {a.dir}: {len(eps)} episodes ({a.expect} expected), {won} won, mean score {sum(max(e['final_score'], 0) for e in eps) / max(1, len(eps)):.1f}")
    if a.expect and len(eps) != a.expect: print(f"WARNING: {len(eps)} episodes != expected {a.expect}"); sys.exit(2)


def cmd_build(a):
    D = ROOT / a.dir; tp = D / "traj.jsonl"; src = str(tp.relative_to(ROOT)); eps = L(tp); wins = [e for e in eps if e["won"]]
    rows, stats = [], {"mode": a.mode, "episodes": len(eps), "wins": len(wins), "win_steps_total": sum(len(e["steps"]) for e in wins)}
    if a.mode == "rftm":
        pool = [(e["id"], t, s["action"]) for e in wins for t, s in enumerate(e["steps"]) if s["action"].strip() not in SKIP]
        rng = random.Random(a.seed); pick = rng.sample(pool, min(a.n, len(pool)))
        rows = [{"kind": "rft", "src": src, "id": i, "t": t, "action": act} for i, t, act in pick]
        stats.update(pool=len(pool), n_target=a.n, n=len(rows), seed=a.seed, from_episodes=len({r["id"] for r in rows}))
    else:
        seen, dup = set(), 0
        for e in wins:
            hist = []
            for t, s in enumerate(e["steps"]):
                key = (render(e["goal"], hist + [{"obs": s["obs"], "action": None}], 10), s["action"]); hist.append({"obs": s["obs"], "action": s["action"]})
                if key in seen: dup += 1; continue
                seen.add(key); rows.append({"kind": "rft", "src": src, "id": e["id"], "t": t, "action": s["action"]})
        stats.update(n=len(rows), exact_duplicates_dropped=dup, from_episodes=len({r["id"] for r in rows}))
    if a.limit:   # smoke test only: the rows with the longest histories (worst-case memory)
        rows = sorted(rows, key=lambda r: -r["t"])[: a.limit]; stats["smoke_limit"] = a.limit
    stats["by_task_wins"] = dict(Counter(e["task"] for e in wins)); stats["skip_actions_in_rows"] = sum(r["action"].strip() in SKIP for r in rows)
    out = D / a.out
    open(str(out) + ".tmp", "w").writelines(json.dumps(r) + "\n" for r in rows); os.replace(str(out) + ".tmp", out)
    json.dump(stats, open(D / (Path(a.out).stem + "_stats.json"), "w"), indent=1)
    print("build:", json.dumps({k: v for k, v in stats.items() if k != "by_task_wins"}))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("merge"); s.add_argument("--dir", required=True); s.add_argument("--inputs", required=True); s.add_argument("--expect", type=int, default=0)
    s = sub.add_parser("build"); s.add_argument("--dir", required=True); s.add_argument("--mode", choices=["rftm", "rftfull"], required=True)
    s.add_argument("--n", type=int, default=0); s.add_argument("--seed", type=int, default=0); s.add_argument("--out", default="train_rows.jsonl"); s.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
