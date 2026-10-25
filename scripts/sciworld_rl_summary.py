"""Summary of a ScienceWorld online-RL baseline eval (Oct 10 2026) on the 848 set, against R0 and our StepFix rounds.
usage: python scripts/sciworld_rl_summary.py TAG [TAG ...]   (reads reports/sciworld/rl_baselines/<TAG>/eval_{test,dev}.jsonl)
Reports wins, mean final score (negatives clipped to 0), episodes ending with score < 0, and exact two-sided McNemar p vs each reference."""
import json, sys
from math import comb
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; S = ROOT / "reports/sciworld"
REFS = {"R0": ["stepfix/r0/eval_test.jsonl", "refix/eval_dev_safe.jsonl"],
        "r1b": ["stepfix/r1b/eval_stepfix/eval_test.jsonl", "stepfix/r1b/eval_stepfix/eval_dev.jsonl"],
        "r2h": ["stepfix/r2h/eval/eval_test.jsonl", "stepfix/r2h/eval/eval_dev.jsonl"],
        "r3h": ["stepfix/r3h/eval/eval_test.jsonl", "stepfix/r3h/eval/eval_dev.jsonl"]}


def load(files):
    out = {}
    for f in files:
        for l in open(S / f):
            try: e = json.loads(l)
            except Exception: continue
            out.setdefault((e["split"], e["task"], int(e["variation"])), e)
    return out


def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def stats(d):
    sc = [e["final_score"] for e in d.values()]
    return sum(e["won"] for e in d.values()), sum(max(s, 0) for s in sc) / len(sc), sum(s < 0 for s in sc)


refs = {k: load(v) for k, v in REFS.items()}
for tag in sys.argv[1:]:
    d = load([f"rl_baselines/{tag}/eval_test.jsonl", f"rl_baselines/{tag}/eval_dev.jsonl"])
    w, ms, neg = stats(d)
    print(f"[{tag}] n={len(d)} wins={w} mean_score_clip0={ms:.2f} score<0={neg}  (test {sum(e['won'] for k, e in d.items() if k[0] == 'test')}/"
          f"{sum(k[0] == 'test' for k in d)}, dev {sum(e['won'] for k, e in d.items() if k[0] == 'dev')}/{sum(k[0] == 'dev' for k in d)})")
    for name, r in refs.items():
        common = set(d) & set(r)
        b = sum(d[k]["won"] and not r[k]["won"] for k in common); c = sum(r[k]["won"] and not d[k]["won"] for k in common)
        rw, rms, rneg = stats({k: r[k] for k in common})
        print(f"   vs {name:4s}: {name} wins {rw}, mean {rms:.2f}, score<0 {rneg} | {tag}-only {b}, {name}-only {c}, McNemar p={mcnemar(b, c):.4f} (n={len(common)})")
