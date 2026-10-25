"""Oct 10: report for corrections + standard-RFT replay (r2hR / r2hRw / r1bR): wins, mean score, #score<0, paired McNemar vs references."""
import json
from math import comb
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; SF = ROOT / "reports/sciworld/stepfix"; BL = ROOT / "reports/sciworld/baselines"
E = {"R0": [SF / "r0/eval_test.jsonl", ROOT / "reports/sciworld/refix/eval_dev_safe.jsonl"],
     "r1b": [SF / "r1b/eval_stepfix/eval_test.jsonl", SF / "r1b/eval_stepfix/eval_dev.jsonl"],
     "r2h": [SF / "r2h/eval/eval_test.jsonl", SF / "r2h/eval/eval_dev.jsonl"],
     "rftfull_R1": [BL / "rftfull/r1/eval/eval_test.jsonl", BL / "rftfull/r1/eval/eval_dev.jsonl"],
     "rftfull_R2": [BL / "rftfull/r2/eval/eval_test.jsonl", BL / "rftfull/r2/eval/eval_dev.jsonl"]}
for n in ("r2hR", "r2hRw", "r1bR"): E[n] = [SF / n / "eval/eval_test.jsonl", SF / n / "eval/eval_dev.jsonl"]


def load(fs):
    R = {}
    for f in fs:
        if not f.exists(): return None
        for l in open(f):
            try: e = json.loads(l)
            except Exception: continue
            R.setdefault((e["split"], e["task"], e["variation"]), e)
    return R


def mc(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


R = {n: load(fs) for n, fs in E.items()}
for n, X in R.items():
    if X is None: print(f"{n}: (missing)"); continue
    print(f"{n:11s} n={len(X)} won {sum(e['won'] for e in X.values())} | test {sum(e['won'] for k, e in X.items() if k[0] == 'test')} dev {sum(e['won'] for k, e in X.items() if k[0] == 'dev')} | "
          f"mean score {sum(max(e['final_score'], 0) for e in X.values()) / len(X):.1f} | score<0 {sum(e['final_score'] < 0 for e in X.values())}")
for a, b in [("R0", "r2hR"), ("r2h", "r2hR"), ("rftfull_R2", "r2hR"), ("R0", "r2hRw"), ("r2h", "r2hRw"), ("rftfull_R2", "r2hRw"), ("r2hR", "r2hRw"),
             ("R0", "r1bR"), ("r1b", "r1bR"), ("rftfull_R1", "r1bR")]:
    A, B = R.get(a), R.get(b)
    if not A or not B: continue
    ks = sorted(set(A) & set(B)); oa = sum(A[k]["won"] and not B[k]["won"] for k in ks); ob = sum(B[k]["won"] and not A[k]["won"] for k in ks)
    print(f"{a} vs {b} [{len(ks)} paired]: {sum(A[k]['won'] for k in ks)} vs {sum(B[k]['won'] for k in ks)}; only-{a} {oa} / only-{b} {ob}; McNemar p = {mc(oa, ob):.2g}")
