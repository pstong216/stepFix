"""Supplementary long-budget report (Oct 10): per type and total success / mean score at the long budget vs at 50 steps for R0, r1b, r2f;
McNemar r2f vs R0 and r1b vs R0 on the long-budget episodes; '848 with long budgets' totals (long-type episodes at the long budget,
the rest from the 50-step eval).   python scripts/sciworld_longbudget_report.py"""
import json, sys
from math import comb
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; SF = ROOT / "reports/sciworld/stepfix"
M50 = {"R0": [SF / "r0/eval_test.jsonl", ROOT / "reports/sciworld/refix/eval_dev_safe.jsonl"],
       "r1b": [SF / "r1b/eval_stepfix/eval_test.jsonl", SF / "r1b/eval_stepfix/eval_dev.jsonl"],
       "r2f": [SF / "r2f/eval/eval_test.jsonl", SF / "r2f/eval/eval_dev.jsonl"]}


def load(fs):
    R = {}
    for f in fs:
        if not Path(f).exists(): continue
        for l in open(f):
            try: e = json.loads(l)
            except Exception: continue
            R.setdefault((e["split"], e["task"], e["variation"]), e)
    return R


def mcn(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


s50 = {m: load(fs) for m, fs in M50.items()}
lb = {m: load([SF / f"longbudget/{m}/eval_test.jsonl", SF / f"longbudget/{m}/eval_dev.jsonl"]) for m in M50}
keys = sorted(set.intersection(*[set(lb[m]) for m in lb])) if all(lb.values()) else sorted(next((set(v) for v in lb.values() if v), set()))
types = sorted({k[1] for k in keys}); print(f"{len(keys)} long-budget episodes ({len(types)} types); models with long-budget results: {[m for m in lb if lb[m]]}")
sc = lambda e: max(e["final_score"], 0)
for t in types + ["TOTAL"]:
    ks = [k for k in keys if t == "TOTAL" or k[1] == t]; row = f"{t[:44]:44s} n={len(ks):3d}"
    for m in M50:
        if not lb[m]: continue
        a = [s50[m][k] for k in ks if k in s50[m]]; b = [lb[m][k] for k in ks]
        row += f" | {m}: 50-step {sum(e['won'] for e in a):3d} ({sum(map(sc, a)) / max(1, len(a)):4.1f}) long {sum(e['won'] for e in b):3d} ({sum(map(sc, b)) / max(1, len(b)):4.1f})"
    print(row)
for m in ("r1b", "r2f"):
    if lb[m] and lb["R0"]:
        o0 = sum(lb["R0"][k]["won"] and not lb[m][k]["won"] for k in keys); o1 = sum(lb[m][k]["won"] and not lb["R0"][k]["won"] for k in keys)
        print(f"long-budget episodes: {m} vs R0: {sum(lb[m][k]['won'] for k in keys)} vs {sum(lb['R0'][k]['won'] for k in keys)}; only-R0 {o0} / only-{m} {o1}, McNemar exact p = {mcn(o0, o1):.2g}")
comb848 = {}
for m in M50:
    if not lb[m]: continue
    R = dict(s50[m]); R.update({k: lb[m][k] for k in keys}); comb848[m] = R
    print(f"848 with long budgets [{m}]: {sum(e['won'] for e in R.values())}/{len(R)} won, mean score {sum(map(sc, R.values())) / len(R):.1f} (50-step: {sum(e['won'] for e in s50[m].values())})")
for m in ("r1b", "r2f"):
    if m in comb848 and "R0" in comb848:
        A, B = comb848["R0"], comb848[m]; ks = sorted(set(A) & set(B)); o0 = sum(A[k]["won"] and not B[k]["won"] for k in ks); o1 = sum(B[k]["won"] and not A[k]["won"] for k in ks)
        print(f"848 with long budgets: {m} vs R0 only-R0 {o0} / only-{m} {o1}, McNemar exact p = {mcn(o0, o1):.2g}")
