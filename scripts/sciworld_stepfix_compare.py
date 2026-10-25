"""Paired comparison of two ScienceWorld evals on the 848 set (test 651 + dev 197), paired by (split, task, variation).
    python scripts/sciworld_stepfix_compare.py --a NAME=test.jsonl,dev.jsonl --b NAME=test.jsonl,dev.jsonl"""
import argparse, json
from math import comb


def load(spec):
    name, files = spec.split("="); rows = {}
    for f in files.split(","):
        for l in open(f):
            try: e = json.loads(l)
            except Exception: continue
            rows.setdefault((e["split"], e["task"], e["variation"]), e)
    return name, rows


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n)


ap = argparse.ArgumentParser(); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); a = ap.parse_args()
na, A = load(a.a); nb, B = load(a.b); keys = sorted(set(A) & set(B))
def summ(R, ks): return sum(R[k]["won"] for k in ks), sum(max(R[k]["final_score"], 0) for k in ks) / max(1, len(ks))
for part, ks in [("848", keys), ("test", [k for k in keys if k[0] == "test"]), ("dev", [k for k in keys if k[0] == "dev"])]:
    wa, sa = summ(A, ks); wb, sb = summ(B, ks); oa = sum(A[k]["won"] and not B[k]["won"] for k in ks); ob = sum(B[k]["won"] and not A[k]["won"] for k in ks)
    print(f"[{part}: {len(ks)} paired] {na}: {wa} won, mean score {sa:.1f} | {nb}: {wb} won, mean score {sb:.1f} | only-{na} {oa} / only-{nb} {ob}, McNemar exact p = {mcnemar(oa, ob):.2g}")
if len(keys) != 848: print(f"WARNING: {len(keys)} paired episodes (A {len(A)}, B {len(B)}), expected 848")
