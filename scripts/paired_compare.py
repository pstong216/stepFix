"""Paired (per-game McNemar) comparison of policies on the pooled evaluation: valid_unseen 134 + valid_seen 140 (+ held-out train
500 when present).  Policies are given as name=dir (dir holds eval_unseen.jsonl / eval_seen.jsonl; the held-out file is looked
up as reports/alfworld/ext/eval_heldout_<name>.jsonl) or name=<tag> for the student baseline.
    python scripts/paired_compare.py env_r2=reports/alfworld/self_evolve3b_rounds/env/r2 placebo_r2=... --vs student
"""
import argparse, json, os, sys
from itertools import combinations
from scipy.stats import binomtest
E = "reports/alfworld/ext"
STUDENT = {"unseen": "reports/alfworld/eval_unseen_student3b_low100.jsonl", "seen": f"{E}/eval_seen_student_low100.jsonl", "heldout": f"{E}/eval_heldout_student_low100.jsonl"}


def load(p):
    if not p or not os.path.exists(p): return {}
    return {json.loads(l)["game_file"]: bool(json.loads(l).get("won")) for l in open(p)}


def pooled(name, d):
    if name == "student": files = STUDENT
    else: files = {"unseen": f"{d}/eval_unseen.jsonl", "seen": f"{d}/eval_seen.jsonl", "heldout": f"{E}/eval_heldout_{name}.jsonl"}
    parts = {k: load(v) for k, v in files.items()}; out = {}
    for k, v in parts.items(): out.update(v)
    return out, {k: (sum(v.values()), len(v)) for k, v in parts.items() if v}


def mc(a, b):
    g = [x for x in a if x in b]; ab = sum(a[x] and not b[x] for x in g); ba = sum(b[x] and not a[x] for x in g)
    return ab, ba, len(g), binomtest(ab, ab + ba).pvalue if ab + ba else 1.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("specs", nargs="+"); ap.add_argument("--vs", default=None, help="compare everything against this name only")
    a = ap.parse_args(); P = {}
    specs = [s.split("=", 1) for s in a.specs] + ([["student", ""]] if a.vs == "student" or any(s == "student" for s in a.specs) else [])
    for name, d in specs:
        P[name], parts = pooled(name, d); print(f"{name:22s} " + "  ".join(f"{k} {w}/{n}" for k, (w, n) in parts.items()) + f"  | pooled {sum(P[name].values())}/{len(P[name])}")
    print()
    pairs = [(a.vs, n) for n in P if n != a.vs] if a.vs else list(combinations(P, 2))
    for x, y in pairs:
        ab, ba, n, p = mc(P[x], P[y]); print(f"{x:22s} vs {y:22s}: {x} wins only {ab:3d}, {y} wins only {ba:3d}, common games {n}, p={p:.3g}")


if __name__ == "__main__":
    main()
