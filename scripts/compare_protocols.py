"""Paired comparison of two self-correction protocol runs (e.g. one_shot vs
debate_round) on their COMMON traces: per-condition recovery + answer-change
rate for each protocol, and exact McNemar of protocol A vs B per condition.

Usage:
    python scripts/compare_protocols.py \
        --a one_shot=reports/selfplay_chain_one_shot_full/corrections.jsonl \
        --b debate_round=reports/selfplay_chain_debate_round_150/corrections.jsonl
"""
import argparse
import json
from collections import defaultdict

from scipy.stats import binomtest

CONDITIONS = ["incorrect_only", "reason_only", "vague_reason", "precise_reason", "precise_only"]


def load(path, keep=None, regrade=False):
    if regrade:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from error_attr.selfplay.debate_runner import is_correct, parse_task_answer
    d = {}
    for l in open(path):
        r = json.loads(l)
        if keep is not None and r["id"] not in keep:
            continue
        if regrade:  # current parser/grader (DECISIONS #25/#26) applied to the stored raw text
            r["correct"] = is_correct(r["benchmark"], parse_task_answer(r["benchmark"], r["raw"]), r["gold"])
        d[(r["id"], r["condition"])] = r
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="NAME=corrections.jsonl")
    ap.add_argument("--b", required=True, help="NAME=corrections.jsonl")
    ap.add_argument("--ids_file", default=None, help="JSON list: restrict to these trace ids")
    ap.add_argument("--regrade", action="store_true", help="recompute `correct` from raw with the current grader")
    args = ap.parse_args()
    keep = set(json.load(open(args.ids_file))) if args.ids_file else None
    na, pa = args.a.split("=", 1); nb, pb = args.b.split("=", 1)
    A, B = load(pa, keep, args.regrade), load(pb, keep, args.regrade)
    common = sorted(set(A) & set(B))
    ids = sorted({i for i, _ in common})
    print(f"common traces: {len(ids)} | common (trace, condition) pairs: {len(common)}\n")
    print(f"{'condition':16s} {na+' rec':>16s} {nb+' rec':>16s} {na+' chg':>12s} {nb+' chg':>12s}   McNemar {nb} vs {na}")
    for c in CONDITIONS:
        keys = [(i, c) for i in ids if (i, c) in A and (i, c) in B]
        if not keys:
            continue
        ra = sum(A[k]["correct"] for k in keys); rb = sum(B[k]["correct"] for k in keys)
        ca = sum(A[k].get("changed", False) for k in keys); cb = sum(B[k].get("changed", False) for k in keys)
        b_only = sum(B[k]["correct"] and not A[k]["correct"] for k in keys)
        a_only = sum(A[k]["correct"] and not B[k]["correct"] for k in keys)
        p = binomtest(b_only, a_only + b_only, 0.5).pvalue if a_only + b_only else float("nan")
        n = len(keys)
        print(f"{c:16s} {ra:5d}/{n:<4d}={ra/n:5.3f} {rb:5d}/{n:<4d}={rb/n:5.3f} {ca/n:12.3f} {cb/n:12.3f}   {nb}-only={b_only:3d} {na}-only={a_only:3d} p={p:.3g}")


if __name__ == "__main__":
    main()
