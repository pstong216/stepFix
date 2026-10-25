"""Cross-condition paired comparison across one or more chain runs on their
COMMON traces (e.g. the controls run vs the main run of the same protocol):
per-condition recovery + answer-change rate, and exact McNemar for requested
condition pairs, where the two conditions may come from different files.

Usage:
    python scripts/compare_conditions.py \
        --run reports/selfplay_chain_debate_round_150/corrections.jsonl \
        --run reports/selfplay_chain_debate_round_150_controls/corrections.jsonl \
        --pair wrong_location:precise_only --pair placebo_reason:reason_only \
        --pair random_attribution:precise_reason --pair wrong_location:incorrect_only
"""
import argparse
import json
from collections import defaultdict

from scipy.stats import binomtest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="corrections.jsonl (repeatable)")
    ap.add_argument("--pair", action="append", default=[], help="condA:condB (repeatable)")
    ap.add_argument("--by_benchmark", action="store_true")
    ap.add_argument("--ids_file", default=None, help="JSON list: restrict to these trace ids (e.g. true_failure_ids.json)")
    ap.add_argument("--regrade", action="store_true",
                    help="recompute `correct` from the stored raw text with the CURRENT parser/grader (DECISIONS #25/#26)")
    args = ap.parse_args()

    keep = set(json.load(open(args.ids_file))) if args.ids_file else None
    if args.regrade:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from error_attr.selfplay.debate_runner import is_correct, parse_task_answer
    per = defaultdict(dict)   # id -> {condition: record}
    for path in args.run:
        for l in open(path):
            r = json.loads(l)
            if keep is not None and r["id"] not in keep:
                continue
            if args.regrade:
                r["correct"] = is_correct(r["benchmark"], parse_task_answer(r["benchmark"], r["raw"]), r["gold"])
            per[r["id"]][r["condition"]] = r
    conds = sorted({c for d in per.values() for c in d})
    common = [i for i, d in per.items() if all(c in d for c in conds)]
    print(f"traces: {len(per)} total, {len(common)} with all {len(conds)} conditions\n")

    print(f"{'condition':20s} {'recovery':>14s} {'changed':>9s} {'correct|changed':>16s}")
    for c in conds:
        rows = [per[i][c] for i in common]
        n = len(rows); rec = sum(r["correct"] for r in rows); chg = sum(r.get("changed", False) for r in rows)
        cc = sum(r["correct"] and r.get("changed", False) for r in rows)
        print(f"{c:20s} {rec:4d}/{n:<4d}={rec/n:5.3f} {chg/n:9.3f} {cc/(chg or 1):16.3f}")

    if args.pair:
        print("\n=== paired exact McNemar (A-only / B-only / p) ===")
        for spec in args.pair:
            a, b = spec.split(":")
            ids = [i for i in per if a in per[i] and b in per[i]]
            a_only = sum(per[i][a]["correct"] and not per[i][b]["correct"] for i in ids)
            b_only = sum(per[i][b]["correct"] and not per[i][a]["correct"] for i in ids)
            p = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else float("nan")
            ra = sum(per[i][a]["correct"] for i in ids) / len(ids); rb = sum(per[i][b]["correct"] for i in ids) / len(ids)
            print(f"{a:20s} vs {b:16s} n={len(ids):3d}  {ra:.3f} vs {rb:.3f}  {a_only:3d}/{b_only:3d}  p={p:.3g}")

    if args.by_benchmark:
        print("\n=== recovery by benchmark x condition (common traces) ===")
        bench = sorted({per[i][conds[0]]["benchmark"] for i in common})
        print(f"{'benchmark':12s} {'n':>4s} " + " ".join(f"{c[:16]:>16s}" for c in conds))
        for b in bench:
            ids = [i for i in common if per[i][conds[0]]["benchmark"] == b]
            print(f"{b:12s} {len(ids):4d} " + " ".join(f"{sum(per[i][c]['correct'] for i in ids)/len(ids):16.3f}" for c in conds))


if __name__ == "__main__":
    main()
