"""Compare attributor variants on the SAME evaluation rows: per-axis accuracy
(agent / step / mode / all-3) plus pairwise exact McNemar tests against a
reference, using the per-example files written by scripts/eval_attributor.py
(--out). All files must cover the same split; rows are matched by id, so
a `--limit`ed file is compared on its own rows only.

Usage:
    python scripts/compare_attributors.py --split val \
        --model SFT=reports/eval_val_finetuned.jsonl \
        --model untuned=reports/eval_val_base_untuned.jsonl \
        --model GRPO-base=reports/eval_val_grpo_basestart_final.jsonl \
        --ref SFT --ref untuned [--by_framework]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
AXES = ("agent", "step", "mode")


def load(path):
    return {json.loads(l)["id"]: json.loads(l) for l in open(path)}


def acc_table(name, rows, ids):
    n = len(ids)
    out = {ax: sum(rows[i]["result"][ax] for i in ids) / n for ax in AXES}
    out["all3"] = sum(all(rows[i]["result"].values()) for i in ids) / n
    return out


def mcnemar(a_rows, b_rows, ids, ax):
    def ok(r):
        return all(r["result"].values()) if ax == "all3" else r["result"][ax]
    a_only = sum(ok(a_rows[i]) and not ok(b_rows[i]) for i in ids)
    b_only = sum(ok(b_rows[i]) and not ok(a_rows[i]) for i in ids)
    p = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else float("nan")
    return a_only, b_only, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["val", "transfer_alfworld"])
    ap.add_argument("--model", action="append", required=True, help="NAME=path/to/per_example.jsonl (repeatable)")
    ap.add_argument("--ref", action="append", default=[], help="model NAME to test the others against (repeatable)")
    ap.add_argument("--by_framework", action="store_true")
    args = ap.parse_args()

    models = {}
    for spec in args.model:
        name, path = spec.split("=", 1)
        models[name] = load(path)
    ids = set.intersection(*(set(r) for r in models.values()))
    fw_of = {json.loads(l)["id"]: json.loads(l)["framework"] for l in open(ROOT / "data" / "attributor_sft" / f"{args.split}.jsonl")}
    ids = sorted(i for i in ids if i in fw_of)
    print(f"split={args.split} | common rows={len(ids)}\n")

    print(f"{'model':14s} {'agent':>7s} {'step':>7s} {'mode':>7s} {'all-3':>7s}")
    for name, rows in models.items():
        t = acc_table(name, rows, ids)
        print(f"{name:14s} {t['agent']:7.3f} {t['step']:7.3f} {t['mode']:7.3f} {t['all3']:7.3f}")

    for ref in args.ref:
        print(f"\n=== paired exact McNemar vs {ref} (X-only / {ref}-only / p) ===")
        for name, rows in models.items():
            if name == ref:
                continue
            cells = []
            for ax in AXES + ("all3",):
                a, b, p = mcnemar(rows, models[ref], ids, ax)
                cells.append(f"{ax}:{a:3d}/{b:3d} p={p:.2g}")
            print(f"{name:14s} " + " | ".join(cells))

    if args.by_framework:
        print("\n=== per-framework accuracy (agent/step/mode) ===")
        by_fw = defaultdict(list)
        for i in ids:
            by_fw[fw_of[i]].append(i)
        header = f"{'framework':13s} {'n':>4s} | " + " | ".join(f"{m:>17s}" for m in models)
        print(header)
        for fw, fids in sorted(by_fw.items(), key=lambda kv: -len(kv[1])):
            cells = []
            for name, rows in models.items():
                t = acc_table(name, rows, fids)
                cells.append(f"{t['agent']:.2f} {t['step']:.2f} {t['mode']:.2f}".rjust(17))
            print(f"{fw:13s} {len(fids):4d} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
