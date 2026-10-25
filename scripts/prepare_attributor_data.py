"""Build the SFT dataset for the specialized attributor (Research Plan Sec.
9.C) from the official Who&When Pro release (text split).

Split strategy:
  - `alfworld` benchmark is held out ENTIRELY as a transfer-test set, never
    seen during attributor training. This isn't just a train/val
    formality: it's the natural substrate for RQ3 (does attribution learned
    elsewhere transfer to an unseen agent/benchmark?) since this project's
    own ALFWorld environment + oracle attribution work (Phase 1) lives
    right next to it -- we can directly compare this trained attributor's
    accuracy against the oracle on the same ALFWorld trajectories.
  - Everything else (7 frameworks, 22 benchmark/framework pairs) is split
    90/10 train/val, stratified by framework so small frameworks (e.g.
    metagpt, 81 rows) still get representation in both splits.

Output: data/attributor_sft/{train,val,transfer_alfworld}.jsonl, each row
{id, framework, benchmark, prompt, target}.
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from whowhen_eval.prompts import load_taxonomy  # noqa: E402

from error_attr.attribution.sft_format import UnattributableRow, build_sft_example  # noqa: E402

DATA_ROOT = ROOT / "data" / "whowhen_pro_official"
OUT_DIR = ROOT / "data" / "attributor_sft"
HOLD_OUT_BENCHMARKS = {"alfworld"}
VAL_FRACTION = 0.10
SEED = 0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full", help="trace-input variant (see error_attr.attribution.ablation.VARIANTS); "
                    "builds SFT data whose PROMPTS are ablated the same way the inference-time ablation does")
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    taxonomy = load_taxonomy(DATA_ROOT)
    if args.variant != "full":
        from whowhen_eval.render import get_renderer
        from whowhen_eval.run import release_from_row
        from error_attr.attribution.ablation import build_prompt
    rows_by_framework = defaultdict(list)
    n_unattributable = 0

    with open(DATA_ROOT / "data" / "text.jsonl") as f:
        for line in f:
            row = json.loads(line)
            rows_by_framework[row["framework"]].append(row)

    train_examples, val_examples, transfer_examples = [], [], []
    rng = random.Random(SEED)

    for framework, rows in rows_by_framework.items():
        held_out_rows = [r for r in rows if r["benchmark"] in HOLD_OUT_BENCHMARKS]
        trainable_rows = [r for r in rows if r["benchmark"] not in HOLD_OUT_BENCHMARKS]

        rng.shuffle(trainable_rows)
        n_val = max(1, int(len(trainable_rows) * VAL_FRACTION)) if trainable_rows else 0
        val_rows = trainable_rows[:n_val]
        train_rows = trainable_rows[n_val:]

        for rows_split, bucket in ((train_rows, train_examples), (val_rows, val_examples), (held_out_rows, transfer_examples)):
            for row in rows_split:
                try:
                    ex = build_sft_example(row, "text", DATA_ROOT, taxonomy)
                except UnattributableRow:
                    n_unattributable += 1
                    continue
                if args.variant != "full":
                    rel = release_from_row(row, "text", DATA_ROOT)
                    rr = get_renderer(rel["framework"])(rel)
                    ex["prompt"], ex["compression"], _ = build_prompt(args.variant, rel, rr, taxonomy, SEED)
                    ex["variant"] = args.variant
                bucket.append(ex)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, examples in (("train", train_examples), ("val", val_examples), ("transfer_alfworld", transfer_examples)):
        path = out_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"{name}: {len(examples)} examples -> {path}")

    print(f"unattributable (skipped): {n_unattributable}")


if __name__ == "__main__":
    main()
