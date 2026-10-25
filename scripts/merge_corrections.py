"""Splice a targeted re-run into an existing chain run: records in --patch
replace the base records with the same (id, condition); everything else is
kept. Used for the nested-brace parser fix (DECISIONS #25): only the 24
affected traces were re-run, the other 126 are byte-identical in intent
(greedy decoding, same prompts).

    python scripts/merge_corrections.py --base reports/X/corrections.jsonl \
        --patch reports/X_rerun24/corrections.jsonl --out reports/X/corrections.merged.jsonl
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--patch", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = [json.loads(l) for l in open(a.base)]
    patch = {(r["id"], r["condition"]): r for r in (json.loads(l) for l in open(a.patch))}
    n_rep = 0
    out = []
    for r in base:
        k = (r["id"], r["condition"])
        if k in patch:
            out.append(patch.pop(k)); n_rep += 1
        else:
            out.append(r)
    # Append only patch records whose CONDITION belongs to this run (a re-run may
    # cover several runs' conditions at once); other conditions stay out of this file.
    base_conds = {r["condition"] for r in base}
    extra = [r for r in patch.values() if r["condition"] in base_conds]
    out += extra
    patch = {k: v for k, v in patch.items() if v["condition"] in base_conds}
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"base {len(base)} records; replaced {n_rep}; appended {len(patch)}; -> {a.out} ({len(out)})")


if __name__ == "__main__":
    main()
