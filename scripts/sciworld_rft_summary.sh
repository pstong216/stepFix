#!/bin/bash
# Summary of the ScienceWorld RFT baselines (Oct 10 2026): per baseline round, 848 wins / mean score (negative clipped to 0) / McNemar vs R0
# and vs our same-round StepFix model (R1 = r1b/eval_stepfix; R2/R3 = r2f/eval and r3f/eval, plus the older r2/r3 eval_stepfix for reference).
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; SF=reports/sciworld/stepfix; BL=reports/sciworld/baselines
R0=$SF/r0/eval_test.jsonl,reports/sciworld/refix/eval_dev_safe.jsonl
ok() { [ "$(cat $1/eval_test.jsonl $1/eval_dev.jsonl 2>/dev/null | grep -c .)" = 848 ]; }
for M in rftm rftfull; do for R in 1 2 3; do D=$BL/$M/r$R/eval; ok $D || { echo "## ${M}_r$R: eval not complete"; continue; }
  echo "## ${M}_r$R ($(grep -c . $BL/$M/r$R/train_rows.jsonl) training rows)"
  $P scripts/sciworld_stepfix_compare.py --a R0=$R0 --b ${M}_r$R=$D/eval_test.jsonl,$D/eval_dev.jsonl | grep "^\[848"
  if [ $R = 1 ]; then OURS="$SF/r1b/eval_stepfix"; else OURS="$SF/r${R}f/eval $SF/r$R/eval_stepfix"; fi
  for O in $OURS; do ok $O && $P scripts/sciworld_stepfix_compare.py --a ours_$(basename $(dirname $O))=$O/eval_test.jsonl,$O/eval_dev.jsonl --b ${M}_r$R=$D/eval_test.jsonl,$D/eval_dev.jsonl | grep "^\[848"; done
done; done
$P - <<'EOF'
import json, glob, os
def load(fs):
    r = {}
    for f in fs:
        for l in open(f):
            try: e = json.loads(l)
            except Exception: continue
            r[(e["split"], e["task"], e["variation"])] = e
    return r
S, B = "reports/sciworld/stepfix", "reports/sciworld/baselines"
cands = [("R0", [f"{S}/r0/eval_test.jsonl", "reports/sciworld/refix/eval_dev_safe.jsonl"]), ("stepfix_r1b", [f"{S}/r1b/eval_stepfix/eval_test.jsonl", f"{S}/r1b/eval_stepfix/eval_dev.jsonl"])]
cands += [(f"stepfix_r{r}f", [f"{S}/r{r}f/eval/eval_test.jsonl", f"{S}/r{r}f/eval/eval_dev.jsonl"]) for r in (2, 3)]
cands += [(f"{m}_r{r}", [f"{B}/{m}/r{r}/eval/eval_test.jsonl", f"{B}/{m}/r{r}/eval/eval_dev.jsonl"]) for m in ("rftm", "rftfull") for r in (1, 2, 3)]
print("\nmodel            won/848  mean_score  score<0  failed_before_budget")
for n, fs in cands:
    if not all(os.path.exists(f) for f in fs): continue
    A = load(fs)
    if len(A) != 848: print(f"{n:16s} incomplete ({len(A)})"); continue
    print(f"{n:16s} {sum(e['won'] for e in A.values()):7d}  {sum(max(e['final_score'], 0) for e in A.values()) / 848:10.1f}  {sum(e['final_score'] < 0 for e in A.values()):7d}  {sum((not e['won']) and e['n_steps'] < e['budget'] for e in A.values()):7d}")
EOF
