#!/bin/bash
# R2 and R3 chain (Oct 9): R2 starts when the R1 evals are 80% done; R3 starts as soon as the R2 StepFix adapter exists (its collection overlaps R2's eval).
source scripts/lib_sciworld_stepfix.sh; D1=$SF/r1b   # Oct 9 23:45: main R1 = r1b (relaxed validity + refill)
until [ $(cat $D1/eval_stepfix/eval_*_s*.jsonl $D1/eval_random/eval_*_s*.jsonl 2>/dev/null | grep -c .) -ge 1357 ]; do sleep 60; done
slog "chain: r1b evals >= 80% done -> start R2 (from r1b)"
bash scripts/run_sciworld_stepfix_round.sh 2 $SF/r1b/stepfix/adapter 0 > $SF/r2_driver.log 2>&1 &
until [ -f $SF/r2/stepfix/adapter/adapter_config.json ]; do sleep 60; done
slog "chain: R2 adapter ready -> start R3"
bash scripts/run_sciworld_stepfix_round.sh 3 $SF/r2/stepfix/adapter 0 > $SF/r3_driver.log 2>&1 &
wait; slog "chain: R2 and R3 drivers finished"
