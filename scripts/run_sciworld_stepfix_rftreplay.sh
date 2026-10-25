#!/bin/bash
# Oct 10 18:15 (user-approved, top priority): corrections + STANDARD-RFT replay (all success steps, exact duplicates dropped, as rftfull).
#   r2hR : r1b adapter + r2h env_top (63) + rftfull-style success rows of r2's collection (3621)
#   r2hRw: same, corrections weighted 3621/63 = 57.48 (trainer path:w; total correction weight = total success-row weight)
#   r1bR : student10 + r1b env_top (62) + rftfull R1 rows (R0 collection, 4008 rows; reports/sciworld/baselines/rftfull/r1/train_rows.jsonl)
source scripts/lib_sciworld_stepfix.sh; export GMEM=9000
arm() { local N=$1 AD=$2; shift 2; local D=$SF/$N; mkdir -p $D
  until [ -f $D/model/adapter/adapter_config.json ]; do echo "### attempt $(date '+%F %T')" >> $D/train.log
    wait_free "train --data .* --out $D/model "; $(S 20000) $P -u scripts/sciworld_stepfix.py train --data "$@" --adapter $AD --out $D/model --seed 2 >> $D/train.log 2>&1 || { slog "$N: train attempt failed (retry)"; sleep 60; }
    for k in $(seq 30); do [ -f $D/model/adapter/adapter_config.json ] && break; sleep 2; done; done
  slog "$N: trained ($(tail -1 $D/model/train_log.jsonl))"
  eval_sw $D/eval $D/model/adapter $N; slog "$N: 848 eval DONE"; slog "$N DONE"; }
W=$($P -c "print(round($(grep -c . $SF/r2hR/success_rows.jsonl) / $(grep -c . $SF/r2h/targets_env_top.jsonl), 4))")
slog "rftreplay: start r2hR / r2hRw (correction weight $W, via trainer path:w) / r1bR"
arm r2hR $SF/r1b/stepfix/adapter $SF/r2h/targets_env_top.jsonl $SF/r2hR/success_rows.jsonl &
sleep 60; arm r2hRw $SF/r1b/stepfix/adapter $SF/r2h/targets_env_top.jsonl:$W $SF/r2hR/success_rows.jsonl &
sleep 60; arm r1bR $STU0 $SF/r1b/targets_env_top.jsonl reports/sciworld/baselines/rftfull/r1/train_rows.jsonl &
wait; slog "rftreplay: all three done"
