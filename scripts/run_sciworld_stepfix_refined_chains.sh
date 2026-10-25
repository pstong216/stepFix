#!/bin/bash
# Oct 10 02:50 chains (user decisions 02:45/02:55): main r2f -> r3f (+ long-budget eval of R0 / r1b / r2f after the r2f eval);
# ablations A (random selection, R2-R3 from the r1b random adapter) and B (no replay, R1-R3) at lower GPU priority.
#   bash scripts/run_sciworld_stepfix_refined_chains.sh main|ablA|ablB
source scripts/lib_sciworld_stepfix.sh; W=$1; RF=scripts/run_sciworld_stepfix_refined.sh
wait_adapter() { until [ -f $1/adapter_config.json ]; do sleep 60; done; }
if [ "$W" = main ]; then
  GMEM=9000 bash $RF r2f 2 $SF/r1b/stepfix/adapter stepfix 1 $SF/r2 "r1b=$SF/r1b/eval_stepfix r2=$SF/r2/eval_stepfix" > $SF/r2f_driver.log 2>&1 &
  wait_adapter $SF/r2f/model/adapter
  GMEM=9000 bash $RF r3f 3 $SF/r2f/model/adapter stepfix 1 "" "r2f=$SF/r2f/eval r3=$SF/r3/eval_stepfix r1b=$SF/r1b/eval_stepfix" > $SF/r3f_driver.log 2>&1 &
  until grep -q "r2f DONE" $SF/queue.log; do sleep 60; done
  for m in "R0 $STU0" "r1b $SF/r1b/stepfix/adapter" "r2f $SF/r2f/model/adapter"; do set -- $m; GMEM=12000 bash scripts/run_sciworld_longbudget_eval.sh $1 $2 > $SF/longbudget_$1.log 2>&1 & sleep 60; done
  wait
elif [ "$W" = ablA ]; then
  export GMEM=16000 TGMEM=18000 TRMEM=26000
  bash $RF abl_rand_r2 2 $SF/r1b/random/adapter random 1 "" "r1b_random=$SF/r1b/eval_random r2f=$SF/r2f/eval" > $SF/abl_rand_r2_driver.log 2>&1 &
  wait_adapter $SF/abl_rand_r2/model/adapter
  bash $RF abl_rand_r3 3 $SF/abl_rand_r2/model/adapter random 1 "" "abl_rand_r2=$SF/abl_rand_r2/eval r3f=$SF/r3f/eval" > $SF/abl_rand_r3_driver.log 2>&1 &
  wait
elif [ "$W" = ablB ]; then
  export GMEM=16000 TGMEM=18000 TRMEM=26000; D=$SF/abl_norep_r1; mkdir -p $D
  slog "abl_norep_r1: start (student10 + r1b env_top corrections only, no replay)"
  until [ -f $D/model/adapter/adapter_config.json ]; do echo "### attempt $(date '+%F %T')" >> $D/train.log
    wait_free "train --data .* --out $D/model "; $(SG $TRMEM) $P -u scripts/sciworld_stepfix.py train --data $SF/r1b/targets_env_top.jsonl --adapter $STU0 --out $D/model --seed 1 >> $D/train.log 2>&1 || sleep 60
    for k in $(seq 30); do [ -f $D/model/adapter/adapter_config.json ] && break; sleep 2; done; done
  slog "abl_norep_r1: trained ($(tail -1 $D/model/train_log.jsonl))"
  ( eval_sw $D/eval $D/model/adapter abl_norep_r1; slog "abl_norep_r1: 848 eval DONE"
    $P scripts/sciworld_stepfix_compare.py --a R0=$SF/r0/eval_test.jsonl,reports/sciworld/refix/eval_dev_safe.jsonl --b abl_norep_r1=$D/eval/eval_test.jsonl,$D/eval/eval_dev.jsonl | tee -a $SF/queue.log
    $P scripts/sciworld_stepfix_compare.py --a r1b=$SF/r1b/eval_stepfix/eval_test.jsonl,$SF/r1b/eval_stepfix/eval_dev.jsonl --b abl_norep_r1=$D/eval/eval_test.jsonl,$D/eval/eval_dev.jsonl | head -1 | tee -a $SF/queue.log ) &
  bash $RF abl_norep_r2 2 $D/model/adapter stepfix 0 "" "r2f=$SF/r2f/eval" > $SF/abl_norep_r2_driver.log 2>&1 &
  wait_adapter $SF/abl_norep_r2/model/adapter
  bash $RF abl_norep_r3 3 $SF/abl_norep_r2/model/adapter stepfix 0 "" "r3f=$SF/r3f/eval abl_norep_r2=$SF/abl_norep_r2/eval" > $SF/abl_norep_r3_driver.log 2>&1 &
  wait
fi
slog "chains $W: finished"
