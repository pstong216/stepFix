#!/bin/bash
# Oct 10 03:35: ablations C (unfiltered) and D (rollout verification 20%), lowest priority (user/coordinator 03:20).
#   LOWPRIO=1 bash scripts/run_sciworld_stepfix_ablCD.sh ablC|ablD
source scripts/lib_sciworld_stepfix.sh; W=$1; RF=scripts/run_sciworld_stepfix_refined.sh
wait_adapter() { until [ -f $1/adapter_config.json ]; do sleep 60; done; }
# Oct 10 04:10 restart: wait until no orphaned C/D GPU process from the previous driver is still running (they finish their shard and write to the same files)
while ps -eo args | grep "^srun --jobid" | grep -q "stepfix/abl_unf\|stepfix/abl_roll"; do sleep 30; done
if false; then :
elif [ "$W" = ablC ]; then   # (C) UNFILTERED: teacher top-1 at every attributed failure, no evidence, no K cap, + the same replay
  export GMEM=16000 TGMEM=18000 TRMEM=26000; D=$SF/abl_unf_r1; mkdir -p $D
  slog "abl_unf_r1: start (student10 + r1b teacher top-1 at all $(grep -c . $SF/r1b/targets_all_top1.jsonl) states + r1b replay)"
  until [ -f $D/model/adapter/adapter_config.json ]; do echo "### attempt $(date '+%F %T')" >> $D/train.log
    wait_free "train --data .* --out $D/model "; $(SG $TRMEM) $P -u scripts/sciworld_stepfix.py train --data $SF/r1b/targets_all_top1.jsonl $SF/r1b/replay.jsonl --adapter $STU0 --out $D/model --seed 1 >> $D/train.log 2>&1 || sleep 60
    for k in $(seq 30); do [ -f $D/model/adapter/adapter_config.json ] && break; sleep 2; done; done
  slog "abl_unf_r1: trained ($(tail -1 $D/model/train_log.jsonl))"
  ( eval_sw $D/eval $D/model/adapter abl_unf_r1; slog "abl_unf_r1: 848 eval DONE"
    $P scripts/sciworld_stepfix_compare.py --a R0=$SF/r0/eval_test.jsonl,reports/sciworld/refix/eval_dev_safe.jsonl --b abl_unf_r1=$D/eval/eval_test.jsonl,$D/eval/eval_dev.jsonl | tee -a $SF/queue.log
    $P scripts/sciworld_stepfix_compare.py --a r1b=$SF/r1b/eval_stepfix/eval_test.jsonl,$SF/r1b/eval_stepfix/eval_dev.jsonl --b abl_unf_r1=$D/eval/eval_test.jsonl,$D/eval/eval_dev.jsonl | head -1 | tee -a $SF/queue.log ) &
  bash $RF abl_unf_r2 2 $D/model/adapter all_top1 1 "" "r2f=$SF/r2f/eval" > $SF/abl_unf_r2_driver.log 2>&1 &
  wait_adapter $SF/abl_unf_r2/model/adapter
  bash $RF abl_unf_r3 3 $SF/abl_unf_r2/model/adapter all_top1 1 "" "r3f=$SF/r3f/eval abl_unf_r2=$SF/abl_unf_r2/eval" > $SF/abl_unf_r3_driver.log 2>&1 &
  wait
elif [ "$W" = ablD ]; then   # (D) ROLLOUT-VERIFICATION 20%, R1 only: r1b states/candidates, greedy student10 continuation to the 50-step budget
  export GMEM=16000 TGMEM=18000 TRMEM=26000; D=$SF/abl_roll_r1; mkdir -p $D
  [ -s $D/traj.jsonl ] || { cp $SF/r1b/{traj.jsonl,ids_fail.jsonl,pointer.jsonl} $D/; ln -sfn ../r1b/teacher $D/teacher; }
  slog "abl_roll_r1: start rollout verification of the r1b candidates with student10 (12 shards)"
  for i in $(seq 0 11); do ( until grep -q "DONE rollout" $D/rollout_s$i.log 2>/dev/null; do echo "### attempt $(date '+%F %T')" >> $D/rollout_s$i.log
      wait_free "rollout --dir $D .*--shard $i 12"; $(SG $TGMEM) $P -u scripts/sciworld_stepfix.py rollout --dir $D --adapter $STU0 --shard $i 12 >> $D/rollout_s$i.log 2>&1 || sleep 30; done ) & sleep 15; done; wait
  $P scripts/sciworld_stepfix.py select_rollout --dir $D --round 1 | sed "s/^/abl_roll_r1 /" | tee -a $SF/queue.log
  until [ -f $D/model/adapter/adapter_config.json ]; do echo "### attempt $(date '+%F %T')" >> $D/train.log
    wait_free "train --data .* --out $D/model "; $(SG $TRMEM) $P -u scripts/sciworld_stepfix.py train --data $D/targets_roll.jsonl $SF/r1b/replay.jsonl --adapter $STU0 --out $D/model --seed 1 >> $D/train.log 2>&1 || sleep 60
    for k in $(seq 30); do [ -f $D/model/adapter/adapter_config.json ] && break; sleep 2; done; done
  slog "abl_roll_r1: trained ($(tail -1 $D/model/train_log.jsonl))"
  eval_sw $D/eval $D/model/adapter abl_roll_r1; slog "abl_roll_r1: 848 eval DONE"
  $P scripts/sciworld_stepfix_compare.py --a R0=$SF/r0/eval_test.jsonl,reports/sciworld/refix/eval_dev_safe.jsonl --b abl_roll_r1=$D/eval/eval_test.jsonl,$D/eval/eval_dev.jsonl | tee -a $SF/queue.log
  $P scripts/sciworld_stepfix_compare.py --a r1b=$SF/r1b/eval_stepfix/eval_test.jsonl,$SF/r1b/eval_stepfix/eval_dev.jsonl --b abl_roll_r1=$D/eval/eval_test.jsonl,$D/eval/eval_dev.jsonl | head -1 | tee -a $SF/queue.log
fi
slog "chains $W: finished"
