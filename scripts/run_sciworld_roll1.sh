#!/bin/bash
# Oct 11 11:05 (user: rollout verification on ScienceWorld is required): ALFWorld rollout-verification baseline, one round, on the final
# recipe's R1 inputs: w1/z1 collection + Model A pointer + top-4 teacher rows (R0 teacher). Every candidate is rolled out to the episode end
# by the round's start policy (R0); K = round(0.2 x failures) random states among those with >=1 winning candidate (all if fewer); target =
# the winning candidate with the higher teacher probability (sciworld_stepfix.py rollout / select_rollout) + the same retention replay (418).
source scripts/lib_sciworld_stepfix.sh; export GMEM=9000; D=$SF/roll1; E=$D/eval; mkdir -p $E
[ -s $D/pointer.jsonl ] || { cp $SF/z1/{traj.jsonl,ids_fail.jsonl,traj.release.jsonl,pointer.jsonl,pointer_stats.json} $D/; mkdir -p $D/teacher; cp $SF/z1/teacher/rows_s*.jsonl $D/teacher/; }
slog "roll1: start (rollout verification of z1 teacher rows with R0)"
if ! grep -q "DONE rollout (merged)" $D/rollout.log 2>/dev/null; then pids=""
  for i in $(seq 0 47); do ( until grep -q "DONE rollout" $D/rollout_s$i.log 2>/dev/null; do $(SG $GMEM) $P -u scripts/sciworld_stepfix.py rollout --dir $D --adapter $STU0 --shard $i 48 >> $D/rollout_s$i.log 2>&1 || sleep 30; done ) & pids="$pids $!"; sleep 6; done; wait $pids
  echo "DONE rollout (merged)" >> $D/rollout.log; fi
$P scripts/sciworld_stepfix.py select_rollout --dir $D --round 1 | sed "s/^/roll1 /" | tee -a $SF/queue.log
until [ -f $D/model/adapter/adapter_config.json ]; do $(SG 20000) $P -u scripts/sciworld_stepfix.py train --data $D/targets_roll.jsonl $SF/w1/replay_ret.jsonl --adapter $STU0 --out $D/model --seed 1 >> $D/train.log 2>&1 || sleep 30; done
slog "roll1: trained ($(tail -1 $D/model/train_log.jsonl))"
pids=""; for i in $(seq 0 31); do shard_loop $E/eval_test_w$i.log scripts/sciworld_agent.py collect --adapter $D/model/adapter --split test --per_task 30 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 32 --out $E/eval_test_w$i.jsonl --tag roll1_test & pids="$pids $!"; sleep 5; done
for i in $(seq 0 7); do shard_loop $E/eval_dev_w$i.log scripts/sciworld_agent.py collect --adapter $D/model/adapter --split dev --per_task 7 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 8 --out $E/eval_dev_w$i.jsonl --tag roll1_dev & pids="$pids $!"; sleep 5; done
wait $pids; cat $E/eval_test_w*.jsonl > $E/eval_test.jsonl; cat $E/eval_dev_w*.jsonl > $E/eval_dev.jsonl; slog "roll1: 848 eval DONE"
for rf in R0=$SF/r0/eval_test.jsonl,reports/sciworld/refix/eval_dev_safe.jsonl w1=$SF/w1/eval/eval_test.jsonl,$SF/w1/eval/eval_dev.jsonl; do $P scripts/sciworld_stepfix_compare.py --a $rf --b roll1=$E/eval_test.jsonl,$E/eval_dev.jsonl | head -1 | tee -a $SF/queue.log; done
slog "roll1 DONE"
