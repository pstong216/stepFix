#!/bin/bash
# Oct 10 23:00 (user-approved): R2 re-selection on x2's teacher rows (scripts/sciworld_select_capped.py): per-type cap on env_top
# corrections (2 x uniform share) + RETENTION replay (by training wins).  ARM x2c: replay 1.2 x K (ALFWorld ratio); ARM x2cr: 2.4 x K
# (user asked whether to raise replay; sensitivity).  Train from x1 (1 ep lr 1e-5 bs 8 seed 2); 848 eval with 32 test + 8 dev shards.
source scripts/lib_sciworld_stepfix.sh; export GMEM=9000; NM=$1; X1=$SF/x1; D=$SF/$NM; E=$D/eval; mkdir -p $E
slog "$NM: start (corrections $(grep -c . $D/targets_cap.jsonl), replay $(grep -c . $D/replay_ret.jsonl))"
until [ -f $D/model/adapter/adapter_config.json ]; do echo "### attempt $(date '+%F %T')" >> $D/train.log
  $(SG 20000) $P -u scripts/sciworld_stepfix.py train --data $D/targets_cap.jsonl $D/replay_ret.jsonl --adapter $X1/model/adapter --out $D/model --seed 2 >> $D/train.log 2>&1 || sleep 30
  for k in $(seq 30); do [ -f $D/model/adapter/adapter_config.json ] && break; sleep 2; done; done
slog "$NM: trained ($(tail -1 $D/model/train_log.jsonl))"
for i in $(seq 0 31); do shard_loop $E/eval_test_w$i.log scripts/sciworld_agent.py collect --adapter $D/model/adapter --split test --per_task 30 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 32 --out $E/eval_test_w$i.jsonl --tag ${NM}_test & sleep 5; done
for i in $(seq 0 7); do shard_loop $E/eval_dev_w$i.log scripts/sciworld_agent.py collect --adapter $D/model/adapter --split dev --per_task 7 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 8 --out $E/eval_dev_w$i.jsonl --tag ${NM}_dev & sleep 5; done
wait; cat $E/eval_test_w*.jsonl > $E/eval_test.jsonl; cat $E/eval_dev_w*.jsonl > $E/eval_dev.jsonl; slog "$NM: 848 eval DONE"
for rf in "R0=$SF/r0/eval_test.jsonl,reports/sciworld/refix/eval_dev_safe.jsonl" "x1=$X1/eval/eval_test.jsonl,$X1/eval/eval_dev.jsonl" "x2=$SF/x2/eval/eval_test.jsonl,$SF/x2/eval/eval_dev.jsonl" "x1rft=$SF/x1rft/eval/eval_test.jsonl,$SF/x1rft/eval/eval_dev.jsonl"; do
  $P scripts/sciworld_stepfix_compare.py --a $rf --b $NM=$E/eval_test.jsonl,$E/eval_dev.jsonl 2>/dev/null | head -1 | tee -a $SF/queue.log; done
slog "$NM DONE"
