#!/bin/bash
# Oct 10 22:20: faster 848 eval for x2 (user asked to speed it up; GPUs were free): 32 test + 8 dev shards instead of 12 + 4
# (same episodes: test per_task 30 seed 0, dev per_task 7 seed 0, greedy, budget 50, --snap safe).  Replaces the eval stage of
# run_sciworld_stepfix_x1500.sh for x2 (that driver was stopped after x2 training; the training process kept running).
source scripts/lib_sciworld_stepfix.sh; export GMEM=9000; X1=$SF/x1; X2=$SF/x2; F2=$SF/x2rft; E=$X2/eval; mkdir -p $E
# the x2 training process died with the old driver at row ~352/429 -> resume it here (trainer checkpoints; same args as trainit())
until [ -f $X2/model/adapter/adapter_config.json ]; do echo "### attempt $(date '+%F %T') (resume)" >> $X2/train.log
  $(SG 20000) $P -u scripts/sciworld_stepfix.py train --data $X2/targets_env_top.jsonl $X2/replay_fw.jsonl --adapter $X1/model/adapter --out $X2/model --seed 2 >> $X2/train.log 2>&1 || sleep 30
  for k in $(seq 30); do [ -f $X2/model/adapter/adapter_config.json ] && break; sleep 2; done; done
slog "x2: trained on $(cat $X2/targets_env_top.jsonl $X2/replay_fw.jsonl | grep -c .) rows ($(tail -1 $X2/model/train_log.jsonl)); wide eval 40 shards"
for i in $(seq 0 31); do shard_loop $E/eval_test_w$i.log scripts/sciworld_agent.py collect --adapter $X2/model/adapter --split test --per_task 30 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 32 --out $E/eval_test_w$i.jsonl --tag x2_test & sleep 5; done
for i in $(seq 0 7); do shard_loop $E/eval_dev_w$i.log scripts/sciworld_agent.py collect --adapter $X2/model/adapter --split dev --per_task 7 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 8 --out $E/eval_dev_w$i.jsonl --tag x2_dev & sleep 5; done
wait; cat $E/eval_test_w*.jsonl > $E/eval_test.jsonl; cat $E/eval_dev_w*.jsonl > $E/eval_dev.jsonl; slog "x2: 848 eval DONE"
for rf in "R0=$SF/r0/eval_test.jsonl,reports/sciworld/refix/eval_dev_safe.jsonl" "x1=$X1/eval/eval_test.jsonl,$X1/eval/eval_dev.jsonl" "r2h=$SF/r2h/eval/eval_test.jsonl,$SF/r2h/eval/eval_dev.jsonl" "rftfull_r2=reports/sciworld/baselines/rftfull/r2/eval/eval_test.jsonl,reports/sciworld/baselines/rftfull/r2/eval/eval_dev.jsonl" "x1rft=$SF/x1rft/eval/eval_test.jsonl,$SF/x1rft/eval/eval_dev.jsonl"; do
  $P scripts/sciworld_stepfix_compare.py --a $rf --b x2=$E/eval_test.jsonl,$E/eval_dev.jsonl 2>/dev/null | head -1 | tee -a $SF/queue.log; done
slog "x2 DONE (x1500 StepFix chain complete)"
