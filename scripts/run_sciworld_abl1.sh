#!/bin/bash
# Oct 11 10:50 (user: start the 3B ablations): ONE-round ablations of the final ScienceWorld recipe, all from R0 with w1's collection
# (1,483 variations), z1's top-4 teacher rows (R0 teacher) and w1's retention replay (418 = 2.4 x K, K = 174).
#   abl_rand : K random teacher top-1 targets (no evidence)   + replay   (sciworld_stepfix.py select -> z1/targets_rand.jsonl)
#   abl_norep: the w1 corrections (targets_cap)               , no replay
#   abl_nofil: teacher top-1 at EVERY attributed failure      + replay   (z1/targets_all_top1.jsonl)
# usage: bash run_sciworld_abl1.sh NAME DATA...   (train from student10, seed 1; 848 eval with 32 test + 8 dev shards)
source scripts/lib_sciworld_stepfix.sh; export GMEM=9000; NM=$1; shift; D=$SF/$NM; E=$D/eval; mkdir -p $E
slog "$NM: start (data: $* ; rows $(cat "$@" | grep -c .))"
until [ -f $D/model/adapter/adapter_config.json ]; do echo "### attempt $(date '+%F %T')" >> $D/train.log
  $(SG 20000) $P -u scripts/sciworld_stepfix.py train --data "$@" --adapter $STU0 --out $D/model --seed 1 >> $D/train.log 2>&1 || sleep 30
  for k in $(seq 30); do [ -f $D/model/adapter/adapter_config.json ] && break; sleep 2; done; done
slog "$NM: trained ($(tail -1 $D/model/train_log.jsonl))"
pids=""; for i in $(seq 0 31); do shard_loop $E/eval_test_w$i.log scripts/sciworld_agent.py collect --adapter $D/model/adapter --split test --per_task 30 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 32 --out $E/eval_test_w$i.jsonl --tag ${NM}_test & pids="$pids $!"; sleep 5; done
for i in $(seq 0 7); do shard_loop $E/eval_dev_w$i.log scripts/sciworld_agent.py collect --adapter $D/model/adapter --split dev --per_task 7 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 8 --out $E/eval_dev_w$i.jsonl --tag ${NM}_dev & pids="$pids $!"; sleep 5; done
wait $pids; cat $E/eval_test_w*.jsonl > $E/eval_test.jsonl; cat $E/eval_dev_w*.jsonl > $E/eval_dev.jsonl; slog "$NM: 848 eval DONE"
for rf in R0=$SF/r0/eval_test.jsonl,reports/sciworld/refix/eval_dev_safe.jsonl w1=$SF/w1/eval/eval_test.jsonl,$SF/w1/eval/eval_dev.jsonl; do $P scripts/sciworld_stepfix_compare.py --a $rf --b $NM=$E/eval_test.jsonl,$E/eval_dev.jsonl | head -1 | tee -a $SF/queue.log; done
slog "$NM DONE"
