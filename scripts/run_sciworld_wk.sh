#!/bin/bash
# Oct 11 04:10 generalized: bash run_sciworld_wk.sh NAME SRC_ROUND START_ADAPTER SEED (w3k = w3 + KL; w2k = w2 + KL). Oct 11 04:00 (user-approved): w3k = w3 with a KL anchor on replay rows: identical corrections (w3/targets_cap.jsonl) and retention replay
# (w3/replay_ret.jsonl), start = w2 adapter, seed 3; replay rows use CE + 10 x KL(p_w2 || p_theta) over every action token (sciworld_stepfix.py
# train --kl_replay); corrections unchanged.  848 eval with 32 test + 8 dev shards.
source scripts/lib_sciworld_stepfix.sh; export GMEM=9000; W3=$SF/$2; D=$SF/$1; E=$D/eval; START=$3; SEED=$4; NM=$1; mkdir -p $E
until [ -s $W3/targets_cap.jsonl ] && [ -s $W3/replay_ret.jsonl ]; do sleep 30; done; sleep 10
cp $W3/targets_cap.jsonl $W3/replay_ret.jsonl $D/; slog "$NM: start (corrections $(grep -c . $D/targets_cap.jsonl), replay $(grep -c . $D/replay_ret.jsonl), KL beta 10)"
until [ -f $D/model/adapter/adapter_config.json ]; do echo "### attempt $(date '+%F %T')" >> $D/train.log
  $(SG 24000) $P -u scripts/sciworld_stepfix.py train --data $D/targets_cap.jsonl $D/replay_ret.jsonl --adapter $START --out $D/model --seed $SEED --kl_replay >> $D/train.log 2>&1 || sleep 30
  for k in $(seq 30); do [ -f $D/model/adapter/adapter_config.json ] && break; sleep 2; done; done
slog "$NM: trained ($(tail -1 $D/model/train_log.jsonl))"
pids=""; for i in $(seq 0 31); do shard_loop $E/eval_test_w$i.log scripts/sciworld_agent.py collect --adapter $D/model/adapter --split test --per_task 30 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 32 --out $E/eval_test_w$i.jsonl --tag ${NM}_test & pids="$pids $!"; sleep 5; done
for i in $(seq 0 7); do shard_loop $E/eval_dev_w$i.log scripts/sciworld_agent.py collect --adapter $D/model/adapter --split dev --per_task 7 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 8 --out $E/eval_dev_w$i.jsonl --tag ${NM}_dev & pids="$pids $!"; sleep 5; done
wait $pids; cat $E/eval_test_w*.jsonl > $E/eval_test.jsonl; cat $E/eval_dev_w*.jsonl > $E/eval_dev.jsonl; slog "$NM: 848 eval DONE"
for rf in R0=$SF/r0/eval_test.jsonl,reports/sciworld/refix/eval_dev_safe.jsonl w1=$SF/w1/eval/eval_test.jsonl,$SF/w1/eval/eval_dev.jsonl w2=$SF/w2/eval/eval_test.jsonl,$SF/w2/eval/eval_dev.jsonl w3=$SF/w3/eval/eval_test.jsonl,$SF/w3/eval/eval_dev.jsonl x1rft=$SF/x1rft/eval/eval_test.jsonl,$SF/x1rft/eval/eval_dev.jsonl; do
  [ -s $(echo ${rf#*=} | cut -d, -f1) ] && $P scripts/sciworld_stepfix_compare.py --a $rf --b $NM=$E/eval_test.jsonl,$E/eval_dev.jsonl | head -1 | tee -a $SF/queue.log; done
slog "$NM DONE"
