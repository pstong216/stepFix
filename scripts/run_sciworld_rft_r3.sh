#!/bin/bash
# Oct 11 21:50 (user: baselines to R3 budget): standard RFT round 3 on ScienceWorld at the same budget: collect the 1,483 variations
# (per_task 75, seed 1, T=1.0) with the RFT R2 adapter (x2rft), train on every step of every win (exact duplicates dropped), seed 3,
# from x2rft; 848 eval (32 test + 8 dev shards).
source scripts/lib_sciworld_stepfix.sh; export GMEM=9000; D=$SF/x3rft; E=$D/eval; mkdir -p $E; AD=$SF/x2rft/model/adapter
slog "x3rft: start"
[ -s $D/traj.jsonl ] || { collect_sw $D/col $AD 1 x3rft 24 --per_task 75; $P scripts/sciworld_baselines.py merge --dir $D --inputs "$D/col/traj_s*.jsonl" --expect 1483 | sed "s/^/x3rft /" | tee -a $SF/queue.log; }
[ -s $D/train_rows.jsonl ] || $P scripts/sciworld_baselines.py build --dir $D --mode rftfull --n 0 --seed 3 | sed "s/^/x3rft /" | tee -a $SF/queue.log
until [ -f $D/model/adapter/adapter_config.json ]; do echo "### attempt $(date '+%F %T')" >> $D/train.log
  $(SG 20000) $P -u scripts/sciworld_stepfix.py train --data $D/train_rows.jsonl --adapter $AD --out $D/model --seed 3 >> $D/train.log 2>&1 || sleep 30
  for k in $(seq 30); do [ -f $D/model/adapter/adapter_config.json ] && break; sleep 2; done; done
slog "x3rft: trained on $(grep -c . $D/train_rows.jsonl) rows"
pids=""; for i in $(seq 0 31); do shard_loop $E/eval_test_w$i.log scripts/sciworld_agent.py collect --adapter $D/model/adapter --split test --per_task 30 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 32 --out $E/eval_test_w$i.jsonl --tag x3rft_test & pids="$pids $!"; sleep 5; done
for i in $(seq 0 7); do shard_loop $E/eval_dev_w$i.log scripts/sciworld_agent.py collect --adapter $D/model/adapter --split dev --per_task 7 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 8 --out $E/eval_dev_w$i.jsonl --tag x3rft_dev & pids="$pids $!"; sleep 5; done
wait $pids; cat $E/eval_test_w*.jsonl > $E/eval_test.jsonl; cat $E/eval_dev_w*.jsonl > $E/eval_dev.jsonl; slog "x3rft: 848 eval DONE"
for rf in R0=$SF/r0/eval_test.jsonl,reports/sciworld/refix/eval_dev_safe.jsonl wk3=$SF/wk3/eval/eval_test.jsonl,$SF/wk3/eval/eval_dev.jsonl; do $P scripts/sciworld_stepfix_compare.py --a $rf --b x3rft=$E/eval_test.jsonl,$E/eval_dev.jsonl | head -1 | tee -a $SF/queue.log; done
slog "x3rft DONE"
