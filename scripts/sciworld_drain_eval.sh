#!/bin/bash
# Oct 10 13:35 (coordinator): help a RUNNING 848 eval whose remaining shards wait for a GPU, using GPUs its (already loaded) picker cannot see
# (a7b_gpu jobs). For every shard whose log ends with "### attempt" (waiting in S), run the same collect command on such a GPU, writing to
# the SAME output file (collect skips finished ids; the waiting loop later finds the shard done).   bash scripts/sciworld_drain_eval.sh DIR ADAPTER TAG
source scripts/lib_sciworld_stepfix.sh; D=$1; AD=$2; TG=$3; export HOLD_NAME_RE='^a7b_gpu$'
for lg in $D/eval_test_s*.log $D/eval_dev_s*.log; do
  tail -n 1 $lg | grep -q "^### attempt" || continue
  b=$(basename $lg .log); sp=${b#eval_}; sp=${sp%_s*}; i=${b##*_s}
  if [ $sp = test ]; then A="--split test --per_task 30 --shard $i 12"; else A="--split dev --per_task 7 --shard $i 4"; fi
  H=$(pick 9000) || { echo "no a7b GPU free for $b"; continue; }; set -- $H
  echo "drain $b on $1 $2"; echo "### drain helper $(date '+%F %T') on $1" >> $D/drain_$b.log
  srun --jobid=$1 --overlap -n1 -c4 --mem=0 -w $2 $P -u scripts/sciworld_agent.py collect --adapter $AD $A --budget 50 --temperature 0 --seed 0 --snap safe --out $D/$b.jsonl --tag ${TG}_$sp >> $D/drain_$b.log 2>&1 &
  sleep 40; done; wait
