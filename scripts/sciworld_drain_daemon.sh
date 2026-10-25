#!/bin/bash
# Oct 10 13:40 (coordinator 13:30): drivers started before the a7b_gpu jobs were allowed cannot see them (their picker regex is loaded).
# Every 90 s: for each listed stage whose shard log ends with "### attempt" (waiting for a GPU) and that has no running process for the same
# output, launch the identical command on a free a7b_gpu job (never 3232903 = 7B filler on agc04). Same output file => collect / rollout skip
# finished items, and the waiting loop later finds its shard done. Eval shards and rollout shards are handled.
#   bash scripts/sciworld_drain_daemon.sh "abl_unf_r2 abl_roll_r2 r3h r3k"
source scripts/lib_sciworld_stepfix.sh; export HOLD_NAME_RE='^a7b_gpu$'; ROUNDS=$1
adapter_of() { case $1 in r3h|r3k|abl_unf_r2|abl_roll_r2|*) echo $SF/$1/model/adapter;; esac; }
start_of() { grep -o "from [^ ,]*" $SF/queue.log | grep -m1 -A0 "" >/dev/null; sed -n "s/.*=== .* $1: start refined round (R=[0-9] from \([^,]*\),.*/\1/p" $SF/queue.log | tail -1; }
while true; do
  for r in $ROUNDS; do D=$SF/$r
    for lg in $D/eval/eval_test_s*.log $D/eval/eval_dev_s*.log; do [ -f $lg ] || continue
      tail -n 1 $lg | grep -q "^### attempt" || continue; [ $(( $(date +%s) - $(stat -c %Y $lg) )) -gt 90 ] || continue
      b=$(basename $lg .log); sp=${b#eval_}; sp=${sp%_s*}; i=${b##*_s}; out=$D/eval/$b.jsonl
      pgrep -f -- "--out $out " > /dev/null && continue
      [ $sp = test ] && A="--split test --per_task 30 --shard $i 12" || A="--split dev --per_task 7 --shard $i 4"
      H=$(pick 9000) || break; set -- $H; echo "=== $(date '+%F %T') drain $r $b on $1 $2"
      srun --jobid=$1 --overlap -n1 -c4 --mem=0 -w $2 $P -u scripts/sciworld_agent.py collect --adapter $D/model/adapter $A --budget 50 --temperature 0 --seed 0 --snap safe --out $out --tag ${r}_$sp >> $D/eval/drain_$b.log 2>&1 &
      sleep 40; done
    for lg in $D/rollout_s*.log; do [ -f $lg ] || continue
      tail -n 1 $lg | grep -q "^### attempt" || continue; [ $(( $(date +%s) - $(stat -c %Y $lg) )) -gt 90 ] || continue
      i=$(basename $lg .log); i=${i#rollout_s}; pgrep -f -- "rollout --dir $D .*--shard $i 12" > /dev/null && continue
      ST=$(start_of $r); [ -n "$ST" ] || continue
      H=$(pick 12000) || break; set -- $H; echo "=== $(date '+%F %T') drain $r rollout_s$i on $1 $2 (policy $ST)"
      srun --jobid=$1 --overlap -n1 -c4 --mem=0 -w $2 $P -u scripts/sciworld_stepfix.py rollout --dir $D --adapter $ST --shard $i 12 >> $D/drain_rollout_s$i.log 2>&1 &
      sleep 40; done
  done; sleep 90; done
