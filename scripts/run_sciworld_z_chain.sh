#!/bin/bash
# Oct 11 00:20 (user-approved): z chain = y chain with teacher --top 4 (user: more teacher candidates). y chain = the anti-drift recipe from R1 on (same rules every round, each round continues from the previous
# adapter): corrections = env_top on teacher rows with per-type cap (2 x share), S >= 2, no look around/inventory/task/wait targets, and
# "focus on" commits only when replacing a FATAL wrong commit (scripts/sciworld_select_capped.py --min_s 2 --skip_info --commit_guard);
# replay = RETENTION replay (by training wins), n = 2.4 x K.  1,483 distinct train variations per round (seed 1), 3B Model A, full-view
# teacher (relaxed validity + refill), trainer 1 ep lr 1e-5 bs 8 seed R, 848 eval with 32 test + 8 dev shards.
# y1 reuses x1's collection / Model A pointer / teacher rows (all computed with student10 = R0, identical to what y1 would compute).
source scripts/lib_sciworld_stepfix.sh; export GMEM=9000; NS=24
SEL="--replay_mult 2.4 --min_s 2 --skip_info --commit_guard"
split_rel() { $P -c "
L=open('$1/traj.release.jsonl').readlines()
for i in range(6): open('$1/rel_s%d.jsonl'%i,'w').writelines(L[i::6])"; }
modelA() { local D=$1; local pids=""
  for i in $(seq 0 5); do ( until grep -q "^DONE" $D/modelA_s$i.log 2>/dev/null; do wait_free "--out $D/modelA_pred_s$i.jsonl"; $(SG $GMEM) $P -u scripts/alfworld_attribute.py --attributor reports/attributor_no_observations/final --variant no_observations --release $D/rel_s$i.jsonl --out $D/modelA_pred_s$i.jsonl >> $D/modelA_s$i.log 2>&1 || sleep 30; done ) & pids="$pids $!"; sleep 20; done; wait $pids
  $P scripts/sciworld_stepfix.py pointer --dir $D | sed "s/^/$(basename $D) /" | tee -a $SF/queue.log; }
teacher() { local D=$1 AD=$2 R=$3; local pids=""
  grep -q "DONE teacher (merged)" $D/teacher.log 2>/dev/null && return 0
  for i in $(seq 0 31); do ( until grep -q "DONE teacher" $D/teacher_s$i.log 2>/dev/null; do echo "### attempt $(date '+%F %T')" >> $D/teacher_s$i.log
      wait_free "teacher --dir $D --adapter .* --shard $i 32"; $(SG 12000) $P -u scripts/sciworld_stepfix.py teacher --dir $D --adapter $AD --shard $i 32 --seed $R --accept_env --max_extra 16 --full_view --top 4 >> $D/teacher_s$i.log 2>&1 || sleep 30; done ) & pids="$pids $!"; sleep 8; done; wait $pids
  echo "DONE teacher (merged)" >> $D/teacher.log; }
trainit() { local D=$1 AD=$2 SD=$3
  until [ -f $D/model/adapter/adapter_config.json ]; do echo "### attempt $(date '+%F %T')" >> $D/train.log
    $(SG 20000) $P -u scripts/sciworld_stepfix.py train --data $D/targets_cap.jsonl $D/replay_ret.jsonl --adapter $AD --out $D/model --seed $SD >> $D/train.log 2>&1 || sleep 30
    for k in $(seq 30); do [ -f $D/model/adapter/adapter_config.json ] && break; sleep 2; done; done
  slog "$(basename $D): trained on $(cat $D/targets_cap.jsonl $D/replay_ret.jsonl | grep -c .) rows ($(tail -1 $D/model/train_log.jsonl))"; }
evalw() { local D=$1 NM=$(basename $1) E=$1/eval; local pids=""; mkdir -p $E
  for i in $(seq 0 31); do shard_loop $E/eval_test_w$i.log scripts/sciworld_agent.py collect --adapter $D/model/adapter --split test --per_task 30 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 32 --out $E/eval_test_w$i.jsonl --tag ${NM}_test & pids="$pids $!"; sleep 5; done
  for i in $(seq 0 7); do shard_loop $E/eval_dev_w$i.log scripts/sciworld_agent.py collect --adapter $D/model/adapter --split dev --per_task 7 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 8 --out $E/eval_dev_w$i.jsonl --tag ${NM}_dev & pids="$pids $!"; sleep 5; done
  wait $pids; cat $E/eval_test_w*.jsonl > $E/eval_test.jsonl; cat $E/eval_dev_w*.jsonl > $E/eval_dev.jsonl; slog "$NM: 848 eval DONE"
  for rf in "R0=$SF/r0/eval_test.jsonl,reports/sciworld/refix/eval_dev_safe.jsonl" "x1=$SF/x1/eval/eval_test.jsonl,$SF/x1/eval/eval_dev.jsonl" "x2d=$SF/x2d/eval/eval_test.jsonl,$SF/x2d/eval/eval_dev.jsonl" "y1=$SF/y1/eval/eval_test.jsonl,$SF/y1/eval/eval_dev.jsonl" "z1=$SF/z1/eval/eval_test.jsonl,$SF/z1/eval/eval_dev.jsonl" "z2=$SF/z2/eval/eval_test.jsonl,$SF/z2/eval/eval_dev.jsonl" "x1rft=$SF/x1rft/eval/eval_test.jsonl,$SF/x1rft/eval/eval_dev.jsonl"; do
    rn=${rf%%=*}; [ "$rn" = "$NM" ] && continue; [ -s ${rf#*=} ] 2>/dev/null || [ -s $(echo ${rf#*=} | cut -d, -f1) ] || continue
    $P scripts/sciworld_stepfix_compare.py --a $rf --b $NM=$E/eval_test.jsonl,$E/eval_dev.jsonl 2>/dev/null | head -1 | tee -a $SF/queue.log; done; }
round() { local NM=$1 R=$2 AD=$3 D=$SF/$1; mkdir -p $D
  if [ ! -s $D/traj.jsonl ]; then collect_sw $D/col $AD 1 $NM $NS --per_task 75
    $P scripts/sciworld_stepfix.py prep --dir $D --inputs "$D/col/traj_s*.jsonl" --expect 1483 | sed "s/^/$NM /" | tee -a $SF/queue.log; split_rel $D; fi
  [ -s $D/pointer.jsonl ] || modelA $D
  teacher $D $AD $R
  [ -s $D/targets_cap.jsonl ] || $P scripts/sciworld_select_capped.py --dir $D --out $D $SEL | tee $D/select.log | sed "s/^/$NM select /" | cut -c1-600 | tee -a $SF/queue.log
  trainit $D $AD $R; evalw $D; }
slog "z chain: start on $(hostname) (pid $$)"
# y1 (teacher top 2, x1's rows) finishes as an ablation in the background, at low priority
( trainit $SF/y1 $STU0 1; evalw $SF/y1 ) &
# z1: x1's collection + Model A pointer (R0), NEW teacher rows with --top 4, then the same selection; train from R0
Z=$SF/z1; mkdir -p $Z
[ -s $Z/pointer.jsonl ] || { cp $SF/x1/{traj.jsonl,ids_fail.jsonl,traj.release.jsonl,pointer.jsonl,pointer_stats.json} $Z/; cp $SF/x1/modelA_pred_s*.jsonl $Z/; }
teacher $Z $STU0 1
[ -s $Z/targets_cap.jsonl ] || $P scripts/sciworld_select_capped.py --dir $Z --out $Z $SEL | tee $Z/select.log | sed "s/^/z1 select /" | cut -c1-600 | tee -a $SF/queue.log
trainit $Z $STU0 1; ( evalw $Z ) & E1=$!
round z2 2 $Z/model/adapter
wait $E1
round z3 3 $SF/z2/model/adapter
slog "z chain DONE"
