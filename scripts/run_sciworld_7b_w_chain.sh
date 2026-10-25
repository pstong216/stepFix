#!/bin/bash
# Oct 11 03:00 (user: "马上按照这个配方跑7B"): 7B ScienceWorld with the w recipe (= 3B w chain): teacher --top 4 (full view, relaxed
# validity + refill), selection scripts/sciworld_select_capped.py --replay_mult 2.4 --min_s 2 --commit_guard --cap_mult 1000 (commit
# guard mode from reports/sciworld/stepfix/.commit_guard_mode = confirmed), retention replay; 7B student + 7B Model A; 1,483 distinct
# variations per round; each round continues from the previous adapter.  w7_1 reuses the 7B x1500 collection + 7B Model A pointer.
source scripts/lib_sciworld_7b.sh; SEL="--replay_mult 2.4 --min_s 2 --commit_guard --cap_mult 1000"
teach7() { local D=$1 AD=$2 R=$3; local pids=""
  grep -q "DONE teacher (merged)" $D/teacher.log 2>/dev/null && return 0
  for i in $(seq 0 47); do NEED=$TEACH7 loop7 $D/teacher_s$i.log "DONE teacher" scripts/sciworld_stepfix.py teacher --model $MODEL --dir $D --adapter $AD --shard $i 48 --seed $R --accept_env --max_extra 16 --full_view --top 4 & pids="$pids $!"; sleep 15; done; wait $pids
  echo "DONE teacher (merged)" >> $D/teacher.log; }
train7() { local D=$1 AD=$2 R=$3
  NEED=$TRAIN7 loop7 $D/train.log "DONE train" scripts/sciworld_stepfix.py train --model $MODEL --data $D/targets_cap.jsonl $D/replay_ret.jsonl --adapter $AD --out $D/model --seed $R
  s7log "$(basename $D): trained ($(tail -1 $D/model/train_log.jsonl))"; }
ev7() { local D=$1 NM=$(basename $1)
  eval7 $D/eval $D/model/adapter $NM; s7log "$NM 848: $(sum848 $D/eval/eval_test.jsonl $D/eval/eval_dev.jsonl)"
  for rf in "R0_7B=$SF7/r0" "x7=$SF7/x1/eval" "w7_1=$SF7/w7_1/eval" "w2_3B=$SF3/w2/eval"; do rn=${rf%%=*}; rd=${rf#*=}; [ "$rn" = "$NM" ] && continue; [ -s $rd/eval_test.jsonl ] || continue
    $P scripts/sciworld_stepfix_compare.py --a $rn=$rd/eval_test.jsonl,$rd/eval_dev.jsonl --b $NM=$D/eval/eval_test.jsonl,$D/eval/eval_dev.jsonl | head -1 | tee -a $SF7/queue.log; done; }
s7log "7B w chain: start (pid $$)"
D=$SF7/w7_1; mkdir -p $D
[ -s $D/pointer.jsonl ] || { cp $SF7/x1/{traj.jsonl,ids_fail.jsonl,traj.release.jsonl,pointer.jsonl,pointer_stats.json} $D/; cp $SF7/x1/modelA_pred_s*.jsonl $D/; }
teach7 $D $STU7 1
[ -s $D/targets_cap.jsonl ] || $P scripts/sciworld_select_capped.py --dir $D --out $D $SEL | tee $D/select.log | sed "s/^/w7_1 select /" | cut -c1-500 | tee -a $SF7/queue.log
train7 $D $STU7 1; ( ev7 $D ) & E1=$!
# 7B R2
D2=$SF7/w7_2; mkdir -p $D2
if [ ! -s $D2/traj.jsonl ]; then collect7 $D2/col $D/model/adapter 1 w7_2 60 --per_task 75
  $P scripts/sciworld_stepfix.py prep --dir $D2 --inputs "$D2/col/traj_s*.jsonl" --expect 1483 | sed "s/^/w7_2 /" | tee -a $SF7/queue.log
  $P -c "
L=open('$D2/traj.release.jsonl').readlines()
for i in range(6): open('$D2/rel_s%d.jsonl'%i,'w').writelines(L[i::6])"; fi
if [ ! -s $D2/pointer.jsonl ]; then pids=""; for i in $(seq 0 5); do NEED=$GMEM7 loop7 $D2/modelA_s$i.log "^DONE" scripts/alfworld_attribute.py --attributor $MA7 --variant no_observations --release $D2/rel_s$i.jsonl --out $D2/modelA_pred_s$i.jsonl & pids="$pids $!"; sleep 20; done; wait $pids
  $P scripts/sciworld_stepfix.py pointer --dir $D2 | sed "s/^/w7_2 /" | tee -a $SF7/queue.log; fi
teach7 $D2 $D/model/adapter 2
[ -s $D2/targets_cap.jsonl ] || $P scripts/sciworld_select_capped.py --dir $D2 --out $D2 $SEL | tee $D2/select.log | sed "s/^/w7_2 select /" | cut -c1-500 | tee -a $SF7/queue.log
train7 $D2 $D/model/adapter 2; ev7 $D2
wait $E1; s7log "7B w chain DONE"
