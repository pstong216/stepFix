#!/bin/bash
# Oct 12 21:05 (user: "至少把7B ScienceWorld的Rollout verification跑完"): rounds 2 and 3 of 7B ScienceWorld rollout verification, continuing
# from roll7_1 (R1 = 315). Same per-round protocol as the 3B abl_roll chain (run_sciworld_abl_chain.sh MODE=roll) with the 7B settings of the
# main 7B chain (run_sciworld_7b_r3.sh): collect 1,483 variations (seed 1) with the start policy -> 7B Model A -> top-4 full-view teacher
# (start policy) -> selection (retention replay 2.4K) -> rollout of every candidate by the start policy -> K random states with a winning
# candidate -> train targets_roll + replay_ret, KL beta 3, lr 2e-5 (as 7B StepFix R2/R3) -> 848 eval (24 test + 6 dev shards).
source scripts/lib_sciworld_7b.sh; umask 0027; SEL="--replay_mult 2.4 --min_s 2 --commit_guard --cap_mult 1000"; export LAUNCH_GAP=45 LOOP7_MAX=400   # Oct 12 23:30: gap 8 s caused OOM thrash (pick() races) and 25-attempt give-ups -> partial collections
teach7() { local D=$1 AD=$2 R=$3 pids=""; grep -q "DONE teacher (merged)" $D/teacher.log 2>/dev/null && return 0
  for i in $(seq 0 47); do NEED=$TEACH7 loop7 $D/teacher_s$i.log "DONE teacher" scripts/sciworld_stepfix.py teacher --model $MODEL --dir $D --adapter $AD --shard $i 48 --seed $R --accept_env --max_extra 16 --full_view --top 4 & pids="$pids $!"; sleep 30; done; wait $pids
  echo "DONE teacher (merged)" >> $D/teacher.log; }
roll7() { local D=$1 AD=$2 pids=""; grep -q "DONE rollout (merged)" $D/rollout.log 2>/dev/null && return 0
  for i in $(seq 0 47); do NEED=$GMEM7 loop7 $D/rollout_s$i.log "DONE rollout" scripts/sciworld_stepfix.py rollout --model $MODEL --dir $D --adapter $AD --shard $i 48 & pids="$pids $!"; sleep 30; done; wait $pids
  echo "DONE rollout (merged)" >> $D/rollout.log; }
ev7() { local D=$1 NM=$(basename $1) E=$1/eval pids=""; mkdir -p $E
  for i in $(seq 0 23); do loop7 $E/eval_test_s$i.log "DONE collect" scripts/sciworld_agent.py collect --model $MODEL --adapter $D/model/adapter --split test --per_task 30 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 24 --out $E/eval_test_s$i.jsonl --tag ${NM}_test & pids="$pids $!"; sleep 30; done
  for i in $(seq 0 5); do loop7 $E/eval_dev_s$i.log "DONE collect" scripts/sciworld_agent.py collect --model $MODEL --adapter $D/model/adapter --split dev --per_task 7 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 6 --out $E/eval_dev_s$i.jsonl --tag ${NM}_dev & pids="$pids $!"; sleep 30; done
  wait $pids; cat $E/eval_test_s*.jsonl > $E/eval_test.jsonl; cat $E/eval_dev_s*.jsonl > $E/eval_dev.jsonl
  s7log "$NM 848: $(sum848 $E/eval_test.jsonl $E/eval_dev.jsonl)"
  for rf in "R0_7B=$SF7/r0" "w7_1=$SF7/w7_1/eval" "w7_2b3lr=$SF7/w7_2b3lr/eval" "w7_3=$SF7/w7_3/eval" "roll7_1=$SF7/roll7_1/eval"; do rn=${rf%%=*}; rd=${rf#*=}; [ "$rn" = "$NM" ] && continue; [ -s $rd/eval_test.jsonl ] || continue
    $P scripts/sciworld_stepfix_compare.py --a $rn=$rd/eval_test.jsonl,$rd/eval_dev.jsonl --b $NM=$E/eval_test.jsonl,$E/eval_dev.jsonl | head -1 | tee -a $SF7/queue.log; done; }
round() { local NM=$1 R=$2 AD=$3 D=$SF7/$1; mkdir -p $D; s7log "$NM: start (rollout verification R$R, from $AD)"
  if [ ! -s $D/traj.jsonl ]; then until [ "$(cat $D/col/traj_s*.jsonl 2>/dev/null | grep -c .)" -ge 1483 ]; do collect7 $D/col $AD 1 $NM 60 --per_task 75; done
    $P scripts/sciworld_stepfix.py prep --dir $D --inputs "$D/col/traj_s*.jsonl" --expect 1483 | sed "s/^/$NM /" | tee -a $SF7/queue.log
    $P -c "
L=open('$D/traj.release.jsonl').readlines()
for i in range(6): open('$D/rel_s%d.jsonl'%i,'w').writelines(L[i::6])"; fi
  if [ ! -s $D/pointer.jsonl ]; then local pids=""; for i in $(seq 0 5); do NEED=$GMEM7 loop7 $D/modelA_s$i.log "^DONE" scripts/alfworld_attribute.py --attributor $MA7 --variant no_observations --release $D/rel_s$i.jsonl --out $D/modelA_pred_s$i.jsonl & pids="$pids $!"; sleep 10; done; wait $pids
    $P scripts/sciworld_stepfix.py pointer --dir $D | sed "s/^/$NM /" | tee -a $SF7/queue.log; fi
  teach7 $D $AD $R
  [ -s $D/replay_ret.jsonl ] || $P scripts/sciworld_select_capped.py --dir $D --out $D $SEL > $D/select_capped.log
  roll7 $D $AD
  [ -s $D/targets_roll.jsonl ] || $P scripts/sciworld_stepfix.py select_rollout --dir $D --round $R | sed "s/^/$NM /" | tee -a $SF7/queue.log
  NEED=$TRAIN7 loop7 $D/train.log "DONE train" scripts/sciworld_stepfix.py train --model $MODEL --data $D/targets_roll.jsonl $D/replay_ret.jsonl --adapter $AD --out $D/model --seed $R --kl_replay --kl_beta 3 --lr 2e-5
  s7log "$NM: trained ($(tail -1 $D/model/train_log.jsonl))"; }
round roll7_2 2 $SF7/roll7_1/model/adapter; ( ev7 $SF7/roll7_2 ) & E2=$!
round roll7_3 3 $SF7/roll7_2/model/adapter; ev7 $SF7/roll7_3
wait $E2; s7log "7B rollout-verification chain (roll7_2, roll7_3) DONE"
