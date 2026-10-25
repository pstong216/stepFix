#!/bin/bash
# Oct 11 14:55 (user: "消融尽量都跑3轮"): rounds 2 and 3 of the 3B ScienceWorld ablations. Each ablation continues from its own R1
# (reports/sciworld/stepfix/<NAME>/model/adapter) and changes ONE component of the final recipe in every round:
#   rand : K random teacher top-1 targets (no evidence)            + retention replay 2.4K, KL-anchored (as the final recipe)
#   nofil: teacher top-1 at every attributed failure (no filter)   + retention replay 2.4K, KL-anchored
#   norep: the final recipe's corrections (S>=2, commit confirmation, top 20%), NO replay (hence no KL anchor)
#   rftm : sampled success steps of the round's collection, as many rows as the main chain trained on that round (R2 632, R3 629)
#   roll : rollout verification (K random states with a winning candidate) + retention replay 2.4K, KL-anchored
# Per round: collect 1,483 variations (seed 1) with the start policy -> 3B Model A -> top-4 full-view teacher -> selection -> train (seed R)
# -> 848 eval (32 test + 8 dev shards).   usage: bash run_sciworld_abl_chain.sh NAME MODE      e.g. abl_rand rand
source scripts/lib_sciworld_stepfix.sh; export GMEM=9000; NS=24; NAME=$1; MODE=$2
split_rel() { $P -c "
L=open('$1/traj.release.jsonl').readlines()
for i in range(6): open('$1/rel_s%d.jsonl'%i,'w').writelines(L[i::6])"; }
modelA() { local D=$1 pids=""
  for i in $(seq 0 5); do ( until grep -q "^DONE" $D/modelA_s$i.log 2>/dev/null; do $(SG $GMEM) $P -u scripts/alfworld_attribute.py --attributor reports/attributor_no_observations/final --variant no_observations --release $D/rel_s$i.jsonl --out $D/modelA_pred_s$i.jsonl >> $D/modelA_s$i.log 2>&1 || sleep 30; done ) & pids="$pids $!"; sleep 20; done; wait $pids
  $P scripts/sciworld_stepfix.py pointer --dir $D | sed "s/^/$(basename $D) /" | tee -a $SF/queue.log; }
teacher() { local D=$1 AD=$2 R=$3 pids=""
  grep -q "DONE teacher (merged)" $D/teacher.log 2>/dev/null && return 0
  for i in $(seq 0 31); do ( until grep -q "DONE teacher" $D/teacher_s$i.log 2>/dev/null; do $(SG 12000) $P -u scripts/sciworld_stepfix.py teacher --dir $D --adapter $AD --shard $i 32 --seed $R --accept_env --max_extra 16 --full_view --top 4 >> $D/teacher_s$i.log 2>&1 || sleep 30; done ) & pids="$pids $!"; sleep 8; done; wait $pids
  echo "DONE teacher (merged)" >> $D/teacher.log; }
rollout() { local D=$1 AD=$2 pids=""
  grep -q "DONE rollout (merged)" $D/rollout.log 2>/dev/null && return 0
  for i in $(seq 0 47); do ( until grep -q "DONE rollout" $D/rollout_s$i.log 2>/dev/null; do $(SG $GMEM) $P -u scripts/sciworld_stepfix.py rollout --dir $D --adapter $AD --shard $i 48 >> $D/rollout_s$i.log 2>&1 || sleep 30; done ) & pids="$pids $!"; sleep 6; done; wait $pids
  echo "DONE rollout (merged)" >> $D/rollout.log; }
evalw() { local D=$1 NM=$(basename $1) E=$1/eval pids=""; mkdir -p $E
  for i in $(seq 0 31); do shard_loop $E/eval_test_w$i.log scripts/sciworld_agent.py collect --adapter $D/model/adapter --split test --per_task 30 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 32 --out $E/eval_test_w$i.jsonl --tag ${NM}_test & pids="$pids $!"; sleep 5; done
  for i in $(seq 0 7); do shard_loop $E/eval_dev_w$i.log scripts/sciworld_agent.py collect --adapter $D/model/adapter --split dev --per_task 7 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 8 --out $E/eval_dev_w$i.jsonl --tag ${NM}_dev & pids="$pids $!"; sleep 5; done
  wait $pids; cat $E/eval_test_w*.jsonl > $E/eval_test.jsonl; cat $E/eval_dev_w*.jsonl > $E/eval_dev.jsonl; slog "$NM: 848 eval DONE"
  for rf in R0=$SF/r0/eval_test.jsonl,reports/sciworld/refix/eval_dev_safe.jsonl w2k=$SF/w2k/eval/eval_test.jsonl,$SF/w2k/eval/eval_dev.jsonl wk3=$SF/wk3/eval/eval_test.jsonl,$SF/wk3/eval/eval_dev.jsonl; do $P scripts/sciworld_stepfix_compare.py --a $rf --b $NM=$E/eval_test.jsonl,$E/eval_dev.jsonl | head -1 | tee -a $SF/queue.log; done; }
abl_round() { local R=$1 AD=$2; local D=$SF/${NAME}_r$R; mkdir -p $D   # Oct 11 17:25 fix: D was expanded before R was set
  slog "${NAME}_r$R: start (mode $MODE, from $AD)"
  if [ ! -s $D/traj.jsonl ]; then collect_sw $D/col $AD 1 ${NAME}_r$R $NS --per_task 75
    $P scripts/sciworld_stepfix.py prep --dir $D --inputs "$D/col/traj_s*.jsonl" --expect 1483 | sed "s/^/${NAME}_r$R /" | tee -a $SF/queue.log; split_rel $D; fi
  local DATA KL=""
  if [ "$MODE" = rftm ]; then local NR=$([ $R = 2 ] && echo 632 || echo 629)
    [ -s $D/train_rows.jsonl ] || $P scripts/sciworld_baselines.py build --dir $D --mode rftm --n $NR --seed $R | sed "s/^/${NAME}_r$R /" | tee -a $SF/queue.log
    DATA="$D/train_rows.jsonl"
  else
    [ -s $D/pointer.jsonl ] || modelA $D
    teacher $D $AD $R
    [ -s $D/replay_ret.jsonl ] || $P scripts/sciworld_select_capped.py --dir $D --out $D --replay_mult 2.4 --min_s 2 --commit_guard --cap_mult 1000 > $D/select_capped.log
    case $MODE in
      norep) DATA="$D/targets_cap.jsonl";;
      rand|nofil) $P scripts/sciworld_stepfix.py select --dir $D --round $R --rho 0.2 > $D/select.log
        DATA="$D/$([ $MODE = rand ] && echo targets_rand || echo targets_all_top1).jsonl $D/replay_ret.jsonl"; KL="--kl_replay";;
      roll) rollout $D $AD; $P scripts/sciworld_stepfix.py select_rollout --dir $D --round $R | sed "s/^/${NAME}_r$R /" | tee -a $SF/queue.log
        DATA="$D/targets_roll.jsonl $D/replay_ret.jsonl"; KL="--kl_replay";;
    esac
  fi
  until [ -f $D/model/adapter/adapter_config.json ]; do echo "### attempt $(date '+%F %T')" >> $D/train.log
    $(SG 24000) $P -u scripts/sciworld_stepfix.py train --data $DATA --adapter $AD --out $D/model --seed $R $KL >> $D/train.log 2>&1 || sleep 30
    for k in $(seq 30); do [ -f $D/model/adapter/adapter_config.json ] && break; sleep 2; done; done
  slog "${NAME}_r$R: trained on $(cat $DATA | grep -c .) rows ($(tail -1 $D/model/train_log.jsonl))"; }
slog "$NAME chain: start (mode $MODE)"
abl_round 2 $SF/$NAME/model/adapter; ( evalw $SF/${NAME}_r2 ) & E2=$!
abl_round 3 $SF/${NAME}_r2/model/adapter; evalw $SF/${NAME}_r3
wait $E2; slog "$NAME chain DONE"
