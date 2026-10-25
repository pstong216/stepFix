#!/bin/bash
# ScienceWorld StepFix REFINED recipe round (user decision Oct 10 02:45):
#   bash scripts/run_sciworld_stepfix_refined.sh NAME R START ARM REPLAY [REUSE_DIR] [REF_NAME=REF_EVAL_DIR]
# ARM = stepfix (margin S > S(a-) + fail-proportional allocation, targets_fp) | random (K random teacher top-1 targets, targets_rand)
# REPLAY = 1 (fail-weighted replay with carry, replay_fw.jsonl) | 0 (no replay).  REUSE_DIR = reuse that dir's collection + Model A pointer.
# Collection: 518 train episodes (per_task 20, seed 1, T=1.0, --snap safe) with START; 3B Model A (no_observations); FULL-VIEW teacher with relaxed
# validity + refill (--accept_env --max_extra 16 --full_view); K = round(0.2 x failures); same trainer; 848 eval at budget 50.
# Priority: GMEM / TGMEM / TRMEM (free MiB required per process) -- ablation chains pass larger values so the main chain keeps the last slots.
source scripts/lib_sciworld_stepfix.sh; NAME=$1; R=$2; START=$3; ARM=$4; REPLAY=$5; REUSE=${6:-}; REF=${7:-}; D=$SF/$NAME; mkdir -p $D
TGMEM=${TGMEM:-12000}; TRMEM=${TRMEM:-20000}
slog "$NAME: start refined round (R=$R from $START, arm=$ARM, replay=$REPLAY, reuse=${REUSE:-none}, GMEM=$GMEM) on $(hostname)"
if [ ! -s $D/pointer.jsonl ]; then
  if [ -n "$REUSE" ]; then cp $REUSE/{traj.jsonl,ids_fail.jsonl,traj.release.jsonl,pointer.jsonl,pointer_stats.json} $D/; cp $REUSE/modelA_pred_s*.jsonl $D/; slog "$NAME: reused collection + pointer of $REUSE"
  else
    [ -s $D/traj.jsonl ] || { collect_sw $D/col $START 1 ${NAME} 15 --per_task 20
      $P scripts/sciworld_stepfix.py prep --dir $D --inputs "$D/col/traj_s*.jsonl" --expect 518 | sed "s/^/$NAME /" | tee -a $SF/queue.log
      $P -c "
L=open('$D/traj.release.jsonl').readlines()
for i in range(3): open('$D/rel_s%d.jsonl'%i,'w').writelines(L[i::3])"; }
    bash scripts/run_sciworld_stepfix_modelA.sh $D | sed "s/^/$NAME /"
  fi
fi
if ! grep -q "DONE teacher (merged)" $D/teacher.log 2>/dev/null; then
  for i in $(seq 0 11); do ( until grep -q "DONE teacher" $D/teacher_s$i.log 2>/dev/null; do echo "### attempt $(date '+%F %T')" >> $D/teacher_s$i.log
      wait_free "teacher --dir $D --adapter .* --shard $i 12"; $(SG $TGMEM) $P -u scripts/sciworld_stepfix.py teacher --dir $D --adapter $START --shard $i 12 --seed $R --accept_env --max_extra 16 --full_view >> $D/teacher_s$i.log 2>&1 || sleep 30; done ) & sleep 15; done; wait
  echo "DONE teacher (merged)" >> $D/teacher.log
fi
if [ "${EVID:-onestep}" = window ] || [ "${EVID:-onestep}" = window_aff ] || [ "${EVID:-onestep}" = window_aff2 ]; then AFF=""; [ "$EVID" = window_aff ] && AFF="--aff"; [ "$EVID" = window_aff2 ] && AFF="--aff2"; WT=${EVID}   # Oct 10 k=3 window evidence (r2s3 / r3s3): greedy 2-step continuation by the round start policy
  for i in $(seq 0 11); do ( until grep -q "DONE window" $D/${WT}_s$i.log 2>/dev/null; do echo "### attempt $(date '+%F %T')" >> $D/${WT}_s$i.log
      wait_free "window --dir $D .*--shard $i 12"; $(SG $TGMEM) $P -u scripts/sciworld_stepfix.py window --dir $D --adapter $START --shard $i 12 --k 3 $AFF >> $D/${WT}_s$i.log 2>&1 || sleep 30; done ) & sleep 15; done; wait
fi
$P scripts/sciworld_stepfix.py select --dir $D --round $R --rho ${RHO:-0.2} --evidence ${EVID:-onestep} > $D/select.log; grep -v "^env_top\|^random" $D/select.log | sed "s/^/$NAME /" | tee -a $SF/queue.log
DATA=$D/targets_fp.jsonl; [ "$ARM" = random ] && DATA=$D/targets_rand.jsonl; [ "$ARM" = all_top1 ] && DATA=$D/targets_all_top1.jsonl; [ "$ARM" = envtop ] && DATA=$D/targets_env_top.jsonl   # all_top1 = ablation C; envtop = hybrid H / K40 (RHO) (Oct 10)
if [ "$ARM" = subgoal ]; then   # Oct 10 next-subgoal relative evidence (r2g / r3g): a- and every candidate continued greedily by the start policy
  for i in $(seq 0 11); do ( until grep -q "DONE subgoal" $D/subgoal_s$i.log 2>/dev/null; do echo "### attempt $(date '+%F %T')" >> $D/subgoal_s$i.log
      wait_free "subgoal --dir $D .*--shard $i 12"; $(SG $TGMEM) $P -u scripts/sciworld_stepfix.py subgoal --dir $D --adapter $START --shard $i 12 --H 9 >> $D/subgoal_s$i.log 2>&1 || sleep 30; done ) & sleep 15; done; wait
  $P scripts/sciworld_stepfix.py select_subgoal --dir $D --H 9 --rho ${RHO:-0.2} | sed "s/^/$NAME /" | tee -a $SF/queue.log; DATA=$D/targets_subgoal.jsonl
fi
if [ "$ARM" = roll ]; then   # ablation D R2/R3 (Oct 10): rollout-verify every teacher candidate with the round start policy, K random among the winners
  for i in $(seq 0 11); do ( until grep -q "DONE rollout" $D/rollout_s$i.log 2>/dev/null; do echo "### attempt $(date '+%F %T')" >> $D/rollout_s$i.log
      wait_free "rollout --dir $D .*--shard $i 12"; $(SG $TGMEM) $P -u scripts/sciworld_stepfix.py rollout --dir $D --adapter $START --shard $i 12 >> $D/rollout_s$i.log 2>&1 || sleep 30; done ) & sleep 15; done; wait
  $P scripts/sciworld_stepfix.py select_rollout --dir $D --round $R | sed "s/^/$NAME /" | tee -a $SF/queue.log; DATA=$D/targets_roll.jsonl
fi
if [ "$REPLAY" = 1 ]; then [ -s $D/replay_fw.jsonl ] || $P scripts/sciworld_stepfix.py replay_fw --dir $D | sed "s/^/$NAME /" | tee -a $SF/queue.log; DATA="$DATA $D/replay_fw.jsonl"; fi
until [ -f $D/model/adapter/adapter_config.json ]; do echo "### attempt $(date '+%F %T')" >> $D/train.log
  wait_free "train --data .* --out $D/model "; $(SG $TRMEM) $P -u scripts/sciworld_stepfix.py train --data $DATA --adapter $START --out $D/model --seed $R >> $D/train.log 2>&1 || { slog "$NAME: train attempt failed (retry)"; sleep 60; }
  for k in $(seq 30); do [ -f $D/model/adapter/adapter_config.json ] && break; sleep 2; done; done
slog "$NAME: trained on $(cat $DATA | grep -c .) rows ($(tail -1 $D/model/train_log.jsonl))"
eval_sw $D/eval $D/model/adapter $NAME
slog "$NAME: 848 eval DONE"
$P scripts/sciworld_stepfix_compare.py --a R0=$SF/r0/eval_test.jsonl,reports/sciworld/refix/eval_dev_safe.jsonl --b $NAME=$D/eval/eval_test.jsonl,$D/eval/eval_dev.jsonl | tee -a $SF/queue.log
for rf in $REF; do rn=${rf%%=*}; rd=${rf#*=}; [ -s $rd/eval_test.jsonl ] && $P scripts/sciworld_stepfix_compare.py --a $rn=$rd/eval_test.jsonl,$rd/eval_dev.jsonl --b $NAME=$D/eval/eval_test.jsonl,$D/eval/eval_dev.jsonl | head -1 | tee -a $SF/queue.log; done
slog "$NAME DONE"
