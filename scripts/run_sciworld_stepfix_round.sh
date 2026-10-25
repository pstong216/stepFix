#!/bin/bash
# ScienceWorld StepFix round R (Oct 9 2026, user-approved):   bash scripts/run_sciworld_stepfix_round.sh R START_ADAPTER [control: 1|0]
# [R>=2: 518 train episodes (per_task 20, seed 1 every round, T=1.0, --snap safe) with START] -> Model A (3B, no_observations) pointer -> generative hindsight
# self-teacher + one-step evidence (12 GPU shards) -> env_top K = round(0.2 x failures) + 100 replay -> train (1 ep, lr 1e-5, bs 8) -> 848 eval.
# control=1 (R1 only): random K of the teacher top-1 targets at the same Model-A states + the same replay, same trainer/start, same eval.
source scripts/lib_sciworld_stepfix.sh; R=$1; START=$2; CTRL=${3:-0}; NAME=${RD:-r$R}; D=$SF/$NAME; mkdir -p $D   # RD = round-dir override (r1b: the relaxed-validity R1, Oct 9 23:45)
slog "$NAME: start (from $START, control=$CTRL) on $(hostname)"
if [ ! -s $D/traj.jsonl ]; then
  collect_sw $D/col $START 1 r${R}sf 15 --per_task 20   # Oct 9 23:25 (coordinator): seed 1 every round = the same 518 train variations (ALFWorld reuses games 0-1499)
  $P scripts/sciworld_stepfix.py prep --dir $D --inputs "$D/col/traj_s*.jsonl" --expect 518 | tee -a $SF/queue.log
  $P -c "
L=open('$D/traj.release.jsonl').readlines()
for i in range(3): open('$D/rel_s%d.jsonl'%i,'w').writelines(L[i::3])"
fi
[ -s $D/pointer.jsonl ] || bash scripts/run_sciworld_stepfix_modelA.sh $D
[ -f $SF/teacher_args.txt ] && TEACHER_ARGS="$(cat $SF/teacher_args.txt)"; slog "$NAME: teacher args: [${TEACHER_ARGS}]"   # decided at teacher time (empty = the R1 rule)
if ! grep -q "DONE teacher (merged)" $D/teacher.log 2>/dev/null; then
  for i in $(seq 0 11); do ( until grep -q "DONE teacher" $D/teacher_s$i.log 2>/dev/null; do echo "### attempt $(date '+%F %T')" >> $D/teacher_s$i.log
      $(S 12000) $P -u scripts/sciworld_stepfix.py teacher --dir $D --adapter $START --shard $i 12 --seed $R ${TEACHER_ARGS} >> $D/teacher_s$i.log 2>&1 || sleep 30; done ) & sleep 15; done; wait
  echo "DONE teacher (merged)" >> $D/teacher.log
fi
$P scripts/sciworld_stepfix.py select --dir $D --round $R | tee -a $SF/queue.log
[ -s $D/replay.jsonl ] || $P scripts/sciworld_stepfix.py replay --dir $D --seed $R | tee -a $SF/queue.log
trainA() { local NM=$1; shift; until [ -f $D/$NM/adapter/adapter_config.json ]; do echo "### attempt $(date '+%F %T')" >> $D/train_$NM.log
    $(S 20000) $P -u scripts/sciworld_stepfix.py train --data "$@" $D/replay.jsonl --adapter $START --out $D/$NM --seed $R >> $D/train_$NM.log 2>&1 || { slog "$NAME $NM: train attempt failed (retry)"; sleep 60; }
    for k in $(seq 30); do [ -f $D/$NM/adapter/adapter_config.json ] && break; sleep 2; done; done; slog "$NAME $NM: trained ($(tail -1 $D/$NM/train_log.jsonl))"; }
evalA() { local NM=$1; eval_sw $D/eval_$NM $D/$NM/adapter ${NAME}_$NM
  slog "$NAME $NM: 848 eval DONE"; $P scripts/sciworld_stepfix_compare.py --a R0=$SF/r0/eval_test.jsonl,reports/sciworld/refix/eval_dev_safe.jsonl --b ${NAME}_$NM=$D/eval_$NM/eval_test.jsonl,$D/eval_$NM/eval_dev.jsonl | tee -a $SF/queue.log; }
( trainA stepfix $D/targets_env_top.jsonl; evalA stepfix ) &
if [ "$CTRL" = 1 ]; then sleep 20; ( trainA random $D/targets_rand.jsonl; evalA random ) & fi
wait
[ "$CTRL" = 1 ] && $P scripts/sciworld_stepfix_compare.py --a ${NAME}_random=$D/eval_random/eval_test.jsonl,$D/eval_random/eval_dev.jsonl --b ${NAME}_stepfix=$D/eval_stepfix/eval_test.jsonl,$D/eval_stepfix/eval_dev.jsonl | tee -a $SF/queue.log
slog "$NAME DONE"
