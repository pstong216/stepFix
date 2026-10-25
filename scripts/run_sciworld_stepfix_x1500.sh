#!/bin/bash
# Oct 10 19:10 (user-approved, TOP priority): ScienceWorld with a REALLY larger collection, ~ALFWorld scale (1,500 distinct games):
#   per_task 75 distinct train variations (seed-1 shuffle positions 0..74) = 1,483 DISTINCT variations (518 = positions 0..19 already
#   collected by student10 in r0fix + r0/ext; new = positions 20..74 via --job_slice 20 75).  Every round collects the same 1,483 (seed 1).
# StepFix = hybrid H recipe (as r1big / r2h): 3B Model A (no_observations) pointer, FULL-VIEW teacher + relaxed validity + refill,
#   one-step generic evidence, env_top K = round(0.2 x failures), replay_fw n = round(1.2 x K) (ALFWorld ratio 82:100; user Oct 10 19:20), trainer 1 ep lr 1e-5 bs 8 seed R, start = previous adapter.
# Baseline = standard RFT (every step of every win, exact duplicates dropped) on the SAME collections budget: R1 uses x1's collection,
#   R2 collects the 1,483 with its own R1 adapter.  Eval = the usual 848 (test 651 + dev 197), greedy, budget 50, --snap safe.
source scripts/lib_sciworld_stepfix.sh; export GMEM=9000; NS=24
cmpall() { local N=$1 E=$2; shift 2
  $P scripts/sciworld_stepfix_compare.py --a R0=$SF/r0/eval_test.jsonl,reports/sciworld/refix/eval_dev_safe.jsonl --b $N=$E/eval_test.jsonl,$E/eval_dev.jsonl | head -1 | tee -a $SF/queue.log
  for rf in "$@"; do rn=${rf%%=*}; rd=${rf#*=}; [ -s $rd/eval_test.jsonl ] && $P scripts/sciworld_stepfix_compare.py --a $rn=$rd/eval_test.jsonl,$rd/eval_dev.jsonl --b $N=$E/eval_test.jsonl,$E/eval_dev.jsonl | head -1 | tee -a $SF/queue.log; done; }
split_rel() { $P -c "
L=open('$1/traj.release.jsonl').readlines()
for i in range(6): open('$1/rel_s%d.jsonl'%i,'w').writelines(L[i::6])"; }
modelA() { local D=$1
  for i in $(seq 0 5); do ( until grep -q "^DONE" $D/modelA_s$i.log 2>/dev/null; do wait_free "--out $D/modelA_pred_s$i.jsonl"; $(SG $GMEM) $P -u scripts/alfworld_attribute.py --attributor reports/attributor_no_observations/final --variant no_observations --release $D/rel_s$i.jsonl --out $D/modelA_pred_s$i.jsonl >> $D/modelA_s$i.log 2>&1 || sleep 30; done ) & sleep 20; done; wait
  $P scripts/sciworld_stepfix.py pointer --dir $D | sed "s/^/$(basename $D) /" | tee -a $SF/queue.log; }
teacher() { local D=$1 AD=$2 R=$3
  grep -q "DONE teacher (merged)" $D/teacher.log 2>/dev/null && return 0
  for i in $(seq 0 31); do ( until grep -q "DONE teacher" $D/teacher_s$i.log 2>/dev/null; do echo "### attempt $(date '+%F %T')" >> $D/teacher_s$i.log
      wait_free "teacher --dir $D --adapter .* --shard $i 32"; $(SG 12000) $P -u scripts/sciworld_stepfix.py teacher --dir $D --adapter $AD --shard $i 32 --seed $R --accept_env --max_extra 16 --full_view >> $D/teacher_s$i.log 2>&1 || sleep 30; done ) & sleep 8; done; wait
  echo "DONE teacher (merged)" >> $D/teacher.log; }
trainit() { local D=$1 AD=$2 SD=$3; shift 3
  until [ -f $D/model/adapter/adapter_config.json ]; do echo "### attempt $(date '+%F %T')" >> $D/train.log
    wait_free "train --data .* --out $D/model "; $(SG 20000) $P -u scripts/sciworld_stepfix.py train --data "$@" --adapter $AD --out $D/model --seed $SD >> $D/train.log 2>&1 || { slog "$(basename $D): train attempt failed (retry)"; sleep 60; }
    for k in $(seq 30); do [ -f $D/model/adapter/adapter_config.json ] && break; sleep 2; done; done
  slog "$(basename $D): trained on $(cat "$@" | grep -c .) rows ($(tail -1 $D/model/train_log.jsonl))"; }
stepfix_round() { local D=$1 AD=$2 R=$3
  teacher $D $AD $R
  $P scripts/sciworld_stepfix.py select --dir $D --round $R --rho 0.2 > $D/select.log; grep -v "^random\|^fp" $D/select.log | sed "s/^/$(basename $D) /" | tee -a $SF/queue.log
  [ -s $D/replay_fw.jsonl ] || $P scripts/sciworld_stepfix.py replay_fw --dir $D --n $(python3 -c "print(round(1.2 * $(grep -c . $D/targets_env_top.jsonl)))") | sed "s/^/$(basename $D) /" | tee -a $SF/queue.log
  trainit $D $AD $R $D/targets_env_top.jsonl $D/replay_fw.jsonl; }
rft_round() { local D=$1 AD=$2 R=$3
  [ -s $D/train_rows.jsonl ] || $P scripts/sciworld_baselines.py build --dir $D --mode rftfull --n 0 --seed $R | sed "s/^/$(basename $D) /" | tee -a $SF/queue.log
  trainit $D $AD $R $D/train_rows.jsonl; }

X1=$SF/x1; X2=$SF/x2; F1=$SF/x1rft; F2=$SF/x2rft; mkdir -p $X1 $X2 $F1 $F2
slog "x1500: driver start on $(hostname) (pid $$)"
# ---- R1 collection: new 965 distinct variations with student10 (positions 20..74); the 518 already exist
if [ ! -s $X1/traj.jsonl ]; then
  collect_sw $X1/col $STU0 1 r0x $NS --job_slice 20 75
  $P scripts/sciworld_stepfix.py prep --dir $X1 --inputs "reports/sciworld/r0fix/traj_s?.jsonl" "$SF/r0/ext/traj_s*.jsonl" "$X1/col/traj_s*.jsonl" --expect 1483 | sed "s/^/x1 /" | tee -a $SF/queue.log
  split_rel $X1
fi
# ---- RFT R1 (same collection) in parallel with StepFix R1.  Oct 10 20:10 fix: the bare `wait` inside modelA()/teacher() also waited
#      for this background RFT branch (hours) -> StepFix stalled after Model A.  SKIP_RFT=1 relaunches only the StepFix part (the RFT
#      branch keeps running in the orphaned subshell of the first driver).
[ -n "${SKIP_RFT:-}" ] || ( [ -s $F1/traj.jsonl ] || cp $X1/traj.jsonl $F1/traj.jsonl
  rft_round $F1 $STU0 1; eval_sw $F1/eval $F1/model/adapter x1rft; slog "x1rft: 848 eval DONE"
  cmpall x1rft $F1/eval "rftfull_r1=reports/sciworld/baselines/rftfull/r1/eval"
  # RFT R2: its own collection of the 1,483 with the RFT R1 adapter
  if [ ! -s $F2/traj.jsonl ]; then collect_sw $F2/col $F1/model/adapter 1 x2rft $NS --per_task 75
    $P scripts/sciworld_baselines.py merge --dir $F2 --inputs "$F2/col/traj_s*.jsonl" --expect 1483 | sed "s/^/x2rft /" | tee -a $SF/queue.log; fi
  rft_round $F2 $F1/model/adapter 2; eval_sw $F2/eval $F2/model/adapter x2rft; slog "x2rft: 848 eval DONE"
  cmpall x2rft $F2/eval "x1rft=$F1/eval" "rftfull_r2=reports/sciworld/baselines/rftfull/r2/eval" ) &
# ---- StepFix R1
[ -s $X1/pointer.jsonl ] || modelA $X1
stepfix_round $X1 $STU0 1
( eval_sw $X1/eval $X1/model/adapter x1; slog "x1: 848 eval DONE"
  cmpall x1 $X1/eval "r1b=$SF/r1b/eval_stepfix" "r1big=$SF/r1big/eval" "r2h=$SF/r2h/eval" "x1rft=$F1/eval" ) &
# ---- StepFix R2 (collection with the x1 adapter overlaps x1's eval)
if [ ! -s $X2/traj.jsonl ]; then collect_sw $X2/col $X1/model/adapter 1 x2 $NS --per_task 75
  $P scripts/sciworld_stepfix.py prep --dir $X2 --inputs "$X2/col/traj_s*.jsonl" --expect 1483 | sed "s/^/x2 /" | tee -a $SF/queue.log
  split_rel $X2; fi
[ -s $X2/pointer.jsonl ] || modelA $X2
stepfix_round $X2 $X1/model/adapter 2
eval_sw $X2/eval $X2/model/adapter x2; slog "x2: 848 eval DONE"
cmpall x2 $X2/eval "x1=$X1/eval" "r2h=$SF/r2h/eval" "rftfull_r2=reports/sciworld/baselines/rftfull/r2/eval" "x2rft=$F2/eval"
wait; slog "x1500: ALL DONE"
