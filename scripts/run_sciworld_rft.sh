#!/bin/bash
# ScienceWorld RFT baselines, 3 rounds (Oct 10 2026):   bash scripts/run_sciworld_rft.sh MODE      MODE = rftm | rftfull
#   rftm    : RFT matched -- per round, as many success steps (uniformly sampled from the round's successful collection episodes; no
#             look around / inventory / task) as StepFix trained on in that round: R1 162 (62+100), R2 163 (63+100), R3 168 (68+100)
#   rftfull : RFT standard -- every step of every successful collection episode of the round (exact duplicates dropped)
# Both start from student10 (R0).  R1 trains on R0's collection (= reports/sciworld/stepfix/r1b/traj.jsonl, 518 eps, the one StepFix R1 used);
# round R>=2 collects the 518 train variations with its own R-1 adapter (T=1.0).  Trainer, seeds (seed = R), collection and 848 eval as StepFix.
# Eval of round R runs in the background while round R+1 collects.  Everything resumes (episodes append; trainer checkpoints).
source scripts/lib_sciworld_rft.sh; M=$1; B=$BL/$M; mkdir -p $B
case $M in rftm|rftfull) ;; *) echo "MODE must be rftm or rftfull"; exit 1;; esac
NROWS=(0 162 163 168); TAGM=$([ $M = rftm ] && echo rftm || echo rftf)
blog "$M: driver start on $(hostname) (pid $$)"
START=$STU0
for R in 1 2 3; do D=$B/r$R; mkdir -p $D
  if [ ! -s $D/traj.jsonl ]; then
    if [ $R = 1 ]; then cp $SF/r1b/traj.jsonl $D/traj.jsonl; blog "$M r1: traj = copy of $SF/r1b/traj.jsonl (R0 = student10 collection, $(grep -c . $D/traj.jsonl) eps, $(grep -c '"won": true' $D/traj.jsonl) won)"
    else blog "$M r$R: collecting 518 train episodes with $START"
      collect_rft $D/col $START r${R}$TAGM 16
      $P scripts/sciworld_baselines.py merge --dir $D --inputs "$D/col/traj_s*.jsonl" --expect 518 | tee -a $BL/queue.log
      [ ${PIPESTATUS[0]} = 0 ] || { blog "FAIL $M r$R: merged collection is not 518 episodes -- driver stops"; rm -f $D/traj.jsonl; exit 1; }
    fi
  fi
  [ -s $D/train_rows.jsonl ] || $P scripts/sciworld_baselines.py build --dir $D --mode $M --n ${NROWS[$R]} --seed $R | tee -a $BL/queue.log
  blog "$M r$R: training $(grep -c . $D/train_rows.jsonl) rows from $START"
  train_rft $D $START $R || exit 1
  blog "$M r$R: trained ($(tail -1 $D/model/train_log.jsonl))"
  ( eval_rft $D/eval $D/model/adapter ${M}_r$R; blog "$M r$R: 848 eval DONE"; cmp848 ${M}_r$R $D/eval $R ) &
  START=$D/model/adapter
done
wait; blog "$M: ALL ROUNDS DONE"
