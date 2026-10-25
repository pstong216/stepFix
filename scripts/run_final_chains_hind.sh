#!/bin/bash
# FINAL-recipe chains (user decision Oct 5 16:30): step = step-back log rule at every round (STEP=logfix); teacher = hindsight +
# own analysis, NO Model A (TEACHER_NO_DIAG=1); env one-step top-86 + 86 replay; one-hot KL.  Dirs: selfattr_hind (seed 0),
# selfattr_rollout_hind, selfattr_placebo_hind.  R1 shared artifacts (pointer, teacher candidates, one-step scan, replay) are
# computed once by the main chain and copied into the two controls.
cd /users/8/pan00389/multi-agent/error-attribution; SA=reports/alfworld/self_attr_rounds; S0=reports/alfworld/student3b_low100/final; A0=reports/alfworld/self_attr/r0/both/adapter
export STEP=logfix TEACHER_NO_DIAG=1 K=100 EVIDENCE=generic
chain() { local CH=$1 SUF=$2 SEED=$3 HOLD=$4; local S=$S0 AT=$A0 start=1
  for R in 1 2 3; do if [ -f $SA/$CH$SUF/r$R/ADAPTER ]; then S=$(cat $SA/$CH$SUF/r$R/ADAPTER); AT=$(cat $SA/$CH$SUF/r$R/ATTR); start=$((R+1)); fi; done
  for R in $(seq $start 3); do CHAIN_SUFFIX=$SUF SEED=$SEED bash scripts/self_attr_round.sh $CH $R $S $AT $HOLD || { echo "=== $(date '+%F %T') FINAL CHAIN $CH$SUF ROUND $R FAILED" | tee -a $SA/queue.log; return 1; }; S=$(cat $SA/$CH$SUF/r$R/ADAPTER); AT=$(cat $SA/$CH$SUF/r$R/ATTR); done
  echo "=== $(date '+%F %T') FINAL CHAIN $CH$SUF DONE" | tee -a $SA/queue.log; }
SRC=$SA/selfattr_hind/r1
share() { local D=$1; mkdir -p $D/self $D/proxy $D/envlab $D/attr_labels
  for f in pointer.jsonl pointer.raw.jsonl diag.raw.jsonl teacher.log onestep.log replay.jsonl ids_fail.jsonl; do [ -e $SRC/$f ] && [ ! -e $D/$f ] && cp $SRC/$f $D/; done
  for f in self/steps.jsonl self/targets.jsonl proxy/onestep.jsonl; do [ -e $SRC/$f ] && [ ! -e $D/$f ] && cp $SRC/$f $D/$f; done; }
case ${1:-main} in
  main) chain selfattr _hind 0 any ;;
  wait_share) until grep -q "DONE informed_teacher" $SRC/teacher.log 2>/dev/null && grep -q "DONE onestep" $SRC/onestep.log 2>/dev/null && [ -s $SRC/replay.jsonl ]; do sleep 120; done
    for d in selfattr_rollout_hind selfattr_placebo_hind selfattr_rollsub_hind selfattr_eqv_hind; do share $SA/$d/r1; done
    echo "=== $(date '+%F %T') final R1 shared artifacts copied to rollout/placebo controls" | tee -a $SA/queue.log
    tmux new-session -d -s hind_rollout "bash scripts/run_final_chains_hind.sh rollout"; sleep 60
    tmux new-session -d -s hind_placebo "bash scripts/run_final_chains_hind.sh placebo"; sleep 60
    tmux new-session -d -s hind_rollsub "bash scripts/run_final_chains_hind.sh rollsub"; sleep 60
    tmux new-session -d -s hind_eqv "bash scripts/run_final_chains_hind.sh eqv" ;;
  rollout) chain selfattr_rollout _hind 0 any ;;
  placebo) chain selfattr_placebo _hind 0 any ;;
  rollsub) chain selfattr_rollsub _hind 0 any ;;
  eqv) chain selfattr_eqv _hind 0 any ;;
esac
