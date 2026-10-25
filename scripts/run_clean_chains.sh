#!/bin/bash
# CLEAN final-recipe chains (user request Oct 5 12:00): the finalized step selector (step-back log rule, STEP=logfix) at EVERY
# round incl. R1; informed teacher (hindsight + Model A diagnosis + own analysis); env one-step top-86 + 86 replay; one-hot KL.
# Chains: selfattr_clean seeds 0/1/2, selfattr_rollout_clean (rollout-verification control), selfattr_placebo_clean (random control).
# R1 artifacts that do not depend on seed/filter (pointer, Model A, teacher candidates, one-step scan, replay) are computed once by
# the seed-0 chain and copied into the other four before they start, so all five R1s share identical candidates.
cd /users/8/pan00389/multi-agent/error-attribution; SA=reports/alfworld/self_attr_rounds; S0=reports/alfworld/student3b_low100/final; A0=reports/alfworld/self_attr/r0/both/adapter
export STEP=logfix
chain() { local CH=$1 SUF=$2 SEED=$3 HOLD=$4; local S=$S0 AT=$A0 start=1
  for R in 1 2 3; do if [ -f $SA/$CH$SUF/r$R/ADAPTER ]; then S=$(cat $SA/$CH$SUF/r$R/ADAPTER); AT=$(cat $SA/$CH$SUF/r$R/ATTR); start=$((R+1)); fi; done
  for R in $(seq $start 3); do CHAIN_SUFFIX=$SUF SEED=$SEED bash scripts/self_attr_round.sh $CH $R $S $AT $HOLD || { echo "=== $(date '+%F %T') CLEAN CHAIN $CH$SUF ROUND $R FAILED" | tee -a $SA/queue.log; return 1; }; S=$(cat $SA/$CH$SUF/r$R/ADAPTER); AT=$(cat $SA/$CH$SUF/r$R/ATTR); done
  echo "=== $(date '+%F %T') CLEAN CHAIN $CH$SUF DONE" | tee -a $SA/queue.log; }
SRC=$SA/selfattr_clean/r1
share() { local D=$1; mkdir -p $D/self $D/proxy $D/envlab $D/attr_labels
  for f in pointer.jsonl pointer.raw.jsonl diag.raw.jsonl diag_modelA.raw.jsonl modelA_pred.jsonl modelA.log teacher.log onestep.log replay.jsonl ids_fail.jsonl; do [ -e $SRC/$f ] && [ ! -e $D/$f ] && cp $SRC/$f $D/; done
  for f in self/steps.jsonl self/targets.jsonl proxy/onestep.jsonl; do [ -e $SRC/$f ] && [ ! -e $D/$f ] && cp $SRC/$f $D/$f; done; }
case ${1:-main} in
  main) chain selfattr _clean 0 any ;;
  wait_share)   # wait for the seed-0 clean R1 shared stages, then copy them into the other four chains and start them
    until grep -q "DONE informed_teacher" $SRC/teacher.log 2>/dev/null && grep -q "DONE onestep" $SRC/onestep.log 2>/dev/null && [ -s $SRC/replay.jsonl ]; do sleep 120; done
    for d in selfattr_clean_s1 selfattr_clean_s2 selfattr_rollout_clean selfattr_placebo_clean; do share $SA/$d/r1; done
    echo "=== $(date '+%F %T') clean R1 shared artifacts copied to s1/s2/rollout/placebo" | tee -a $SA/queue.log
    tmux new-session -d -s clean_s1 "bash scripts/run_clean_chains.sh s1"; sleep 60
    tmux new-session -d -s clean_s2 "bash scripts/run_clean_chains.sh s2"; sleep 60
    tmux new-session -d -s clean_rollout "bash scripts/run_clean_chains.sh rollout"; sleep 60
    tmux new-session -d -s clean_placebo "bash scripts/run_clean_chains.sh placebo" ;;
  s1) chain selfattr _clean_s1 1 any ;;
  s2) chain selfattr _clean_s2 2 any ;;
  rollout) chain selfattr_rollout _clean 0 any ;;
  placebo) chain selfattr_placebo _clean 0 any ;;
esac
