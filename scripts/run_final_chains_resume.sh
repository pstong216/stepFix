#!/bin/bash
# Resume the final-recipe chains from their latest completed round (all stages resume from files).  Used after swapping the round
# script (Oct 5 01:45: step selector -> log-evidence rule from round 2 on).
cd /users/8/pan00389/multi-agent/error-attribution; SA=reports/alfworld/self_attr_rounds; S0=reports/alfworld/student3b_low100/final; A0=reports/alfworld/self_attr/r0/both/adapter
resume() { local CH=$1 SUF=$2 SEED=$3 HOLD=$4; local S=$S0 AT=$A0 start=1
  for R in 1 2 3; do if [ -f $SA/$CH$SUF/r$R/ADAPTER ]; then S=$(cat $SA/$CH$SUF/r$R/ADAPTER); AT=$(cat $SA/$CH$SUF/r$R/ATTR); start=$((R+1)); fi; done
  for R in $(seq $start 3); do CHAIN_SUFFIX=$SUF SEED=$SEED bash scripts/self_attr_round.sh $CH $R $S $AT $HOLD || break; S=$(cat $SA/$CH$SUF/r$R/ADAPTER); AT=$(cat $SA/$CH$SUF/r$R/ATTR); done
  echo "=== $(date '+%F %T') FINAL CHAIN $CH$SUF DONE" | tee -a $SA/queue.log; }
resume selfattr "" 0 error_attr_hold_igpu_qw &
sleep 90; resume selfattr _s1 1 any &
sleep 90; resume selfattr _s2 2 any &
sleep 90; resume selfattr_rollout "" 0 any &
sleep 90; resume selfattr_placebo "" 0 any &
wait
