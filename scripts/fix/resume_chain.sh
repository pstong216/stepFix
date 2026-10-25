#!/bin/bash
# Resume one final-recipe chain from its latest completed round.  Usage: resume_chain.sh <chain> <suffix|""> <seed> <hold|any>
cd /users/8/pan00389/multi-agent/error-attribution; SA=reports/alfworld/self_attr_rounds; S0=reports/alfworld/student3b_low100/final; A0=reports/alfworld/self_attr/r0/both/adapter
CH=$1; SUF=$2; SEED=$3; HOLD=$4; S=$S0; AT=$A0; start=1
for R in 1 2 3; do if [ -f $SA/$CH$SUF/r$R/ADAPTER ]; then S=$(cat $SA/$CH$SUF/r$R/ADAPTER); AT=$(cat $SA/$CH$SUF/r$R/ATTR); start=$((R+1)); fi; done
for R in $(seq $start 3); do CHAIN_SUFFIX=$SUF SEED=$SEED bash scripts/self_attr_round.sh $CH $R $S $AT $HOLD || { echo "=== $(date '+%F %T') FINAL CHAIN $CH$SUF ROUND $R FAILED" | tee -a $SA/queue.log; exit 1; }; S=$(cat $SA/$CH$SUF/r$R/ADAPTER); AT=$(cat $SA/$CH$SUF/r$R/ATTR); done
echo "=== $(date '+%F %T') FINAL CHAIN $CH$SUF DONE" | tee -a $SA/queue.log
