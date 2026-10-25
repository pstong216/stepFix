#!/bin/bash
# FINAL METHOD chain (user Oct 6 01:55: all compute to validating ENVIRONMENT ADJUDICATION, R1 -> R2 -> R3).  Every round: proposals =
# first-step + stall-entry states (generic evidence), teacher = hindsight + own analysis, one-step scan on both, filter env_pool
# (per failure the step with the largest verified improvement; top-K by alternative score), K=100 + 100 replay, one-hot KL.
# R1 reuses the already computed teacher/one-step outputs of the two R1 proposal sets (identical inputs).
cd /users/8/pan00389/multi-agent/error-attribution; SA=reports/alfworld/self_attr_rounds; S0=reports/alfworld/student3b_low100/final; A0=reports/alfworld/self_attr/r0/both/adapter
export STEP=logfix STEP_POOL=1 TEACHER_NO_DIAG=1 K=100 EVIDENCE=generic
CH=selfattr; SUF=_pool; S=$S0; AT=$A0; start=1
for R in 1 2 3; do if [ -f $SA/$CH$SUF/r$R/ADAPTER ]; then S=$(cat $SA/$CH$SUF/r$R/ADAPTER); AT=$(cat $SA/$CH$SUF/r$R/ATTR); start=$((R+1)); fi; done
for R in $(seq $start 3); do CHAIN_SUFFIX=$SUF SEED=0 bash scripts/self_attr_round.sh $CH $R $S $AT any || { echo "=== $(date '+%F %T') POOL CHAIN ROUND $R FAILED" | tee -a $SA/queue.log; exit 1; }
  S=$(cat $SA/$CH$SUF/r$R/ADAPTER); AT=$(cat $SA/$CH$SUF/r$R/ATTR)
  Q=reports/alfworld/ext/heldout_queue.txt; grep -q "^selfattr_pool_r$R	" $Q || printf "selfattr_pool_r%s\t%s\n" $R $S >> $Q
done
echo "=== $(date '+%F %T') POOL CHAIN DONE" | tee -a $SA/queue.log
