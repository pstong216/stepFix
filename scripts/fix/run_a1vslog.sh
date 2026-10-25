#!/bin/bash
# Direct step-selector test in round 2 (Oct 5 07:10): same main-chain R2 trajectories, same Model A diagnoses, same R1 start
# adapter as the main chain's R2 (A1 attributor steps -> 189); only the step changes to the step-back log rule.
cd /users/8/pan00389/multi-agent/error-attribution; SA=reports/alfworld/self_attr_rounds
CHAIN_SUFFIX=_a1vslog SEED=0 bash scripts/self_attr_round.sh selfattr 2 $(cat $SA/selfattr/r1/ADAPTER) reports/alfworld/self_attr/r0/both/adapter any
echo "=== $(date '+%F %T') A1-vs-logfix R2 comparison DONE (main R2 with A1 steps = 189 on 274)" | tee -a $SA/queue.log
