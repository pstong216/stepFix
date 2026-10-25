#!/bin/bash
# Continue both adjudication lineages to R3 with the chosen budget (corrections 40% of failures, replay 100):
#   A) all-adjudication: R1 selfattr_pool (K100) -> R2 selfattr_poolk40r100 -> R3 here
#   B) first-step R1 -> adjudication: R1 selfattr_hind -> R2 selfattr_hindpoolk40r100 -> R3 here
cd /users/8/pan00389/multi-agent/error-attribution; SA=reports/alfworld/self_attr_rounds
SUF=$1; EVIDENCE=generic STEP=logfix STEP_POOL=1 TEACHER_NO_DIAG=1 K_FRAC=0.4 REPLAY_N=100 CHAIN_SUFFIX=$SUF SEED=0 bash scripts/self_attr_round.sh selfattr 3 $(cat $SA/selfattr$SUF/r2/ADAPTER) reports/alfworld/self_attr/r0/both/adapter any
echo "=== $(date '+%F %T') R3 (budget 40%+100) DONE (selfattr$SUF)" | tee -a $SA/queue.log
