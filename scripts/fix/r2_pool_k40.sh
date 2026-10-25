#!/bin/bash
# R2 same-trajectory adjudication with K = 40% of the round's failures (user asked whether 100 is too few; R1 774: top-175 539 > top-100 508)
cd /users/8/pan00389/multi-agent/error-attribution; SA=reports/alfworld/self_attr_rounds
EVIDENCE=generic STEP=logfix STEP_POOL=1 TEACHER_NO_DIAG=1 K=138 CHAIN_SUFFIX=_hindpoolk40 SEED=0 bash scripts/self_attr_round.sh selfattr 2 $(cat $SA/selfattr_hind/r1/ADAPTER) reports/alfworld/self_attr/r0/both/adapter any
echo "=== $(date '+%F %T') R2 same-trajectory control DONE (selfattr_hindpoolk40, K=138)" | tee -a $SA/queue.log
