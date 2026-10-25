#!/bin/bash
# same R2 trajectories / proposals / teacher / scan as selfattr_hindpool/r2 (generic evidence, 182 / 774 514); ONLY the filter evidence
# changes to the ALFWorld-specific one (old main chain rose 166->189->210 with it).  K=100 + 100 replay.
cd /users/8/pan00389/multi-agent/error-attribution; SA=reports/alfworld/self_attr_rounds
STEP=logfix STEP_POOL=1 TEACHER_NO_DIAG=1 K=100 CHAIN_SUFFIX=_hindpoolspec SEED=0 bash scripts/self_attr_round.sh selfattr 2 $(cat $SA/selfattr_hind/r1/ADAPTER) reports/alfworld/self_attr/r0/both/adapter any
echo "=== $(date '+%F %T') R2 same-trajectory control DONE (selfattr_hindpoolspec: specific evidence)" | tee -a $SA/queue.log
