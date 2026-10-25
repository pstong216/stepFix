#!/bin/bash
# Lineage requested by the user (Oct 6 02:00): R1 = first-step rule (selfattr_hind/r1, the existing final R1) -> R2 = environment
# adjudication on that model's R2 trajectories (selfattr_hindpool/r2) -> R3 = environment adjudication on its own new trajectories.
cd /users/8/pan00389/multi-agent/error-attribution; SA=reports/alfworld/self_attr_rounds
until [ -f $SA/selfattr_hindpool/r2/ADAPTER ]; do sleep 120; done
EVIDENCE=generic STEP=logfix STEP_POOL=1 TEACHER_NO_DIAG=1 K=100 CHAIN_SUFFIX=_hindpool SEED=0 bash scripts/self_attr_round.sh selfattr 3 $(cat $SA/selfattr_hindpool/r2/ADAPTER) $(cat $SA/selfattr_hindpool/r2/ATTR) any
echo "=== $(date '+%F %T') FIRST-step R1 -> adjudication R2 -> adjudication R3 chain DONE (selfattr_hindpool)" | tee -a $SA/queue.log
