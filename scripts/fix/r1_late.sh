#!/bin/bash
# R1 with the LATE (plateau:2) step rule (user Oct 5 22:40): same R1 trajectories / teacher (hindsight + own analysis) / env top-100 +
# replay as the final chain's R1; only the step differs (pointer precomputed -> the round script resumes from it).
cd /users/8/pan00389/multi-agent/error-attribution
EVIDENCE=generic STEP=logfix TEACHER_NO_DIAG=1 K=100 CHAIN_SUFFIX=_hindlate SEED=0 bash scripts/self_attr_round.sh selfattr 1 reports/alfworld/student3b_low100/final reports/alfworld/self_attr/r0/both/adapter any
echo "=== $(date '+%F %T') R1 LATE-rule arm DONE (selfattr_hindlate)" | tee -a reports/alfworld/self_attr_rounds/queue.log
