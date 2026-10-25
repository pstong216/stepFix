#!/bin/bash
# R1 adjudication with K = 40% of failures (same candidates as selfattr_pool/r1), to pair with the K=100 R1 (181) -- uses idle GPUs
cd /users/8/pan00389/multi-agent/error-attribution; SA=reports/alfworld/self_attr_rounds
EVIDENCE=generic STEP=logfix STEP_POOL=1 TEACHER_NO_DIAG=1 K_FRAC=0.4 CHAIN_SUFFIX=_poolk40 SEED=0 bash scripts/self_attr_round.sh selfattr 1 reports/alfworld/student3b_low100/final reports/alfworld/self_attr/r0/both/adapter any
echo "=== $(date '+%F %T') R1 adjudication K=40% arm DONE (selfattr_poolk40)" | tee -a $SA/queue.log
