#!/bin/bash
# all-adjudication chain R2 dropped (160 vs R1 181): rerun its R2 on the SAME trajectories with the budget that worked elsewhere
# (corrections 40% of failures, replay 100), from the same R1 adapter
cd /users/8/pan00389/multi-agent/error-attribution; SA=reports/alfworld/self_attr_rounds
EVIDENCE=generic STEP=logfix STEP_POOL=1 TEACHER_NO_DIAG=1 K_FRAC=0.4 REPLAY_N=100 CHAIN_SUFFIX=_poolk40r100 SEED=0 bash scripts/self_attr_round.sh selfattr 2 $(cat $SA/selfattr_pool/r1/ADAPTER) reports/alfworld/self_attr/r0/both/adapter any
echo "=== $(date '+%F %T') pool-chain R2 rerun DONE (selfattr_poolk40r100: K=40% corrections, replay 100)" | tee -a $SA/queue.log
