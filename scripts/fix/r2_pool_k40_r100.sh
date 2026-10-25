#!/bin/bash
# R2 same trajectories: adjudication, corrections K = 40% of failures (138), replay FIXED at 100 (isolates the replay-ratio question)
cd /users/8/pan00389/multi-agent/error-attribution; SA=reports/alfworld/self_attr_rounds
EVIDENCE=generic STEP=logfix STEP_POOL=1 TEACHER_NO_DIAG=1 K_FRAC=0.4 REPLAY_N=100 CHAIN_SUFFIX=_hindpoolk40r100 SEED=0 bash scripts/self_attr_round.sh selfattr 2 $(cat $SA/selfattr_hind/r1/ADAPTER) reports/alfworld/self_attr/r0/both/adapter any
echo "=== $(date '+%F %T') R2 same-trajectory control DONE (selfattr_hindpoolk40r100: K=40% corrections, replay 100)" | tee -a $SA/queue.log
