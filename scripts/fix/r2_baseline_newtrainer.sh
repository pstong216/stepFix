#!/bin/bash
# baseline for the Model A check retrained with the NEW trainer (the original selfattr_hindpoolk40r100 run lost 2 batches to the
# old bf16 accumulation bug): same data, no Model A, K=40% + replay 100
cd /users/8/pan00389/multi-agent/error-attribution; SA=reports/alfworld/self_attr_rounds
EVIDENCE=generic STEP=logfix STEP_POOL=1 TEACHER_NO_DIAG=1 K_FRAC=0.4 REPLAY_N=100 CHAIN_SUFFIX=_hindpoolk40r100new SEED=0 bash scripts/self_attr_round.sh selfattr 2 $(cat $SA/selfattr_hind/r1/ADAPTER) reports/alfworld/self_attr/r0/both/adapter any
echo "=== $(date '+%F %T') R2 baseline (no Model A, new trainer) DONE (selfattr_hindpoolk40r100new)" | tee -a $SA/queue.log
