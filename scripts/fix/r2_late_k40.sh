#!/bin/bash
# User hypothesis (Oct 6 11:30): R2 needs LATER pointers.  Same R2 trajectories; late-rule steps only (selfattr_hind/r2 teacher + scan);
# env top-K with the chosen budget (40% of failures, replay 100), new trainer.  Compare with adjudication K40 (no Model A, new trainer).
cd /users/8/pan00389/multi-agent/error-attribution; SA=reports/alfworld/self_attr_rounds
EVIDENCE=generic STEP=logfix TEACHER_NO_DIAG=1 K_FRAC=0.4 REPLAY_N=100 CHAIN_SUFFIX=_hindlatek40 SEED=0 bash scripts/self_attr_round.sh selfattr 2 $(cat $SA/selfattr_hind/r1/ADAPTER) reports/alfworld/self_attr/r0/both/adapter any
echo "=== $(date '+%F %T') R2 late-rule K40 DONE (selfattr_hindlatek40)" | tee -a $SA/queue.log
