#!/bin/bash
# Relaunch the seed-2 final chain from round 2 after repairing its corrupted round-2 trajectory files (Oct 5 03:20).
cd /users/8/pan00389/multi-agent/error-attribution; SA=reports/alfworld/self_attr_rounds; A0=reports/alfworld/self_attr/r0/both/adapter
S=$(cat $SA/selfattr_s2/r1/ADAPTER); AT=$A0
for R in 2 3; do CHAIN_SUFFIX=_s2 SEED=2 bash scripts/self_attr_round.sh selfattr $R $S $AT any || break; S=$(cat $SA/selfattr_s2/r$R/ADAPTER); AT=$(cat $SA/selfattr_s2/r$R/ATTR); done
echo "=== $(date '+%F %T') FINAL CHAIN selfattr_s2 DONE (relaunched after repair)" | tee -a $SA/queue.log
