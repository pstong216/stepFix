#!/bin/bash
# User (Oct 6 11:50): test the 3B Model A on R2 built on the previous Model-A R1 (old main chain selfattr/r1, R2 = 189 with 7B Model A).
# Same R2 trajectories, same A1 attributor steps, same old recipe (ALFWorld-specific evidence, env_top 86 + 86 replay); ONLY the teacher's
# diagnosis changes 7B -> 3B Model A.  The 7B arm is retrained with the new trainer from its original teacher outputs for a fair pair.
cd /users/8/pan00389/multi-agent/error-attribution; SA=reports/alfworld/self_attr_rounds; R1=$(cat $SA/selfattr/r1/ADAPTER); AT=$(cat $SA/selfattr/r1/ATTR)
( K=86 CHAIN_SUFFIX=_oldA7bnew SEED=0 bash scripts/self_attr_round.sh selfattr 2 $R1 $AT any; echo "=== $(date '+%F %T') old-recipe R2, 7B Model A, new trainer DONE (selfattr_oldA7bnew)" | tee -a $SA/queue.log ) &
MODELA_PATH=reports/attributor_no_observations/final K=86 CHAIN_SUFFIX=_oldA3b SEED=0 bash scripts/self_attr_round.sh selfattr 2 $R1 $AT any
echo "=== $(date '+%F %T') old-recipe R2, 3B Model A DONE (selfattr_oldA3b)" | tee -a $SA/queue.log
wait
