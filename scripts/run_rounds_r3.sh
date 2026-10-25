#!/bin/bash
# Round 3 of the three chains on the Oct-3 holds (after the home-quota failure; reports/ now lives on scratch.global).
cd /users/8/pan00389/multi-agent/error-attribution; B=reports/alfworld/self_evolve3b_rounds; M=$B/queue.log
echo "=== $(date '+%F %T') R3 relaunch on holds A=2705994 B=2705996 (reports moved to scratch.global)" | tee -a $M
( bash scripts/self_evolve_round.sh rollout 3 $(cat $B/rollout/r2/ADAPTER) $(cat $B/rollout/r2/ESTIMATOR) 2705994 ) &
( bash scripts/self_evolve_round.sh env 3 $(cat $B/env/r2/ADAPTER) $(cat $B/env/r2/ESTIMATOR) 2705996
  bash scripts/self_evolve_round.sh placebo 3 $(cat $B/placebo/r2/ADAPTER) $(cat $B/placebo/r2/ESTIMATOR) 2705996 ) &
wait; echo "=== $(date '+%F %T') ROUNDS R3 DONE" | tee -a $M
