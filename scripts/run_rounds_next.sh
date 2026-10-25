#!/bin/bash
# Resume round 3 of the three chains on the Oct-4 holds (after 2705994/2705996 expire ~02:39 Oct 4).  Every stage of
# self_evolve_round.sh resumes from its files (reports/ now lives on scratch.global).
cd /users/8/pan00389/multi-agent/error-attribution; B=reports/alfworld/self_evolve3b_rounds; M=$B/queue.log
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
hold() { squeue -u pan00389 -h -o "%i %j %T" | awk -v n="$1" '$2==n && $3=="RUNNING"{print $1}' | sort -n | tail -n1; }
until [ -z "$(squeue -j 2705994 -h -o %T 2>/dev/null | grep RUNNING)" ] && [ -n "$(hold error_attr_hold_igpu)" ] && [ -n "$(hold error_attr_hold_igpu_qw)" ]; do sleep 120; done
HA=$(hold error_attr_hold_igpu); HB=$(hold error_attr_hold_igpu_qw); log "R3 resume on holds A=$HA B=$HB"
tmux kill-session -t rounds_r3 2>/dev/null
# a distill log that ended with "nothing left" because the candidate stage failed must not short-circuit the rollout chain
for c in rollout; do grep -q "nothing left" $B/$c/r3/distill.log 2>/dev/null && [ ! -f $B/$c/r3/model_rollout/adapter/adapter_config.json ] && rm -f $B/$c/r3/distill.log; done
( bash scripts/self_evolve_round.sh rollout 3 $(cat $B/rollout/r2/ADAPTER) $(cat $B/rollout/r2/ESTIMATOR) $HA ) &
( bash scripts/self_evolve_round.sh env 3 $(cat $B/env/r2/ADAPTER) $(cat $B/env/r2/ESTIMATOR) $HB
  bash scripts/self_evolve_round.sh placebo 3 $(cat $B/placebo/r2/ADAPTER) $(cat $B/placebo/r2/ESTIMATOR) $HB ) &
wait; log "ROUNDS R3 (resumed on the Oct-4 holds) DONE"
