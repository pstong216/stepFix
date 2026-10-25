#!/bin/bash
# Resume the multi-round chains on the next holds (after 2644179/2644180 expire at ~02:38 Oct 3).  Every stage of
# self_evolve_round.sh resumes from its files, so re-running a round with the new hold id continues where it stopped.
#   hold A (ee5239):  rollout R2 -> rollout R3
#   hold B (qianwen): placebo R2 -> env R3 -> placebo R3
cd /users/8/pan00389/multi-agent/error-attribution; B=reports/alfworld/self_evolve3b_rounds; M=$B/queue.log; EST1=reports/alfworld/progress_model/model/adapter
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
hold() { squeue -u pan00389 -h -o "%i %j %T" | awk -v n="$1" '$2==n && $3=="RUNNING"{print $1}' | sort -n | tail -n1; }
until [ -z "$(squeue -j 2644179 -h -o %T 2>/dev/null | grep RUNNING)" ] && [ -n "$(hold error_attr_hold_igpu)" ] && [ -n "$(hold error_attr_hold_igpu_qw)" ]; do sleep 120; done
HA=$(hold error_attr_hold_igpu); HB=$(hold error_attr_hold_igpu_qw); log "resume on holds A=$HA B=$HB"
for s in rounds yield_arms literal_opd attr3b stepxfilter; do tmux kill-session -t $s 2>/dev/null; done
( bash scripts/self_evolve_round.sh rollout 2 $(cat $B/rollout/r1/ADAPTER) $EST1 $HA
  bash scripts/self_evolve_round.sh rollout 3 $(cat $B/rollout/r2/ADAPTER) $(cat $B/rollout/r2/ESTIMATOR) $HA ) &
( bash scripts/self_evolve_round.sh placebo 2 $(cat $B/placebo/r1/ADAPTER) $EST1 $HB
  bash scripts/self_evolve_round.sh env 3 $(cat $B/env/r2/ADAPTER) $(cat $B/env/r2/ESTIMATOR) $HB
  bash scripts/self_evolve_round.sh placebo 3 $(cat $B/placebo/r2/ADAPTER) $(cat $B/placebo/r2/ESTIMATOR) $HB ) &
wait; log "ROUNDS RESUME (R2 rollout/placebo, R3 env/rollout/placebo) DONE"
