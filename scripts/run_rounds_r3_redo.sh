#!/bin/bash
# Continue R3 on the next holds.  rollout/r3: if its pointer file is incomplete (<300 states; the Oct-3 run scored only 15/357 after
# its scorer steps were killed), drop the downstream artifacts and let self_evolve_round.sh resume from the (resumable) scoring step,
# reusing the collected trajectories and refreshed M5 adapter; otherwise resume in place.  Then env R3 / placebo R3 if not logged.
cd /users/8/pan00389/multi-agent/error-attribution; B=reports/alfworld/self_evolve3b_rounds; M=$B/queue.log
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
hold() { squeue -u pan00389 -h -o "%i %j %T" | awk -v n="$1" '$2==n && $3=="RUNNING"{print $1}' | sort -n | tail -n1; }
until [ -z "$(squeue -j 2705994 -h -o %T 2>/dev/null | grep RUNNING)" ] && [ -n "$(hold error_attr_hold_igpu)" ] && [ -n "$(hold error_attr_hold_igpu_qw)" ]; do sleep 120; done
HA=$(hold error_attr_hold_igpu); HB=$(hold error_attr_hold_igpu_qw); log "R3 continue on holds A=$HA B=$HB"
tmux kill-session -t rounds_r3 2>/dev/null
D=$B/rollout/r3; NS=$(wc -l < $D/self/steps.jsonl 2>/dev/null || echo 0)
if [ "$NS" -lt 300 ] && ! grep -q "CHAIN rollout R3: .*verified [0-9]" $M; then
  log "rollout r3: pointer file has $NS states -> dropping downstream artifacts, rescoring (trajectories + M5 adapter reused)"
  rm -f $D/self/steps.jsonl $D/self/targets.jsonl $D/cands.log $D/distill.log $D/eval*.jsonl $D/eval.log $D/ADAPTER $D/ESTIMATOR $D/m5/pointer.jsonl; rm -rf $D/model_rollout; mkdir -p $D/self
fi
( grep -q "CHAIN rollout R3: .*verified [0-9]" $M || bash scripts/self_evolve_round.sh rollout 3 $(cat $B/rollout/r2/ADAPTER) $(cat $B/rollout/r2/ESTIMATOR) $HA ) &
( grep -q "CHAIN env R3" $M || bash scripts/self_evolve_round.sh env 3 $(cat $B/env/r2/ADAPTER) $(cat $B/env/r2/ESTIMATOR) $HB
  grep -q "CHAIN placebo R3" $M || bash scripts/self_evolve_round.sh placebo 3 $(cat $B/placebo/r2/ADAPTER) $(cat $B/placebo/r2/ESTIMATOR) $HB ) &
wait; log "ROUNDS R3 CONTINUE DONE"
