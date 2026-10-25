#!/bin/bash
# Scale up counterfactual (injected-error) failures, AgenTracer / Who&When-Pro style: one injection process per trajectory
# collection (own wins), 1000 attempts each, spread over whichever GPU holds have >= 10 GB free.  Output self_attr_labels/scale/.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; O=reports/alfworld/self_attr_labels/scale; mkdir -p $O
B=reports/alfworld/self_evolve3b_rounds; M=reports/alfworld/self_evolve3b/queue.log; export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
pick() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge "${1:-10000}" ] 2>/dev/null && echo $best; }
# (collection, adapter that played it) -- the injection is continued by the SAME policy that produced the win
run() { local name=$1 traj=$2 ad=$3 seed=$4
  until grep -q "DONE synth" $O/$name.log 2>/dev/null; do H=""; until H=$(pick 10000); [ -n "$H" ]; do sleep 90; done
    srun --jobid=$H --overlap -n1 -c4 --mem=0 $P -u scripts/synth_failures.py --traj $traj --adapter $ad --n 1000 --seed $seed --out $O/$name.jsonl >> $O/$name.log 2>&1 || sleep 60; done
  log "synth scale $name: $(grep DONE $O/$name.log | tail -n1 | cut -c1-160)"; }
run orig    reports/alfworld/student3b_low100_train_b50.jsonl reports/alfworld/student3b_low100/final 11 &
sleep 120; run env_r2  $B/env/r2/traj.jsonl  reports/alfworld/self_evolve3b/proxy_arms/m5_plain_env_top/model/adapter 12 &
sleep 120; run env_r3  $B/env/r3/traj.jsonl  $B/env/r2/model/adapter 13 &
sleep 120; run roll_r2 $B/rollout/r2/traj.jsonl $B/rollout/r1/model_rollout/adapter 14 &
sleep 120; run roll_r3 $B/rollout/r3/traj.jsonl $B/rollout/r2/model_rollout/adapter 15 &
wait; log "SYNTH SCALE DONE: $(cat $O/*.jsonl | wc -l) synthetic failures"
