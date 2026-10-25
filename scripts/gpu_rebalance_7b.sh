#!/bin/bash
# Oct 11 15:35: 7B-first GPU rebalancer. Jobs renamed sw7b_gpu are 7B-only. Every 2 min: a non-core sw7b_gpu job whose GPU has >= 25 GB free
# on two consecutive checks (7B shards are not using it) goes back to tcded_gpu so the 3B chains can use the surplus. When all three 7B R2
# tuning evals are complete, every non-core job goes back. Core 7B reservations (kept): 3402382 3402383 3409005.
cd /users/8/pan00389/multi-agent/error-attribution; S7=reports/sciworld/stepfix7b; CORE=" 3402382 3402383 3409005 "; declare -A seen
while true; do
  done7=0; for n in w7_2b3 w7_2b1 w7_2b3lr; do grep -q "$n 848:" $S7/queue.log && done7=$((done7+1)); done
  for H in $(squeue -u pan00389 -h -t RUNNING -o "%i %j" | awk '$2=="sw7b_gpu"{print $1}'); do
    case "$CORE" in *" $H "*) continue;; esac
    if [ $done7 = 3 ]; then scontrol update JobId=$H JobName=tcded_gpu; echo "$(date +%T) $H -> tcded_gpu (7B evals done)"; continue; fi
    N=$(squeue -h -j $H -o %N); m=$(timeout 20 srun --jobid=$H --overlap -n1 -c1 --mem=0 -w $N nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits </dev/null 2>/dev/null | head -1)
    if [ "${m:-0}" -ge 25000 ] 2>/dev/null; then seen[$H]=$(( ${seen[$H]:-0} + 1 )); else seen[$H]=0; fi
    if [ "${seen[$H]}" -ge 2 ]; then scontrol update JobId=$H JobName=tcded_gpu; echo "$(date +%T) $H -> tcded_gpu (surplus ${m} MiB)"; seen[$H]=0; fi
  done
  [ $done7 = 3 ] && break; sleep 120; done
