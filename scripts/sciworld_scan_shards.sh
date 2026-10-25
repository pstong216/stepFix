#!/bin/bash
# 3 parallel shards of the ScienceWorld evidence scan (resumable per (id,t)); afterwards relaunch the round-1 driver.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; R=reports/sciworld/round1; STU=reports/sciworld/student3/final
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pick() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge 12000 ] 2>/dev/null && echo $best; }
for i in 0 1 2; do
  ( until grep -q "DONE scan" $R/scan_s$i.log 2>/dev/null; do H=""; until H=$(pick); [ -n "$H" ]; do sleep 60; done
      srun --jobid=$H --overlap -n1 -c4 --mem=0 $P -u scripts/sciworld_evidence.py scan --traj $R/traj_train.jsonl --out_dir $R --k 50 --n_samples 6 --adapter $STU --shard $i 3 >> $R/scan_s$i.log 2>&1 || sleep 60; done ) &
  sleep 90
done
wait
echo "DONE scan" >> $R/scan.log; echo "DONE scan" >> $R/scan_s0.log
echo "=== $(date '+%F %T') sciworld scan shards done: $(wc -l < $R/scan.jsonl) rows" | tee -a reports/sciworld/queue.log
tmux new-session -d -s sciworld_r1 "bash scripts/sciworld_round.sh 50"
