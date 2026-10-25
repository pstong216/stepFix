#!/bin/bash
# Speed-up (Oct 6 20:15, R3 copy of the R2 script, same evaluation as the approved queue entry): 774-game eval of selfattr_a3bfinal R2 split into 4 shards of the
# 500 held-out train games (1500-1999, 125 each) on separate GPUs, started as soon as the R2 adapter exists (in parallel with the 274 eval).
# Episode ids/game files are identical to the single-process daemon run; shards are concatenated into eval_heldout_a3bfinal_r3.jsonl.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=reports/alfworld/self_attr_rounds/selfattr_a3bfinal/r3; AD=$R/model/adapter; E=reports/alfworld/ext; OUT=$E/eval_heldout_a3bfinal_r3.jsonl
pick() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge 10000 ] 2>/dev/null && echo $best; }
until [ -f $AD/adapter_config.json ]; do sleep 30; done; sleep 150   # let the two 274-eval processes load first
for i in 0 1 2 3; do off=$((1500 + 125 * i)); o=$E/eval_heldout_a3bfinal_r3_s$i.jsonl
  ( until [ "$(grep -c . $o 2>/dev/null)" -ge 125 ]; do H=""; until H=$(pick); [ -n "$H" ]; do sleep 60; done
      srun --jobid=$H --overlap -n1 -c4 --mem=0 $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $AD --split train --game_offset $off --n_games 125 --budget 50 --temperature 0 --out $o --tag h500_a3bfinal_r3 >> $E/heldout_a3bfinal_r3_s$i.log 2>&1 || sleep 30; done ) &
  sleep 60; done; wait
cat $E/eval_heldout_a3bfinal_r3_s{0,1,2,3}.jsonl > $OUT.tmp && mv $OUT.tmp $OUT
cat $E/eval_heldout_a3bfinal_r3_s{0,1,2,3}.release.jsonl > $E/eval_heldout_a3bfinal_r3.release.jsonl 2>/dev/null
until [ -s $R/ADAPTER ]; do sleep 30; done
w=$(grep -c '"won": true' $OUT); t=$(( $(grep -c '"won": true' $R/eval_unseen.jsonl) + $(grep -c '"won": true' $R/eval_seen.jsonl) ))
echo "=== $(date '+%F %T') heldout500 a3bfinal_r3 (4 shards): $w/$(grep -c . $OUT) (student 280/500) | 274: $t | 774 total: $((w + t))" | tee -a reports/alfworld/self_attr_rounds/queue.log reports/alfworld/eval_extended.log
