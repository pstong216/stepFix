#!/bin/bash
# Greedy replay of a FIXED set of 300 training games (first 300 of the train split) by the student and each chain round's adapter.
# Answers: did the 274-game gains show up on the training distribution when decoding greedily (the 1500-game collections are T=1.0)?
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; B=reports/alfworld/self_evolve3b_rounds; M=$B/queue.log
O=$B/greedy300; mkdir -p $O; export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
hold() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; echo $best; }
S() { local H; until H=$(hold); [ -n "$H" ]; do sleep 120; done; echo "srun --jobid=$H --overlap -n1 -c4 --mem=0"; }
guard() { until [ "$($(S) nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1)" -ge "$1" ] 2>/dev/null; do sleep 60; done; }
run() { local name=$1 ad=$2; local out=$O/$name.jsonl
  until [ "$(grep -c . $out 2>/dev/null)" -ge 300 ]; do guard 10000; $(S) $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct ${ad:+--adapter $ad} --split train --n_games 300 --budget 50 --temperature 0 --out $out --tag g300_$name >> $O/$name.log 2>&1; sleep 10; done
  log "greedy300 $name: $(grep -c '"won": true' $out)/300 training games won (greedy)"; }
run student reports/alfworld/student3b_low100/final
for c in env rollout placebo; do for r in 1 2 3; do a=$(cat $B/$c/r$r/ADAPTER 2>/dev/null); [ -n "$a" ] && [ -f $a/adapter_config.json ] && run ${c}_r$r $a; done; done
log "GREEDY300 DONE"
