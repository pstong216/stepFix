#!/bin/bash
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; O=reports/alfworld/self_evolve7b
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ev() { local a=$1 lr=$2 J=$3; srun --jobid=$J --overlap -n1 -c4 --mem=0 $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-7B-Instruct --adapter $O/$a/sweep_lr$lr/adapter --split valid_unseen --budget 50 --temperature 0 --out $O/$a/eval_unseen_sweep_lr$lr.jsonl --tag sw_${a}_$lr > $O/$a/eval_sweep_lr$lr.log 2>&1
  echo "=== $(date '+%F %T') sweep $a lr$lr: unseen $(grep -c '"won": true' $O/$a/eval_unseen_sweep_lr$lr.jsonl)/$(wc -l < $O/$a/eval_unseen_sweep_lr$lr.jsonl) (zero-shot 60/134)" >> $O/queue.log; }
(ev placebo 1e-6 2467981; ev placebo 3e-6 2467981) & (ev attr 1e-6 2467982; ev attr 3e-6 2467982) & wait
echo "=== sweep eval DONE" >> $O/queue.log
