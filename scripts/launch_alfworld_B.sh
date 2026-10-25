#!/bin/bash
# When hold $1 starts: (a) 3B student greedy on 134 unseen games, (b) 14B teacher greedy on unseen, (c) Model A (7B full, 7B no_obs) on collected student failures.
J=$1; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
until [ "$(squeue -j $J -h -o %t 2>/dev/null)" = "R" ]; do sleep 30; done; sleep 15
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SR="srun --jobid=$J --overlap -n1 -c4 --mem=0"; L=reports/alfworld/stepB.log
echo "=== $(date) hold $J node $(squeue -j $J -h -o %N) GPU $($SR nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tail -1)" | tee -a $L
echo "=== $(date) [a] 3B student greedy, valid_unseen" | tee -a $L
$SR $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --split valid_unseen --budget 30 --temperature 0 --out reports/alfworld/eval_unseen_student3b_zeroshot.jsonl --tag s3b0 >> $L 2>&1
echo "=== $(date) [b] 14B teacher greedy, valid_unseen" | tee -a $L
$SR $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-14B-Instruct --split valid_unseen --budget 30 --temperature 0 --out reports/alfworld/eval_unseen_teacher14b_zeroshot.jsonl --tag t14b >> $L 2>&1
echo "=== $(date) [c] Model A on student failures collected so far" | tee -a $L
$SR $P -u scripts/alfworld_attribute.py --attributor reports/attributor_7b_full/final --release reports/alfworld/student3b_train.release.jsonl --out reports/alfworld/attr7bfull_student3b_train.jsonl >> $L 2>&1
$SR $P -u scripts/alfworld_attribute.py --attributor reports/attributor_7b_no_observations/final --variant no_observations --release reports/alfworld/student3b_train.release.jsonl --out reports/alfworld/attr7bnoobs_student3b_train.jsonl >> $L 2>&1
echo "=== $(date) step B batch finished" | tee -a $L
