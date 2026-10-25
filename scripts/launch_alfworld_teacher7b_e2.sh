#!/bin/bash
# 7B teacher SFT on all expert trajectories (hold $1) -> greedy eval on unseen.
J=$1; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; L=reports/alfworld/chain_teacher7b_e2.log; SR="srun --jobid=$J --overlap -n1 -c4 --mem=0"
echo "=== $(date) [1] SFT 7B teacher on all 1,338 expert trajectories, 2 epochs (hold $J, $(squeue -j $J -h -o %N))" | tee -a $L
$SR $P -u scripts/alfworld_sft.py --traj reports/alfworld/expert_all_train_full.jsonl --epochs 2 --bs 4 --grad_accum 4 --model Qwen/Qwen2.5-7B-Instruct --out reports/alfworld/teacher7b_sft_e2 >> $L 2>&1 || { echo "SFT FAILED" | tee -a $L; exit 1; }
echo "=== $(date) [2] greedy eval on valid_unseen" | tee -a $L
$SR $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-7B-Instruct --adapter reports/alfworld/teacher7b_sft_e2/final --split valid_unseen --budget 50 --temperature 0 --out reports/alfworld/eval_unseen_teacher7b_sft.jsonl --tag t7bsft2 >> $L 2>&1
echo "=== $(date) TEACHER7B_DONE" | tee -a $L
