#!/bin/bash
# 7B student: SFT on the SAME 100 expert trajectories as the 3B student (1 epoch), then greedy eval on unseen + seen (274).
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; O=reports/alfworld/student7b_low100; M=reports/alfworld/self_evolve7b/queue.log
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun --jobid=2543688 --overlap -n1 -c4 --mem=0 $P -u scripts/alfworld_sft.py --traj reports/alfworld/expert_all_train_full.jsonl --max_games 100 --epochs 1 --bs 4 --grad_accum 4 --model Qwen/Qwen2.5-7B-Instruct --out $O > $O/sft.log 2>&1 || { echo "=== student7b SFT FAILED" >> $M; exit 1; }
srun --jobid=2543688 --overlap -n1 -c4 --mem=0 $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-7B-Instruct --adapter $O/final --split valid_unseen --budget 50 --temperature 0 --out $O/eval_unseen.jsonl --tag s7b > $O/eval.log 2>&1 &
srun --jobid=2543689 --overlap -n1 -c4 --mem=0 $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-7B-Instruct --adapter $O/final --split valid_seen --budget 50 --temperature 0 --out $O/eval_seen.jsonl --tag s7b >> $O/eval.log 2>&1 &
wait
echo "=== $(date '+%F %T') student7b_low100 (SFT 100 demos, 1 epoch): unseen $(grep -c '"won": true' $O/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $O/eval_seen.jsonl)/140 | 7B zero-shot 60/134 50/140; 3B student 56/134" >> $M
