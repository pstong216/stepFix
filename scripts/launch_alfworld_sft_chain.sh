#!/bin/bash
# On hold $1, once planner-expert (budget 50) data exists: SFT -> greedy eval on unseen (budget 50) -> T=1 failure collection on train games 0-1499 (budget 50).
J=$1; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; L=reports/alfworld/sft_chain.log
until [ -f reports/alfworld/cpu_1439539.log ] && grep -q "PLANNER_DONE" reports/alfworld/cpu_1439539.log; do sleep 60; done
[ "$(squeue -j $J -h -o %t 2>/dev/null)" = "R" ] || { echo "hold $J gone; exiting" | tee -a $L; exit 1; }
SR="srun --jobid=$J --overlap -n1 -c4 --mem=0"
echo "=== $(date) SFT data: planner50 won $(grep -c '"won": true' reports/alfworld/expert_planner50_train.jsonl) / handcoded30 won $(grep -c '"won": true' reports/alfworld/expert_train.jsonl)" | tee -a $L
cat reports/alfworld/expert_planner50_train.jsonl reports/alfworld/expert_train.jsonl > reports/alfworld/expert_all_train.jsonl
echo "=== $(date) [1] SFT" | tee -a $L
$SR $P -u scripts/alfworld_sft.py --traj reports/alfworld/expert_all_train.jsonl --model Qwen/Qwen2.5-3B-Instruct --out reports/alfworld/student3b_sft --epochs 2 >> $L 2>&1 || { echo "SFT FAILED" | tee -a $L; exit 1; }
echo "=== $(date) [2] SFT student greedy on valid_unseen (budget 50)" | tee -a $L
$SR $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter reports/alfworld/student3b_sft/final --split valid_unseen --budget 50 --temperature 0 --out reports/alfworld/eval_unseen_sft3b_b50.jsonl --tag sft3b >> $L 2>&1
echo "=== $(date) [3] SFT student T=1 failure collection, train games 0-1499 (budget 50)" | tee -a $L
$SR $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter reports/alfworld/student3b_sft/final --split train --n_games 1500 --budget 50 --temperature 1.0 --out reports/alfworld/sft3b_train_b50.jsonl --tag sft3b >> $L 2>&1
echo "=== $(date) SFT chain finished" | tee -a $L
