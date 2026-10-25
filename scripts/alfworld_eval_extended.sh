#!/bin/bash
# Extended greedy evaluation (budget 50) of one policy on: valid_seen (140) + held-out train games 1500-1999 (500, untouched by SFT/OPD).
# Usage: alfworld_eval_extended.sh <hold> <tag> [adapter]
J=$1; TAG=$2; AD=${3:-}; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; SR="srun --jobid=$J --overlap -n1 -c4 --mem=0"; L=reports/alfworld/eval_extended.log
A=""; [ -n "$AD" ] && A="--adapter $AD"
echo "=== $(date) extended eval $TAG ($AD) on hold $J" | tee -a $L
$SR $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct $A --split valid_seen --budget 50 --temperature 0 --out reports/alfworld/ext/eval_seen_$TAG.jsonl --tag $TAG >> $L 2>&1
$SR $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct $A --split train --game_offset 1500 --n_games 500 --budget 50 --temperature 0 --out reports/alfworld/ext/eval_heldout_$TAG.jsonl --tag $TAG >> $L 2>&1
echo "=== $(date) extended eval $TAG done: seen $(grep -c '"won": true' reports/alfworld/ext/eval_seen_$TAG.jsonl)/$(wc -l < reports/alfworld/ext/eval_seen_$TAG.jsonl), heldout $(grep -c '"won": true' reports/alfworld/ext/eval_heldout_$TAG.jsonl)/$(wc -l < reports/alfworld/ext/eval_heldout_$TAG.jsonl)" | tee -a $L
