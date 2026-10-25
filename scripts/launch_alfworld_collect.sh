#!/bin/bash
# When hold $1 starts: 2-game smoke (greedy), then the full student collection on the train split.
J=$1; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
until [ "$(squeue -j $J -h -o %t 2>/dev/null)" = "R" ]; do sleep 30; done; sleep 15
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== $(date) collect on hold $J node $(squeue -j $J -h -o %N) GPU $(srun --jobid=$J --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=name --format=csv,noheader | tail -1)" | tee -a reports/alfworld/collect.log
srun --jobid=$J --overlap -n1 -c4 --mem=0 $P -u scripts/alfworld_collect.py --split train --n_games 2 --budget 30 --temperature 0 --out reports/alfworld/_smoke.jsonl --tag smoke >> reports/alfworld/collect.log 2>&1 || { echo "SMOKE FAILED" | tee -a reports/alfworld/collect.log; exit 1; }
echo "=== $(date) smoke ok; full train collection" | tee -a reports/alfworld/collect.log
srun --jobid=$J --overlap -n1 -c4 --mem=0 $P -u scripts/alfworld_collect.py --split train --n_games 2500 --budget 30 --temperature 1.0 --out reports/alfworld/student3b_train.jsonl --tag student3b >> reports/alfworld/collect.log 2>&1
echo "=== $(date) collection finished (exit $?)" | tee -a reports/alfworld/collect.log
