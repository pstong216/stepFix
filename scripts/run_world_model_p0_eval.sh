#!/bin/bash
# Evaluate the world model checkpoint saved at 20k training rows (training stopped early by request).
H=$1; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; O=reports/alfworld/world_model/p0; M=$O/queue.log
echo "=== $(date '+%F %T') training stopped at 20k rows by request; eval 20k checkpoint" >> $M
srun --jobid=$H --overlap -n1 -c4 --mem=0 $P -u scripts/alfworld_world_model.py eval --adapter $O/adapter --out $O >> $O/eval.log 2>&1 \
 && echo "=== $(date '+%F %T') world model p0 eval done" >> $M || echo "=== $(date '+%F %T') eval FAILED" >> $M
