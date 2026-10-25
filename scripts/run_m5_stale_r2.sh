#!/bin/bash
# Control: score round-2 failures with the STALE (round-1) progress estimator, same pointer rule -> compare with refreshed.
H=$1; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PM=reports/alfworld/progress_model
T2=reports/alfworld/student3b_low100_round2_progress_nofb_train_b50.jsonl
srun --jobid=$H --overlap -n1 -c4 --mem=0 $P -u scripts/alfworld_progress_model.py score --traj $T2 --adapter $PM/model/adapter --out $PM/scores_r2_stale.jsonl >> $PM/log_r2_stale 2>&1 \
 && $P scripts/alfworld_progress_model.py pointer --scores $PM/scores_r2_stale.jsonl --out $PM/pointer_r2_stale.jsonl --delta 1.0 >> $PM/log_r2_stale 2>&1 \
 && echo "=== $(date '+%F %T') stale M5 pointer on R2 done" >> $PM/log_r2_stale || echo "=== $(date '+%F %T') stale M5 FAILED" >> $PM/log_r2_stale
