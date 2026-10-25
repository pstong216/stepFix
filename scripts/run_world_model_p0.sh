#!/bin/bash
# World model phase 0: train the 4-head world model on the qianwen hold once the const0 R1 evaluation is done, then evaluate.
H=$1; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SR="srun --jobid=$H --overlap -n1 -c4 --mem=0"; O=reports/alfworld/world_model/p0; mkdir -p $O; M=$O/queue.log
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
until grep -q "const0 R1 done\|const0 R1 FAILED" reports/alfworld/opd/student3b_low100/const0_queue.log; do sleep 60; done
[ "$(squeue -j $H -h -o %T)" = "RUNNING" ] || { log "hold $H not running"; exit 1; }
log "train world model"; $SR $P -u scripts/alfworld_world_model.py train --out $O >> $O/train.log 2>&1 || { log "train FAILED"; exit 1; }
log "eval world model"; $SR $P -u scripts/alfworld_world_model.py eval --adapter $O/adapter --out $O >> $O/eval.log 2>&1 || { log "eval FAILED"; exit 1; }
log "world model p0 done: $(tr -d '\n ' < $O/eval_summary.json | cut -c1-400)"
