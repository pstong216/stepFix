#!/bin/bash
# Use the ee5239 interactive hold for the (resumable) compression step once the M5/const0 round-2 queue is done;
# stop as soon as the A100 batch job 2422047 starts (it resumes from the same cache).
H=$1; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; O=reports/modelA_compressed
[ -n "$2" ] && until grep -q "ALL DONE\|FAILED" reports/alfworld/opd/student3b_low100/progress_nofb/round2_queue.log; do sleep 60; done
[ "$(squeue -j 2422047 -h -o %T)" = "PENDING" ] || exit 0
echo "=== $(date '+%F %T') compression on hold $H" >> $O/compress.log
srun --jobid=$H --overlap -n1 -c4 --mem=0 $P -u scripts/compress_bodies.py generate --batch 16 >> $O/compress.log 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do [ "$(squeue -j 2422047 -h -o %T)" = "RUNNING" ] && { kill $PID; echo "=== $(date '+%F %T') batch job started; hold compression stopped" >> $O/compress.log; }; sleep 60; done
