#!/bin/bash
# ScienceWorld: weak student (3 gold demos per task type) and strong teacher (all 702), then greedy dev eval of both.
H=$1; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; SR="srun --jobid=$H --overlap -n1 -c4 --mem=0"
O=reports/sciworld; M=$O/queue.log; log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
until grep -q "world model p0 eval done\|eval FAILED" reports/alfworld/world_model/p0/queue.log; do sleep 60; done
log "student SFT (3/task)"; $SR $P -u scripts/sciworld_agent.py sft --per_task 3 --epochs 3 --out $O/student3 >> $O/sft_student.log 2>&1 || { log "student SFT FAILED"; exit 1; }
log "student dev eval"; $SR $P -u scripts/sciworld_agent.py collect --adapter $O/student3/final --split dev --per_task 5 --out $O/eval_dev_student3.jsonl --tag student3 >> $O/eval_student.log 2>&1 || { log "student eval FAILED"; exit 1; }
log "student dev: $(tail -1 $O/eval_student.log)"
log "teacher SFT (all)"; $SR $P -u scripts/sciworld_agent.py sft --per_task 30 --epochs 2 --out $O/teacher_all >> $O/sft_teacher.log 2>&1 || { log "teacher SFT FAILED"; exit 1; }
log "teacher dev eval"; $SR $P -u scripts/sciworld_agent.py collect --adapter $O/teacher_all/final --split dev --per_task 5 --out $O/eval_dev_teacher.jsonl --tag teacher >> $O/eval_teacher.log 2>&1 || { log "teacher eval FAILED"; exit 1; }
log "teacher dev: $(tail -1 $O/eval_teacher.log)"
