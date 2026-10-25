#!/bin/bash
# UQ signals on attr_nofb's 528 round-2 failures: four attributors, then the generating student (attr_nofb/round1).
H=$1; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SR="srun --jobid=$H --overlap -n1 -c4 --mem=0"; M=reports/alfworld/uq/queue_uq.log; mkdir -p reports/alfworld/uq
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
REL=reports/alfworld/student3b_low100_round2_attr_nofb_train_b50.release.jsonl
until [ "$(squeue -j $H -h -o %T)" = "RUNNING" ]; do sleep 30; done; log "hold $H running"
for spec in stale:attributor_7b_no_observations verify:attributor_7b_verify pl450:attributor_7b_pl450 v1:attributor_7b_alf; do
  n=${spec%%:*}; d=${spec##*:}; log "attributor $n start"
  $SR $P -u scripts/alfworld_uq.py attributor --release $REL --attributor reports/$d/final --out reports/alfworld/uq/attributor_${n}_r2_attr_nofb.jsonl >> $M.$n 2>&1 && log "attributor $n done" || log "attributor $n FAILED"
done
log "student start"
$SR $P -u scripts/alfworld_uq.py student --traj reports/alfworld/student3b_low100_round2_attr_nofb_train_b50.jsonl --adapter reports/alfworld/opd/student3b_low100/attr_nofb/round1/adapter --out reports/alfworld/uq/student_r2_attr_nofb.jsonl >> $M.student 2>&1 && log "student done" || log "student FAILED"
log "UQ QUEUE FINISHED"
