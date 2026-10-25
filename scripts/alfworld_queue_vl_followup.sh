#!/bin/bash
# Runs after alfworld_queue_vl.sh prints QUEUE FINISHED, on the same hold $1 (strictly sequential GPU use).
# E. round-2 branch from attr_nofb/round1 with the PLANNER-refreshed pointer (v1): the clean reference for the
#    verify-refreshed branch (attrvl_nofb) and the stale branch (attr_nofb round 2) -- all three from one checkpoint.
H=$1; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SR="srun --jobid=$H --overlap -n1 -c4 --mem=0"; A=reports/alfworld/opd/student3b_low100; M=reports/alfworld/queue_vl.log
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
until grep -q "QUEUE FINISHED" $M; do sleep 60; done
[ "$(squeue -j $H -h -o %T)" = "RUNNING" ] || { log "E skipped: hold $H not running"; exit 1; }

O=$A/attrpl_nofb; mkdir -p $O; L=$O/round2.log; T2=reports/alfworld/student3b_low100_round2_attr_nofb_train_b50.jsonl
log "E start (planner-refreshed pointer, from attr_nofb/round1, same 528 round-2 failures, same teacher)"
$P -u scripts/alfworld_attr_opd.py choose --traj $T2 --arm attr --attr reports/alfworld/attr7balf_student3b_low100_round2_attr_nofb.jsonl --max_failures 1000 --out $O/steps_round2.jsonl >> $L 2>&1 || { log "E FAILED choose"; exit 1; }
$SR $P -u scripts/alfworld_attr_opd.py teacher --traj $T2 --steps $O/steps_round2.jsonl --teacher Qwen/Qwen2.5-3B-Instruct --teacher_adapter reports/alfworld/student3b_sft/final --out $O/targets_round2.jsonl >> $L 2>&1 || { log "E FAILED teacher"; exit 1; }
n0=$(grep -c "DONE distill" $L)
for chunk in 1 2 3 4 5 6 7 8; do
  $SR $P -u scripts/alfworld_attr_opd.py distill --traj $T2 --targets $O/targets_round2.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter $A/attr_nofb/round1/adapter --out $O/round2 --verify_k 2 --exclude_a_minus --max_examples_per_run 250 >> $L 2>&1 || { log "E FAILED distill"; exit 1; }
  [ "$(grep -c "DONE distill" $L)" -gt "$n0" ] && break
done
$SR $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $O/round2/adapter --split valid_unseen --budget 50 --temperature 0 --out $O/eval_unseen_round2.jsonl --tag round2_attrpl_nofb >> $L 2>&1
log "round2_attrpl_nofb unseen: $(grep -c '"won": true' $O/eval_unseen_round2.jsonl)/$(wc -l < $O/eval_unseen_round2.jsonl)"
scripts/alfworld_eval_extended.sh $H opd_attrpl_nofb_r2 $O/round2/adapter
log "E done; FOLLOWUP FINISHED"
