#!/bin/bash
# Method 5: planner-free progress estimator -> pointer -> the attr_nofb recipe (online verify + KL), round 1.
H=$1; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SR="srun --jobid=$H --overlap -n1 -c4 --mem=0"; PM=reports/alfworld/progress_model; M=reports/alfworld/opd_plus/queue.log; mkdir -p $PM
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
until grep -q "OPD+ QUEUE FINISHED" $M 2>/dev/null; do sleep 60; done
[ "$(squeue -j $H -h -o %T)" = "RUNNING" ] || { log "M5 skipped: hold not running"; exit 1; }
T=reports/alfworld/student3b_low100_train_b50.jsonl; A=reports/alfworld/opd/student3b_low100; IDS=$A/attr_nofb/steps.jsonl
[ -f $PM/model/adapter/adapter_config.json ] || { log "M5 train progress model"; $SR $P -u scripts/alfworld_progress_model.py train --n 4000 --out $PM/model >> $PM/log 2>&1 || { log "M5 train FAILED"; exit 1; }; }
log "M5 score 319 round-1 failures"; $SR $P -u scripts/alfworld_progress_model.py score --adapter $PM/model/adapter --ids_from $IDS --out $PM/scores_r1.jsonl >> $PM/log 2>&1 || { log "M5 score FAILED"; exit 1; }
$P scripts/alfworld_progress_model.py pointer --scores $PM/scores_r1.jsonl --out $PM/pointer_r1.jsonl --delta 1.0 >> $PM/log 2>&1
O=$A/progress_nofb; mkdir -p $O; L=$O/run.log; log "M5 OPD arm start"
$P -u scripts/alfworld_attr_opd.py choose --traj $T --arm attr --attr $PM/pointer_r1.jsonl --ids_from $IDS --max_failures 1000 --out $O/steps.jsonl >> $L 2>&1 || { log "M5 choose FAILED"; exit 1; }
$SR $P -u scripts/alfworld_attr_opd.py teacher --traj $T --steps $O/steps.jsonl --teacher Qwen/Qwen2.5-3B-Instruct --teacher_adapter reports/alfworld/student3b_sft/final --out $O/targets.jsonl >> $L 2>&1 || { log "M5 teacher FAILED"; exit 1; }
n0=$(grep -c "DONE distill" $L)
for c in 1 2 3 4 5 6 7 8; do
  $SR $P -u scripts/alfworld_attr_opd.py distill --traj $T --targets $O/targets.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter reports/alfworld/student3b_low100/final --out $O/round1 --verify_k 2 --exclude_a_minus --max_examples_per_run 250 >> $L 2>&1 || { log "M5 distill FAILED"; exit 1; }
  [ "$(grep -c "DONE distill" $L)" -gt "$n0" ] && break
done
$SR $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $O/round1/adapter --split valid_unseen --budget 50 --temperature 0 --out $O/eval_unseen_round1.jsonl --tag opd_progress_nofb >> $L 2>&1
log "M5 progress_nofb unseen: $(grep -c '"won": true' $O/eval_unseen_round1.jsonl)/$(wc -l < $O/eval_unseen_round1.jsonl)"
scripts/alfworld_eval_extended.sh $H opd_progress_nofb $O/round1/adapter; log "M5 done"
