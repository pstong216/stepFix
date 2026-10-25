#!/bin/bash
# M5 round 2, oracle-free: new trajectories from M5's round-1 student -> refresh the progress estimator on the new
# student's OWN wins -> pointer (same pre-registered rule) -> teacher + verify + distill -> eval.
# Then the round-2 "always step 0" control from the same start on the same failures.
H=$1; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; SR="srun --jobid=$H --overlap -n1 -c4 --mem=0"
A=reports/alfworld/opd/student3b_low100; O=$A/progress_nofb; PM=reports/alfworld/progress_model; M=$O/round2_queue.log
T2=reports/alfworld/student3b_low100_round2_progress_nofb_train_b50.jsonl; R1=$O/round1/adapter
TEACH="--teacher Qwen/Qwen2.5-3B-Instruct --teacher_adapter reports/alfworld/student3b_sft/final"
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
distill_loop() {  # traj targets from out log
  local n0; n0=$(grep -c "DONE distill" $5 2>/dev/null)
  for c in 1 2 3 4 5 6 7 8; do
    $SR $P -u scripts/alfworld_attr_opd.py distill --traj $1 --targets $2 --student Qwen/Qwen2.5-3B-Instruct --adapter $3 --out $4 --verify_k 2 --exclude_a_minus --max_examples_per_run 250 >> $5 2>&1 || return 1
    [ "$(grep -c "DONE distill" $5)" -gt "$n0" ] && return 0
  done; return 1; }
until [ "$(squeue -j $H -h -o %T)" = "RUNNING" ]; do sleep 30; done; log "hold $H running"
log "collect round-2 trajectories with M5's round-1 student"
for c in 1 2 3 4; do
  $SR $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $R1 --split train --n_games 1500 --budget 50 --temperature 1.0 --out $T2 --tag round2_progress_nofb --max_episodes_per_run 450 >> $M.collect 2>&1
  grep -q "^DONE" <(tail -3 $M.collect) && break
done
log "collected $(wc -l < $T2) games, $(grep -c '"won": true' $T2) won, $(grep -c '"won": false' $T2) failed"
[ -f $PM/model_r2/adapter/adapter_config.json ] || { log "refresh estimator on the new student's wins"; $SR $P -u scripts/alfworld_progress_model.py train --traj $T2 --adapter $PM/model/adapter --n 4000 --out $PM/model_r2 >> $PM/log_r2 2>&1 || { log "estimator FAILED"; exit 1; }; }
log "score round-2 failures"; $SR $P -u scripts/alfworld_progress_model.py score --traj $T2 --adapter $PM/model_r2/adapter --out $PM/scores_r2.jsonl >> $PM/log_r2 2>&1 || { log "score FAILED"; exit 1; }
$P scripts/alfworld_progress_model.py pointer --scores $PM/scores_r2.jsonl --out $PM/pointer_r2.jsonl --delta 1.0 >> $PM/log_r2 2>&1
L=$O/round2.log
$P -u scripts/alfworld_attr_opd.py choose --traj $T2 --arm attr --attr $PM/pointer_r2.jsonl --max_failures 1000 --out $O/steps_round2.jsonl >> $L 2>&1 \
 && $SR $P -u scripts/alfworld_attr_opd.py teacher --traj $T2 --steps $O/steps_round2.jsonl $TEACH --out $O/targets_round2.jsonl >> $L 2>&1 \
 && distill_loop $T2 $O/targets_round2.jsonl $R1 $O/round2 $L || { log "M5 R2 training FAILED"; exit 1; }
$SR $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $O/round2/adapter --split valid_unseen --budget 50 --temperature 0 --out $O/eval_unseen_round2.jsonl --tag round2_progress_nofb >> $L 2>&1
log "M5 R2 verified $(wc -l < $O/round2/examples.jsonl)/$(wc -l < $O/steps_round2.jsonl), unseen $(grep -c '"won": true' $O/eval_unseen_round2.jsonl)/134"
scripts/alfworld_eval_extended.sh $H opd_progress_nofb_r2 $O/round2/adapter; log "M5 R2 done"
C=$A/const0_r2; mkdir -p $C; L=$C/round2.log; log "const0 R2 control (same start, same failures)"
$P -u scripts/alfworld_attr_opd.py choose --traj $T2 --arm const --const_step 0 --ids_from $O/steps_round2.jsonl --max_failures 1000 --out $C/steps_round2.jsonl >> $L 2>&1 \
 && $SR $P -u scripts/alfworld_attr_opd.py teacher --traj $T2 --steps $C/steps_round2.jsonl $TEACH --out $C/targets_round2.jsonl >> $L 2>&1 \
 && distill_loop $T2 $C/targets_round2.jsonl $R1 $C/round2 $L || { log "const0 R2 FAILED"; exit 1; }
$SR $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $C/round2/adapter --split valid_unseen --budget 50 --temperature 0 --out $C/eval_unseen_round2.jsonl --tag round2_const0 >> $L 2>&1
log "const0 R2 verified $(wc -l < $C/round2/examples.jsonl)/$(wc -l < $C/steps_round2.jsonl), unseen $(grep -c '"won": true' $C/eval_unseen_round2.jsonl)/134"
scripts/alfworld_eval_extended.sh $H opd_const0_r2 $C/round2/adapter; log "ALL DONE"
