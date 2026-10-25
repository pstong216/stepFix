#!/bin/bash
# Untrained (zero-shot) teacher + attributor hindsight hint, instead of the 3B SFT teacher; same 319 states and recipe as attr_nofb.
# Uses whichever of the two holds starts first and cancels the other.
HA=$1; HB=$2; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=reports/alfworld/opd/student3b_low100; T=reports/alfworld/student3b_low100_train_b50.jsonl; M=$A/untrained_teacher_queue.log
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
H=""; while [ -z "$H" ]; do
  for j in $HA $HB; do [ "$(squeue -j $j -h -o %T)" = "RUNNING" ] && H=$j && break; done; [ -z "$H" ] && sleep 30; done
for j in $HA $HB; do [ "$j" != "$H" ] && scancel $j; done; log "using hold $H ($(squeue -j $H -h -o %N)), cancelled the other"
SR="srun --jobid=$H --overlap -n1 -c4 --mem=0"
for spec in "t7b_hint_nofb:Qwen/Qwen2.5-7B-Instruct:" "t7b_nohint_nofb:Qwen/Qwen2.5-7B-Instruct:--no_hint"; do
  IFS=: read -r name teacher flag <<< "$spec"; O=$A/$name; L=$O/run.log; mkdir -p $O; cp $A/attr_nofb/steps.jsonl $O/steps.jsonl
  log "$name teacher scoring ($teacher $flag)"
  $SR $P -u scripts/alfworld_attr_opd.py teacher --traj $T --steps $O/steps.jsonl --teacher $teacher $flag --out $O/targets.jsonl >> $L 2>&1 || { log "$name teacher FAILED"; continue; }
  n0=$(grep -c "DONE distill" $L); ok=0
  for c in 1 2 3 4 5 6 7 8; do
    $SR $P -u scripts/alfworld_attr_opd.py distill --traj $T --targets $O/targets.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter reports/alfworld/student3b_low100/final --out $O/round1 --verify_k 2 --exclude_a_minus --max_examples_per_run 250 >> $L 2>&1 || break
    [ "$(grep -c "DONE distill" $L)" -gt "$n0" ] && ok=1 && break
  done
  [ $ok = 1 ] || { log "$name distill FAILED"; continue; }
  log "$name verified $(wc -l < $O/round1/examples.jsonl)/319"
  $SR $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $O/round1/adapter --split valid_unseen --budget 50 --temperature 0 --out $O/eval_unseen_round1.jsonl --tag opd_$name >> $L 2>&1
  log "$name unseen: $(grep -c '"won": true' $O/eval_unseen_round1.jsonl)/$(wc -l < $O/eval_unseen_round1.jsonl)"
  scripts/alfworld_eval_extended.sh $H opd_$name $O/round1/adapter; log "$name done"
done
log "UNTRAINED TEACHER QUEUE FINISHED"
