#!/bin/bash
# Replaces alfworld_queue_vl.sh from stage B on (that queue was stopped right after B's teacher stage).
# Order: planner-label controls (round-1 data only) -> B distill/eval (resumes) -> C const1 rerun -> D round-2 offsets.
H=$1; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SR="srun --jobid=$H --overlap -n1 -c4 --mem=0"; A=reports/alfworld/opd/student3b_low100; M=reports/alfworld/queue_vl.log
TEACH="--teacher Qwen/Qwen2.5-3B-Instruct --teacher_adapter reports/alfworld/student3b_sft/final"
REL=reports/alfworld/student3b_low100_round2_attr_nofb_train_b50.release.jsonl
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
distill_loop() {  # $1 traj $2 targets $3 from-adapter $4 out-dir $5 log
  local n0; n0=$(grep -c "DONE distill" $5 2>/dev/null)
  for chunk in 1 2 3 4 5 6 7 8; do
    $SR $P -u scripts/alfworld_attr_opd.py distill --traj $1 --targets $2 --student Qwen/Qwen2.5-3B-Instruct --adapter $3 --out $4 --verify_k 2 --exclude_a_minus --max_examples_per_run 250 >> $5 2>&1 || return 1
    [ "$(grep -c "DONE distill" $5)" -gt "$n0" ] && return 0
  done; return 1; }
eval_unseen() { $SR $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $1 --split valid_unseen --budget 50 --temperature 0 --out $2 --tag $3 >> $4 2>&1
  log "$3 unseen: $(grep -c '"won": true' $2)/$(wc -l < $2)"; }
[ "$(squeue -j $H -h -o %T)" = "RUNNING" ] || { log "queue2: hold $H not running"; exit 1; }

# ---------- CTRL: planner labels, round-1 data only ----------
for spec in 'pl450:2.45' 'pl187:5.9'; do
  name=${spec%%:*}; ep=${spec##*:}; L=reports/alfworld/attributor_$name.log
  log "CTRL $name fine-tune ($(wc -l < data/attributor_sft_$name/train.jsonl) planner-labelled round-1 rows, $ep epochs)"
  if $SR $P -u scripts/finetune_attributor_alf.py --init reports/attributor_7b_no_observations/final --data data/attributor_sft_$name/train.jsonl --out reports/attributor_7b_$name --epochs $ep --lr 3e-5 >> $L 2>&1 \
     && $SR $P -u scripts/alfworld_attribute.py --attributor reports/attributor_7b_$name/final --variant no_observations --release $REL --out reports/alfworld/attr7b${name}_student3b_low100_round2_attr_nofb.jsonl >> $L 2>&1; then
    log "CTRL $name done"
  else log "CTRL $name FAILED"; fi
done

# ---------- B (resume): verify-refreshed pointer branch ----------
O=$A/attrvl_nofb; L=$O/round2.log; T2=reports/alfworld/student3b_low100_round2_attr_nofb_train_b50.jsonl
log "B resume (targets $(wc -l < $O/targets_round2.jsonl)/528)"
if $SR $P -u scripts/alfworld_attr_opd.py teacher --traj $T2 --steps $O/steps_round2.jsonl $TEACH --out $O/targets_round2.jsonl >> $L 2>&1 \
   && distill_loop $T2 $O/targets_round2.jsonl $A/attr_nofb/round1/adapter $O/round2 $L; then
  eval_unseen $O/round2/adapter $O/eval_unseen_round2.jsonl round2_attrvl_nofb $L
  scripts/alfworld_eval_extended.sh $H opd_attrvl_nofb_r2 $O/round2/adapter; log "B done (extended eval written)"
else log "B FAILED"; fi

# ---------- C: const1_nofb with the canonical teacher ----------
log "C start const1_nofb rerun (3B SFT teacher)"
if TAG=student3b_low100 TEACHER=Qwen/Qwen2.5-3B-Instruct TEACHER_ADAPTER=reports/alfworld/student3b_sft/final ROUND=round1 IDS_FROM=$A/attr/steps.jsonl \
   bash scripts/run_alfworld_opd_arm.sh $H const1_nofb 1000 2 && [ -f $A/const1_nofb/round1/adapter/adapter_config.json ]; then
  log "const1_nofb unseen: $(grep -c '"won": true' $A/const1_nofb/eval_unseen_round1.jsonl)/$(wc -l < $A/const1_nofb/eval_unseen_round1.jsonl)"
  scripts/alfworld_eval_extended.sh $H opd_const1_nofb $A/const1_nofb/round1/adapter; log "C done"
else log "C FAILED"; fi

# ---------- D: round-2 oracle-pointer offsets ----------
T2=reports/alfworld/student3b_low100_round2_oracle_nofb_train_b50.jsonl; PROG=reports/alfworld/progress_student3b_low100_round2_oracle_nofb.jsonl
IDS=$A/oracle_nofb/steps_round2.jsonl; FROM=$A/oracle_nofb/round1/adapter
for spec in 'm1:-1' 'p1:1' 'p2:2'; do
  name=${spec%%:*}; off=${spec##*:}; O=$A/oracle_nofb_r2$name; L=$O/run.log; mkdir -p $O
  log "D offset $off start"
  [ -s $O/steps.jsonl ] || $P -u scripts/alfworld_attr_opd.py choose --traj $T2 --arm oracle --progress $PROG --offset $off --ids_from $IDS --max_failures 1000 --out $O/steps.jsonl >> $L 2>&1
  if $SR $P -u scripts/alfworld_attr_opd.py teacher --traj $T2 --steps $O/steps.jsonl $TEACH --out $O/targets.jsonl >> $L 2>&1 \
     && distill_loop $T2 $O/targets.jsonl $FROM $O/round2 $L; then
    eval_unseen $O/round2/adapter $O/eval_unseen_round2.jsonl opd_oracle_nofb_r2$name $L
  else log "D offset $off FAILED"; fi
done
log "QUEUE FINISHED"
