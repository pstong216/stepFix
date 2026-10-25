#!/bin/bash
# One queued ALFWorld task on GPU hold $2, guarded by a lock so the igpu sequential queue and the returning-hold dispatcher never run it twice.
# Usage: alfworld_task.sh <attralf_r2_ext|oracle_r2_fin|const1|r3_fin|seed1|cv> <hold>
T=$1; H=$2; cd /users/8/pan00389/multi-agent/error-attribution; A=reports/alfworld/opd/student3b_low100; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; SR="srun --jobid=$H --overlap -n1 -c4 --mem=0"
mkdir reports/alfworld/locks/$T 2>/dev/null || { echo "task $T already taken"; exit 0; }
echo "=== $(date) task $T START on hold $H ($(squeue -j $H -h -o %N))" | tee -a reports/alfworld/dispatch.log
run_distill_loop() {  # $1 traj $2 targets $3 from-adapter $4 out-dir $5 log, then extra distill args
  local T2=$1 TG=$2 FROM=$3 OUT=$4 L=$5; shift 5; local n0; n0=$(grep -c "DONE distill" $L)
  for chunk in 1 2 3 4 5; do
    $SR $P -u scripts/alfworld_attr_opd.py distill --traj $T2 --targets $TG --student Qwen/Qwen2.5-3B-Instruct --adapter $FROM --out $OUT --verify_k 2 --exclude_a_minus --max_examples_per_run 250 "$@" >> $L 2>&1 || return 1
    [ "$(grep -c "DONE distill" $L)" -gt "$n0" ] && return 0
  done; return 1; }
eval_unseen() { $SR $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $1 --split valid_unseen --budget 50 --temperature 0 --out $2 --tag $3 >> $4 2>&1; echo "=== $(date) $3: $(grep -c '"won": true' $2)/$(wc -l < $2) unseen" | tee -a $4 reports/alfworld/dispatch.log; }
case $T in
  attralf_r2_ext)
    until grep -q "ROUND_DONE round2 attralf_nofb" $A/attralf_nofb/round2.log; do sleep 60; done
    scripts/alfworld_eval_extended.sh $H opd_attralf_nofb_r2 $A/attralf_nofb/round2/adapter;;
  oracle_r2_fin)
    O=$A/oracle_nofb; T2=reports/alfworld/student3b_low100_round2_oracle_nofb_train_b50.jsonl; L=$O/round2.log
    echo "=== $(date) round2 oracle_nofb RESUMED at teacher stage on hold $H" >> $L
    $SR $P -u scripts/alfworld_attr_opd.py teacher --traj $T2 --steps $O/steps_round2.jsonl --teacher Qwen/Qwen2.5-3B-Instruct --teacher_adapter reports/alfworld/student3b_sft/final --out $O/targets_round2.jsonl >> $L 2>&1 || exit 1
    run_distill_loop $T2 $O/targets_round2.jsonl $O/round1/adapter $O/round2 $L || exit 1
    eval_unseen $O/round2/adapter $O/eval_unseen_round2.jsonl round2_oracle_nofb $L; echo "=== $(date) ROUND_DONE round2 oracle_nofb" | tee -a $L
    scripts/alfworld_eval_extended.sh $H opd_oracle_nofb_r2 $O/round2/adapter;;
  const1)
    export TAG=student3b_low100 TEACHER=Qwen/Qwen2.5-3B-Instruct TEACHER_ADAPTER=reports/alfworld/student3b_sft/final ROUND=round1 IDS_FROM=$A/attr/steps.jsonl
    scripts/run_alfworld_opd_arm.sh $H const1_nofb 1000 2; echo "=== const1_nofb finished" >> reports/alfworld/opd_round1.log
    scripts/alfworld_eval_extended.sh $H opd_const1_nofb $A/const1_nofb/round1/adapter;;
  r3_fin)
    O=$A/attr_nofb; T2=reports/alfworld/student3b_low100_round3_attr_nofb_train_b50.jsonl; L=$O/round3.log
    echo "=== $(date) round3 attr_nofb RESUMED at distill stage on hold $H" >> $L
    run_distill_loop $T2 $O/targets_round3.jsonl $O/round2/adapter $O/round3 $L || exit 1
    eval_unseen $O/round3/adapter $O/eval_unseen_round3.jsonl round3_attr_nofb $L; echo "=== $(date) ROUND_DONE round3 attr_nofb" | tee -a $L
    scripts/alfworld_eval_extended.sh $H opd_attr_nofb_r3 $O/round3/adapter;;
  seed1)
    O=$A/attr_nofb; L=$O/round1_seed1.log; echo "=== $(date) attr_nofb round1 seed replicate (seed 1) on hold $H" >> $L
    run_distill_loop reports/alfworld/student3b_low100_train_b50.jsonl $O/targets.jsonl reports/alfworld/student3b_low100/final $O/round1_seed1 $L --seed 1 || exit 1
    eval_unseen $O/round1_seed1/adapter $O/eval_unseen_round1_seed1.jsonl opd_attr_nofb_seed1 $L
    scripts/alfworld_eval_extended.sh $H opd_attr_nofb_seed1 $O/round1_seed1/adapter;;
  cv)
    echo "=== $(date) counterfactual attralf_nofb round1 (resume) on hold $H" >> reports/alfworld/counterfactual.log
    $SR $P -u scripts/alfworld_verify_counterfactual.py --arm attralf_nofb --round round1 >> reports/alfworld/counterfactual.log 2>&1
    O=$A/attr_nofb; L=$O/round1_cv.log; echo "=== $(date) attr_nofb round1 CONTRASTIVE verification on hold $H" >> $L
    run_distill_loop reports/alfworld/student3b_low100_train_b50.jsonl $O/targets.jsonl reports/alfworld/student3b_low100/final $O/round1_cv $L --contrastive --seed 0 || exit 1
    eval_unseen $O/round1_cv/adapter $O/eval_unseen_round1_cv.jsonl opd_attr_nofb_cv $L
    scripts/alfworld_eval_extended.sh $H opd_attr_nofb_cv $O/round1_cv/adapter;;
  tstar_nofb|tstar_nofb_p1|tstar_nofb_p2)   # t* (point-of-no-return oracle) exact/+1/+2, verified-only; only 233/319 failures have a t_star label (86 timed out at 900s, retrying with 1800s in the background) -- restrict to those 233 so every failure used actually has a real t_star, not a silent skip
    export TAG=student3b_low100 TEACHER=Qwen/Qwen2.5-3B-Instruct TEACHER_ADAPTER=reports/alfworld/student3b_sft/final ROUND=round1 IDS_FROM=reports/alfworld/tstar_ready_ids.jsonl
    scripts/run_alfworld_opd_arm.sh $H $T 1000 2 || exit 1; [ -f $A/$T/round1/adapter/adapter_config.json ] || { echo "no adapter produced for $T, aborting"; exit 1; }; echo "=== $T finished" >> reports/alfworld/opd_round1.log
    scripts/alfworld_eval_extended.sh $H opd_$T $A/$T/round1/adapter;;
  attr_nofb_p1|attr_nofb_p2)   # Model A's own predicted step +1/+2 (Model A itself, not the planner oracle); restricted to the SAME canonical 319 shared failures as every other round-1 arm (progress_${TAG}_train_b50.jsonl has since grown to 454 valid entries from later, unrelated CPU labelling jobs -- must not use it directly here)
    export TAG=student3b_low100 TEACHER=Qwen/Qwen2.5-3B-Instruct TEACHER_ADAPTER=reports/alfworld/student3b_sft/final ROUND=round1 IDS_FROM=$A/attr_nofb/steps.jsonl
    scripts/run_alfworld_opd_arm.sh $H $T 1000 2 || exit 1; [ -f $A/$T/round1/adapter/adapter_config.json ] || { echo "no adapter produced for $T, aborting"; exit 1; }; echo "=== $T finished" >> reports/alfworld/opd_round1.log
    scripts/alfworld_eval_extended.sh $H opd_$T $A/$T/round1/adapter;;
  off_p1|off_p2|off_m2|off_p5|off_m5)   # imprecision controls: planner-oracle step shifted by +1 / +2 / -2 / +5 / -5 actions, verified-only, same 319 failures, same teacher
    ARM=oracle_nofb_${T#off_}; export TAG=student3b_low100 TEACHER=Qwen/Qwen2.5-3B-Instruct TEACHER_ADAPTER=reports/alfworld/student3b_sft/final ROUND=round1 IDS_FROM=$A/attr/steps.jsonl
    scripts/run_alfworld_opd_arm.sh $H $ARM 1000 2; echo "=== $ARM finished" >> reports/alfworld/opd_round1.log
    scripts/alfworld_eval_extended.sh $H opd_$ARM $A/$ARM/round1/adapter;;
  *) echo "unknown task $T"; exit 1;;
esac
echo "=== $(date) task $T FINISHED on hold $H" | tee -a reports/alfworld/dispatch.log
