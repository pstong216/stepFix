#!/bin/bash
#SBATCH --job-name=step_select_finish
#SBATCH --account=ee5239
#SBATCH --partition=a100-4,a100-8
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/users/8/pan00389/multi-agent/error-attribution/reports/alfworld/step_select/finish_%j.out
# Finish whatever the step-selection arms did not complete before the interactive holds expired (all steps resume).
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
while [ "$(squeue -j 2358103,2358104 -h -o %T 2>/dev/null | grep -c RUNNING)" -gt 0 ]; do sleep 60; done
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; T=reports/alfworld/student3b_low100_train_b50.jsonl; M=reports/alfworld/step_select/queue.log
until $P scripts/prep_step_select_entropy.py >> $M 2>&1; do
  for s in a b c; do :; done
  $P -u scripts/student_entropy_all_steps.py reports/alfworld/step_select/ids100.jsonl $T reports/alfworld/step_select/entropy_c.jsonl >> reports/alfworld/step_select/entropy_c.log 2>&1
done
for ARM in attr random entropy; do
  O=reports/alfworld/step_select/$ARM; L=$O/run.log
  [ "$(wc -l < $O/eval_unseen.jsonl 2>/dev/null || echo 0)" -ge 134 ] && continue
  [ -s $O/targets.jsonl ] || $P -u scripts/alfworld_attr_opd.py teacher --traj $T --steps $O/steps.jsonl --teacher Qwen/Qwen2.5-3B-Instruct --teacher_adapter reports/alfworld/student3b_sft/final --out $O/targets.jsonl >> $L 2>&1
  until grep -q "DONE distill" $L; do $P -u scripts/alfworld_attr_opd.py distill --traj $T --targets $O/targets.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter reports/alfworld/student3b_low100/final --out $O/round1 --verify_k 2 --exclude_a_minus --max_examples_per_run 250 >> $L 2>&1 || sleep 30; done
  grep -q "$ARM verified" $M || echo "=== $(date '+%F %T') $ARM verified $(wc -l < $O/round1/examples.jsonl)/100" >> $M
  $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $O/round1/adapter --split valid_unseen --budget 50 --temperature 0 --out $O/eval_unseen.jsonl --tag ss_$ARM >> $L 2>&1
  echo "=== $(date '+%F %T') $ARM unseen $(grep -c '"won": true' $O/eval_unseen.jsonl)/$(wc -l < $O/eval_unseen.jsonl) (finished on A100)" >> $M
done
