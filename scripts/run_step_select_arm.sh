#!/bin/bash
# Same 100 failures; the supervised step is chosen by ARM (attr | random | entropy); 3B SFT teacher; verified-only; eval unseen.
H=$1; ARM=$2; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; SR="srun --jobid=$H --overlap -n1 -c4 --mem=0"
O=reports/alfworld/step_select/$ARM; L=$O/run.log; T=reports/alfworld/student3b_low100_train_b50.jsonl; M=reports/alfworld/step_select/queue.log
[ -s $O/targets.jsonl ] || $SR $P -u scripts/alfworld_attr_opd.py teacher --traj $T --steps $O/steps.jsonl --teacher Qwen/Qwen2.5-3B-Instruct --teacher_adapter reports/alfworld/student3b_sft/final --out $O/targets.jsonl >> $L 2>&1
until grep -q "DONE distill" $L; do
  $SR $P -u scripts/alfworld_attr_opd.py distill --traj $T --targets $O/targets.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter reports/alfworld/student3b_low100/final --out $O/round1 --verify_k 2 --exclude_a_minus --max_examples_per_run 250 >> $L 2>&1 || sleep 60
done
echo "=== $(date '+%F %T') $ARM verified $(wc -l < $O/round1/examples.jsonl)/100" >> $M
$SR $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $O/round1/adapter --split valid_unseen --budget 50 --temperature 0 --out $O/eval_unseen.jsonl --tag ss_$ARM >> $L 2>&1
echo "=== $(date '+%F %T') $ARM unseen $(grep -c '"won": true' $O/eval_unseen.jsonl)/$(wc -l < $O/eval_unseen.jsonl)" >> $M
