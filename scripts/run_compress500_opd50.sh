#!/bin/bash
# Quick downstream demo: OPD on the same 50 round-1 failures, pointer from the 500-row original vs compressed Model A.
H=$1; V=$2; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; SR="srun --jobid=$H --overlap -n1 -c4 --mem=0"
O=reports/modelA_compress500/opd50_$V; mkdir -p $O; L=$O/run.log; T=reports/alfworld/student3b_low100_train_b50.jsonl; M=reports/modelA_compress500/queue.log
$P -u scripts/alfworld_attr_opd.py choose --traj $T --arm attr --attr reports/modelA_compress500/alf_$V.jsonl --ids_from reports/modelA_compress500/ids50.jsonl --max_failures 1000 --out $O/steps.jsonl >> $L 2>&1 || { echo "=== opd50 $V choose FAILED" >> $M; exit 1; }
$SR $P -u scripts/alfworld_attr_opd.py teacher --traj $T --steps $O/steps.jsonl --teacher Qwen/Qwen2.5-3B-Instruct --teacher_adapter reports/alfworld/student3b_sft/final --out $O/targets.jsonl >> $L 2>&1 || { echo "=== opd50 $V teacher FAILED" >> $M; exit 1; }
$SR $P -u scripts/alfworld_attr_opd.py distill --traj $T --targets $O/targets.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter reports/alfworld/student3b_low100/final --out $O/round1 --verify_k 2 --exclude_a_minus --max_examples_per_run 250 >> $L 2>&1 || { echo "=== opd50 $V distill FAILED" >> $M; exit 1; }
$SR $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $O/round1/adapter --split valid_unseen --budget 50 --temperature 0 --out $O/eval_unseen.jsonl --tag opd50_$V >> $L 2>&1
echo "=== $(date '+%F %T') opd50 $V: verified $(wc -l < $O/round1/examples.jsonl)/50, unseen $(grep -c '"won": true' $O/eval_unseen.jsonl)/$(wc -l < $O/eval_unseen.jsonl)" >> $M
