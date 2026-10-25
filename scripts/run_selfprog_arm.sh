#!/bin/bash
# Finish the self-trained-attributor arm (progress estimator on own wins) after its scoring step was killed.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; O=reports/alfworld/self_evolve7b; M=$O/queue.log
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SE_DIR=$O SE_TRAJ=$O/r1_train.jsonl SE_MODEL=Qwen/Qwen2.5-7B-Instruct SE_ADAPTER= SE_ATTR=$O/attr7b.jsonl
S="srun --jobid=2467982 --overlap -n1 -c4 --mem=0"; SC="srun --jobid=2239109 --overlap -n1 -c2 --mem=24G"
$S $P -u scripts/alfworld_progress_model.py score --traj $O/r1_train.jsonl --student Qwen/Qwen2.5-7B-Instruct --adapter $O/selfprog_model/adapter --ids_from $O/ids60.jsonl --out $O/selfprog_scores.jsonl >> $O/selfprog.log 2>&1 || { echo "=== selfprog score FAILED again" >> $M; exit 1; }
$P scripts/alfworld_progress_model.py pointer --scores $O/selfprog_scores.jsonl --out $O/selfprog_pointer.jsonl --delta 1.0 >> $O/selfprog.log 2>&1
until [ -s $O/attr/targets.jsonl ]; do sleep 30; done   # main pipeline has built its arms
$P scripts/self_evolve.py choose --arms selfprog >> $M 2>&1 && $P scripts/self_evolve.py targets --arms selfprog >> $M 2>&1
until grep -q "DONE onestep" $O/selfprog/onestep.log 2>/dev/null; do $SC $P -u scripts/env_evidence_filter.py --teacher_dir $O/selfprog --traj $O/r1_train.jsonl onestep >> $O/selfprog/onestep.log 2>&1; sleep 3; done
$P scripts/self_evolve.py select --arms selfprog >> $M 2>&1
until grep -q "SELF-EVOLVE 7B ROUND 1 DONE" $M; do sleep 60; done   # do not compete with the main arms for the A40s
$S $P -u scripts/alfworld_opd_plus.py train --data $O/selfprog/train_proxy.jsonl --student Qwen/Qwen2.5-7B-Instruct --adapter "" --lr 1e-4 --epochs 3 --out $O/selfprog/model > $O/selfprog/train.log 2>&1 \
 && $S $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-7B-Instruct --adapter $O/selfprog/model/adapter --split valid_unseen --budget 50 --temperature 0 --out $O/selfprog/eval_unseen.jsonl --tag se7b_selfprog >> $O/selfprog/train.log 2>&1
echo "=== $(date '+%F %T') selfprog: trained on $(wc -l < $O/selfprog/train_proxy.jsonl) proxy-selected corrections, unseen $(grep -c '"won": true' $O/selfprog/eval_unseen.jsonl)/$(wc -l < $O/selfprog/eval_unseen.jsonl) (7B zero-shot 60/134)" >> $M
