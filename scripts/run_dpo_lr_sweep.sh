#!/bin/bash
# DPO update-size sweep (1 epoch) for the attr corrections and the placebo pairs, then the flip probe on all of them.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; O=reports/alfworld/self_evolve7b
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; S="srun --jobid=2467982 --overlap -n1 -c4 --mem=0"
SPECS=""
for lr in 1e-6 3e-6 1e-5; do for a in placebo attr; do
  $S $P -u scripts/alfworld_opd_plus.py train --data $O/$a/dpo_pairs.jsonl --student Qwen/Qwen2.5-7B-Instruct --adapter "" --lr $lr --epochs 1 --beta 0.1 --out $O/$a/sweep_lr$lr >> $O/sweep.log 2>&1
  SPECS="$SPECS $a-lr$lr=$O/$a/sweep_lr$lr/adapter"
done; done
$S $P -u scripts/measure_flip.py $SPECS > $O/sweep_flip.log 2>&1
echo DONE >> $O/sweep_flip.log
