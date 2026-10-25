#!/bin/bash
# v5 = v4 selection (veto + hindsight-teacher ranking + verb cap) with the calibrated update: DPO lr 1e-6, 1 epoch, beta 0.1.
# All 6 step-selection arms + placebo; eval on unseen + seen (274). attr/placebo lr1e-6 unseen already done (reused).
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; O=reports/alfworld/self_evolve7b; M=$O/queue.log
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {
  local a=$1 J=$2; local D=$O/$a; local S="srun --jobid=$J --overlap -n1 -c4 --mem=0"
  [ -f $D/sweep_lr1e-6/adapter/adapter_config.json ] || $S $P -u scripts/alfworld_opd_plus.py train --data $D/dpo_pairs.jsonl --student Qwen/Qwen2.5-7B-Instruct --adapter "" --lr 1e-6 --epochs 1 --beta 0.1 --out $D/sweep_lr1e-6 > $D/train_v5.log 2>&1
  [ "$(wc -l < $D/eval_unseen_sweep_lr1e-6.jsonl 2>/dev/null || echo 0)" -ge 134 ] || $S $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-7B-Instruct --adapter $D/sweep_lr1e-6/adapter --split valid_unseen --budget 50 --temperature 0 --out $D/eval_unseen_sweep_lr1e-6.jsonl --tag v5_$a >> $D/train_v5.log 2>&1
  $S $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-7B-Instruct --adapter $D/sweep_lr1e-6/adapter --split valid_seen --budget 50 --temperature 0 --out $D/eval_seen_v5.jsonl --tag v5_$a >> $D/train_v5.log 2>&1
  echo "=== $(date '+%F %T') v5 $a: $(wc -l < $D/dpo_pairs.jsonl) pairs | unseen $(grep -c '"won": true' $D/eval_unseen_sweep_lr1e-6.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen_v5.jsonl)/$(wc -l < $D/eval_seen_v5.jsonl) (zero-shot 60/134, 50/140)" >> $M
}
echo "=== $(date '+%F %T') v5 start (DPO lr 1e-6)" >> $M
(run placebo 2467981; run entropy 2467981; run hkl 2467981; run selfprog 2467981) &
(run attr 2467982; run rand_early 2467982; run selfattr 2467982) &
wait; echo "=== $(date '+%F %T') v5 ALL DONE" >> $M
