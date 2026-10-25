#!/bin/bash
# 7B round 1, training recipe v2: SFT on the 30 proxy-selected corrections + 150 anchor states from the model's OWN wins,
# lr 2e-5, 2 epochs, fresh LoRA on the untrained 7B; plus an anchor-only control. Two arms at a time (one per A40).
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; O=reports/alfworld/self_evolve7b; M=$O/queue.log
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {  # arm job
  local a=$1 J=$2; local D=$O/$a; local data="$O/anchor150.jsonl"; [ "$a" != "anchor_only" ] && data="$D/train_sft.jsonl $O/anchor150.jsonl"
  rm -rf $D/model_v2 $D/eval_unseen_v2.jsonl $D/eval_unseen_v2.release.jsonl
  srun --jobid=$J --overlap -n1 -c4 --mem=0 $P -u scripts/alfworld_opd_plus.py train --data $data --student Qwen/Qwen2.5-7B-Instruct --adapter "" --lr 2e-5 --epochs 2 --out $D/model_v2 > $D/train_v2.log 2>&1 \
   && srun --jobid=$J --overlap -n1 -c4 --mem=0 $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-7B-Instruct --adapter $D/model_v2/adapter --split valid_unseen --budget 50 --temperature 0 --out $D/eval_unseen_v2.jsonl --tag se7b2_$a >> $D/train_v2.log 2>&1
  echo "=== $(date '+%F %T') v2 $a: unseen $(grep -c '"won": true' $D/eval_unseen_v2.jsonl)/$(wc -l < $D/eval_unseen_v2.jsonl) (7B zero-shot 60/134)" >> $M
}
echo "=== $(date '+%F %T') v2 training start (corrections + own-win anchor, lr 2e-5, 2 epochs)" >> $M
(run anchor_only 2467981; run attr 2467981; run hkl 2467981; run selfprog 2467981) &
(run rand_early 2467982; run entropy 2467982; run selfattr 2467982) &
wait; echo "=== $(date '+%F %T') v2 ALL DONE" >> $M
