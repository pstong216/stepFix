#!/bin/bash
# v3: corrections (SFT) + KL anchor to the untrained model's own distribution (300 own states); lr 2e-5, 1 epoch. Check arms first.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; O=reports/alfworld/self_evolve7b; M=$O/queue.log
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -s $O/anchor_kl300.jsonl ] || srun --jobid=2467981 --overlap -n1 -c4 --mem=0 $P -u scripts/self_evolve_anchor.py > $O/anchor_kl.log 2>&1 || { echo "=== anchor FAILED" >> $M; exit 1; }
run() {
  local a=$1 J=$2; local D=$O/$a; mkdir -p $D; local data="$O/anchor_kl300.jsonl"; [ "$a" != "anchor_only" ] && data="$D/train_sft.jsonl $O/anchor_kl300.jsonl"
  rm -rf $D/model_v3 $D/eval_unseen_v3.jsonl $D/eval_unseen_v3.release.jsonl
  srun --jobid=$J --overlap -n1 -c4 --mem=0 $P -u scripts/alfworld_opd_plus.py train --data $data --student Qwen/Qwen2.5-7B-Instruct --adapter "" --lr 2e-5 --epochs 1 --out $D/model_v3 > $D/train_v3.log 2>&1 \
   && srun --jobid=$J --overlap -n1 -c4 --mem=0 $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-7B-Instruct --adapter $D/model_v3/adapter --split valid_unseen --budget 50 --temperature 0 --out $D/eval_unseen_v3.jsonl --tag se7b3_$a >> $D/train_v3.log 2>&1
  echo "=== $(date '+%F %T') v3 $a: unseen $(grep -c '"won": true' $D/eval_unseen_v3.jsonl)/$(wc -l < $D/eval_unseen_v3.jsonl) (7B zero-shot 60/134)" >> $M
}
echo "=== $(date '+%F %T') v3 start: check arms anchor_only + attr" >> $M
(run anchor_only 2467981) & (run attr 2467982) & wait
echo "=== $(date '+%F %T') v3 check arms done" >> $M
