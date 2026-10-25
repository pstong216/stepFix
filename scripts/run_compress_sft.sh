#!/bin/bash
# Trajectory-compression study: SFT the SFT'd 3B student on the same 200 own wins in three forms (raw / pruned causal path /
# path + one detour and its exit), lr 1e-5, 1 epoch (alfworld_opd_plus.py kind sft), then greedy eval on unseen 134 + seen 140.
# Runs on the qianwen hold after its self-evolve arms (m5, entropy) are done.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; O=reports/alfworld/compress_traj; M=$O/queue.log
STU=reports/alfworld/student3b_low100/final; export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=2543689; S="srun --jobid=$G --overlap -n1 -c4 --mem=0"
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
until grep -q "3B self entropy:" reports/alfworld/self_evolve3b/queue.log 2>/dev/null; do sleep 60; done
log "compress-SFT: start (qianwen hold free of self-evolve arms)"
for v in raw pruned keep1; do
  D=$O/$v
  $S $P -u scripts/alfworld_opd_plus.py train --data $D/train.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter $PWD/$STU --lr 1e-5 --epochs 1 --batch 8 --out $D/model > $D/train.log 2>&1 || { log "compress-SFT $v: TRAIN FAILED"; continue; }
  for sp in valid_unseen valid_seen; do $S $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $D/model/adapter --split $sp --budget 50 --temperature 0 --out $D/eval_${sp#valid_}.jsonl --tag cmp_$v >> $D/train.log 2>&1; done
  log "compress-SFT $v: states $(wc -l < $D/train.jsonl) | unseen $(grep -c '"won": true' $D/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen.jsonl)/140 (student 56/134, 80/140)"
done
log "COMPRESS-SFT DONE"
