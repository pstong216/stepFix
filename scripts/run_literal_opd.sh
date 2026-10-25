#!/bin/bash
# Literal on-policy distillation against the hindsight self-teacher: KL to the teacher's FULL soft distribution over admissible
# commands at the pointer step, no filter, no verification (319 states; and the 86 rollout-verified states).  Hold A, guarded.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; SE=reports/alfworld/self_evolve3b; M=$SE/queue.log
STU=reports/alfworld/student3b_low100/final; export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S="srun --jobid=2644179 --overlap -n1 -c4 --mem=0"
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
free_mb() { $S nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1; }
guard() { until [ "$(free_mb)" -ge "$1" ] 2>/dev/null; do sleep 60; done; }
until grep -q "YIELD ARMS DONE" $M 2>/dev/null; do sleep 120; done
for name in hind_soft86 hind_soft319; do D=$SE/proxy_arms/attr_$name
  guard 24000; $S $P -u scripts/alfworld_opd_plus.py train --data $D/train.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter $PWD/$STU --lr 1e-5 --epochs 1 --batch 8 --out $D/model > $D/train.log 2>&1 || { log "literal OPD $name: TRAIN FAILED"; continue; }
  guard 10000; for sp in valid_unseen valid_seen; do $S $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $D/model/adapter --split $sp --budget 50 --temperature 0 --out $D/eval_${sp#valid_}.jsonl --tag px_$name >> $D/train.log 2>&1; done
  log "3B literal OPD attr/$name (full hindsight distribution, no filter): $(wc -l < $D/train.jsonl) states | unseen $(grep -c '"won": true' $D/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen.jsonl)/140 (student 56/134, 80/140)"
done
log "LITERAL OPD DONE"
