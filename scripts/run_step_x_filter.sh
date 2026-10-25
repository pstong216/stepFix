#!/bin/bash
# Decouple step choice from the action filter: the best filter (env_top86) on plain (no-hindsight) candidates at M5 steps (fully
# self-contained) and at random-early steps.  Hold B, after the Oct-2 replications.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; SE=reports/alfworld/self_evolve3b; M=$SE/queue.log
STU=reports/alfworld/student3b_low100/final; export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S="srun --jobid=2644180 --overlap -n1 -c4 --mem=0"
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
free_mb() { $S nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1; }
guard() { until [ "$(free_mb)" -ge "$1" ] 2>/dev/null; do sleep 60; done; }
arm() { local A=$1 rule=$2; local D=$SE/proxy_arms/${A}_$rule
  $P scripts/proxy_arms_se3b.py --arm $A --rule $rule --k 86 | tee -a $M
  guard 24000; $S $P -u scripts/alfworld_opd_plus.py train --data $D/train.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter $PWD/$STU --lr 1e-5 --epochs 1 --batch 8 --out $D/model > $D/train.log 2>&1 || { log "proxy $A/$rule: TRAIN FAILED"; return; }
  guard 10000
  for sp in valid_unseen valid_seen; do $S $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $D/model/adapter --split $sp --budget 50 --temperature 0 --out $D/eval_${sp#valid_}.jsonl --tag px_${A}_$rule >> $D/train.log 2>&1; done
  log "3B proxy $A/$rule (no hindsight, no rollout): $(wc -l < $D/train.jsonl) targets | unseen $(grep -c '"won": true' $D/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen.jsonl)/140 (student 56/134, 80/140)"; }
until grep -q "OCT2 REPLICATIONS DONE" $M 2>/dev/null && grep -q "DONE onestep" $SE/proxy/onestep4.log 2>/dev/null; do sleep 120; done
log "step x filter arms (hold B): m5_plain/env_top86, rand_early_plain/env_top86"
arm m5_plain env_top; arm rand_early_plain env_top
log "STEP X FILTER ARMS DONE"
