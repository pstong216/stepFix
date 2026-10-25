#!/bin/bash
# "Pure self" proxy arms: candidates from the student itself WITHOUT hindsight (its own top-2 at the pointer state), the action
# chosen by one-step environment evidence (env_filter), no rollout.  m5_plain uses the self-trained M5 pointer (no external
# component at all); attr_plain uses the Model-A pointer (apples-to-apples with self_nofb, which rollout-verified the same
# candidates).  Runs on the ee5239 hold after the first proxy runner's all_top1 arm, memory-guarded.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; SE=reports/alfworld/self_evolve3b; M=$SE/queue.log
STU=reports/alfworld/student3b_low100/final; export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=2543688; S="srun --jobid=$G --overlap -n1 -c4 --mem=0"
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
free_mb() { $S nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1; }
guard() { until [ "$(free_mb)" -ge "$1" ] 2>/dev/null; do sleep 60; done; }
arm() { local A=$1 rule=$2; local D=$SE/proxy_arms/${A}_$rule
  $P scripts/proxy_arms_se3b.py --arm $A --rule $rule --k 86 | tee -a $M
  [ -s $D/train.jsonl ] || { log "proxy $A/$rule: no targets, skipped"; return; }
  guard 24000; $S $P -u scripts/alfworld_opd_plus.py train --data $D/train.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter $PWD/$STU --lr 1e-5 --epochs 1 --batch 8 --out $D/model > $D/train.log 2>&1 || { log "proxy $A/$rule: TRAIN FAILED"; return; }
  guard 10000
  for sp in valid_unseen valid_seen; do $S $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $D/model/adapter --split $sp --budget 50 --temperature 0 --out $D/eval_${sp#valid_}.jsonl --tag px_${A}_$rule >> $D/train.log 2>&1; done
  log "3B proxy $A/$rule (no hindsight, no rollout): $(wc -l < $D/train.jsonl) targets | unseen $(grep -c '"won": true' $D/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen.jsonl)/140 (student 56/134, 80/140)"; }
true
log "pure-self proxy arms (combo_top86): m5_plain, attr_plain"
arm m5_plain combo_top; arm attr_plain combo_top
log "PURE-SELF ARMS DONE"
