#!/bin/bash
# Proxy arms, part 4 (qianwen hold, memory-guarded): split the fut_filter set by its (unused) rollout outcome -- fut_pass (the
# targets that would pass verification) vs fut_fail (the ones that would fail).  Tests whether proxy-approved-but-rollout-failing
# corrections are useful training signal on their own.  Same trainer / lr / eval.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; SE=reports/alfworld/self_evolve3b; M=$SE/queue.log
STU=reports/alfworld/student3b_low100/final; export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=2543689; S="srun --jobid=$G --overlap -n1 -c4 --mem=0"
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
free_mb() { $S nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1; }
guard() { until [ "$(free_mb)" -ge "$1" ] 2>/dev/null; do sleep 60; done; }
arm() { local rule=$1; local D=$SE/proxy_arms/attr_$rule
  $P scripts/proxy_arms_se3b.py --arm attr --rule $rule | tee -a $M
  [ -s $D/train.jsonl ] || { log "proxy $rule: no targets, skipped"; return; }
  guard 24000; $S $P -u scripts/alfworld_opd_plus.py train --data $D/train.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter $PWD/$STU --lr 1e-5 --epochs 1 --batch 8 --out $D/model > $D/train.log 2>&1 || { log "proxy $rule: TRAIN FAILED"; return; }
  guard 10000
  for sp in valid_unseen valid_seen; do $S $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $D/model/adapter --split $sp --budget 50 --temperature 0 --out $D/eval_${sp#valid_}.jsonl --tag px_$rule >> $D/train.log 2>&1; done
  log "3B proxy attr/$rule: $(wc -l < $D/train.jsonl) targets ($(python3 -c "import json;s=json.load(open('$D/stats.json'));print(f\"{s['rollout_won']} would pass rollout\")")) | unseen $(grep -c '"won": true' $D/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen.jsonl)/140 (student 56/134, 80/140)"; }
until grep -q "PROXY ARMS PART 3 DONE" $M 2>/dev/null; do sleep 60; done
log "proxy arms part 4 (qianwen hold): fut_fail (proxy-approved, rollout-failing half), fut_pass"
arm fut_fail; arm fut_pass
log "PROXY ARMS PART 4 DONE"
