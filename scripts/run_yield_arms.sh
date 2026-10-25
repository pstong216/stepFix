#!/bin/bash
# "Same failures, more usable data": train on the UNCAPPED combo_filter set (137 targets) and on env_top with K=137, after the
# Oct-2 replications free hold A.  Same trainer / eval as every other arm.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; SE=reports/alfworld/self_evolve3b; M=$SE/queue.log
STU=reports/alfworld/student3b_low100/final; export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S="srun --jobid=2644179 --overlap -n1 -c4 --mem=0"
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
free_mb() { $S nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1; }
guard() { until [ "$(free_mb)" -ge "$1" ] 2>/dev/null; do sleep 60; done; }
arm() { local rule=$1 K=$2 name=$3; local D=$SE/proxy_arms/attr_$name
  $P scripts/proxy_arms_se3b.py --arm attr --rule $rule --k $K --out $D | tee -a $M
  guard 24000; $S $P -u scripts/alfworld_opd_plus.py train --data $D/train.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter $PWD/$STU --lr 1e-5 --epochs 1 --batch 8 --out $D/model > $D/train.log 2>&1 || { log "proxy $name: TRAIN FAILED"; return; }
  guard 10000
  for sp in valid_unseen valid_seen; do $S $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $D/model/adapter --split $sp --budget 50 --temperature 0 --out $D/eval_${sp#valid_}.jsonl --tag px_$name >> $D/train.log 2>&1; done
  log "3B proxy attr/$name: $(wc -l < $D/train.jsonl) targets ($(python3 -c "import json;s=json.load(open('$D/stats.json'));print(f\"{s['rollout_won']} would pass rollout\")")) | unseen $(grep -c '"won": true' $D/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen.jsonl)/140 (student 56/134, 80/140)"; }
until grep -q "OCT2 REPLICATIONS DONE" $M 2>/dev/null; do sleep 120; done
log "yield arms (uncapped combo_filter 137, env_top K=137)"
arm combo_filter 0 combo_filter137; arm env_top 137 env_top137
log "YIELD ARMS DONE"
