#!/bin/bash
# Proxy-selected self-correction arms (no rollout verification) on the Model-A pointer states of the 3B self-evolve run.
# Each arm: build one-hot KL targets by a proxy rule -> train from the SFT student (lr 1e-5, 1 epoch, same trainer as the
# placebo) -> greedy eval on unseen 134 + seen 140.  Runs on the ee5239 hold next to the running arms; a memory guard waits for
# enough free GPU memory before every train (24 GB) and eval (10 GB) so nothing OOMs.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; SE=reports/alfworld/self_evolve3b; M=$SE/queue.log
STU=reports/alfworld/student3b_low100/final; export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=2543688; S="srun --jobid=$G --overlap -n1 -c4 --mem=0"; K=86
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
free_mb() { $S nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1; }
guard() { until [ "$(free_mb)" -ge "$1" ] 2>/dev/null; do sleep 60; done; }
arm() { local rule=$1; local D=$SE/proxy_arms/attr_$rule
  $P scripts/proxy_arms_se3b.py --arm attr --rule $rule --k $K | tee -a $M
  [ -s $D/train.jsonl ] || { log "proxy $rule: no targets, skipped"; return; }
  guard 24000; $S $P -u scripts/alfworld_opd_plus.py train --data $D/train.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter $PWD/$STU --lr 1e-5 --epochs 1 --batch 8 --out $D/model > $D/train.log 2>&1 || { log "proxy $rule: TRAIN FAILED"; return; }
  guard 10000
  for sp in valid_unseen valid_seen; do $S $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $D/model/adapter --split $sp --budget 50 --temperature 0 --out $D/eval_${sp#valid_}.jsonl --tag px_$rule >> $D/train.log 2>&1; done
  log "3B proxy attr/$rule: $(wc -l < $D/train.jsonl) targets ($(python3 -c "import json;s=json.load(open('$D/stats.json'));print(f\"{s['rollout_won']} would pass rollout\")")) | unseen $(grep -c '"won": true' $D/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen.jsonl)/140 (student 56/134, 80/140)"; }
# env proxy needs the one-step executions of the attr candidates
until [ -f $SE/proxy/onestep.jsonl ] && $P - <<'PY'
import json, sys
SE="reports/alfworld/self_evolve3b"
need={(json.loads(l)["id"],json.loads(l)["steps"][0]) for l in open(f"{SE}/attr/steps.jsonl")}
have={(json.loads(l)["id"],json.loads(l)["t"]) for l in open(f"{SE}/proxy/onestep.jsonl")}
sys.exit(0 if need <= have else 1)
PY
do sleep 60; done
log "proxy arms (attr pointers, no rollout): env_filter, hind_top$K, all_top1, then det_top$K when the detour detector is scored"
arm env_filter; arm hind_top; arm all_top1
# detector proxy: wait for the detour detector's adapter, score every candidate, then build and train
until [ -f reports/alfworld/detour_detector/adapter/adapter_config.json ] && grep -q "=== detour pointer" $M; do sleep 60; done
guard 12000; $S $P -u scripts/alfworld_detour_detector.py score_cands --adapter reports/alfworld/detour_detector/adapter --cands $SE/proxy/cands.jsonl --out $SE/proxy/detector_cands.jsonl > $SE/proxy/score_cands.log 2>&1
arm det_top; arm env_top
log "PROXY ARMS DONE"
