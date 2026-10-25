#!/bin/bash
# Objective ablation on the SAME 86 rollout-verified states of the self-evolve attr arm: token-level SFT on the winning action
# (hard label) instead of the KL to the teacher's distribution over winning candidates.  Same lr / epochs / eval.
# Runs on the qianwen hold after the combo_top proxy arm, memory-guarded.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; SE=reports/alfworld/self_evolve3b; M=$SE/queue.log
STU=reports/alfworld/student3b_low100/final; export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=2543689; S="srun --jobid=$G --overlap -n1 -c4 --mem=0"; D=$SE/proxy_arms/attr_sft_hard; mkdir -p $D
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
free_mb() { $S nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1; }
guard() { until [ "$(free_mb)" -ge "$1" ] 2>/dev/null; do sleep 60; done; }
$P - <<'PY'
import json
SE="reports/alfworld/self_evolve3b"; n=0
with open(f"{SE}/proxy_arms/attr_sft_hard/train.jsonl","w") as f:
    for l in open(f"{SE}/attr/round1/examples.jsonl"):
        r=json.loads(l)
        if r.get("kind")!="verified": continue
        f.write(json.dumps({"kind":"sft","src":"reports/alfworld/student3b_low100_train_b50.jsonl","id":r["id"],"t":r["t"],"action":r["target_top"]})+"\n"); n+=1
print("=== sft_hard: token-level SFT on the winning action at", n, "verified attr states")
PY
until grep -q "3B proxy attr/combo_top:" $M 2>/dev/null; do sleep 60; done
guard 24000; $S $P -u scripts/alfworld_opd_plus.py train --data $D/train.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter $PWD/$STU --lr 1e-5 --epochs 1 --batch 8 --out $D/model > $D/train.log 2>&1 || { log "sft_hard: TRAIN FAILED"; exit 1; }
guard 10000
for sp in valid_unseen valid_seen; do $S $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $D/model/adapter --split $sp --budget 50 --temperature 0 --out $D/eval_${sp#valid_}.jsonl --tag sft_hard >> $D/train.log 2>&1; done
log "3B attr sft_hard (token SFT on winning action, same 86 verified states): unseen $(grep -c '"won": true' $D/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen.jsonl)/140 (KL version 64/134, 84/140; student 56/134, 80/140)"
