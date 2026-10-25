#!/bin/bash
# Same M5 steps, HINDSIGHT candidates (student sees the full failed log), env_top86 filter, no rollout -- vs m5_plain/env_top (167).
# Runs on the next qianwen hold once R3 of the chains is logged there (memory-guarded).
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; SE=reports/alfworld/self_evolve3b; M=$SE/queue.log
STU=reports/alfworld/student3b_low100/final; export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hold() { squeue -u pan00389 -h -o "%i %j %T" | awk -v n="$1" '$2==n && $3=="RUNNING"{print $1}' | sort -n | tail -n1; }
until [ -z "$(squeue -j 2705996 -h -o %T 2>/dev/null | grep RUNNING)" ] && [ -n "$(hold error_attr_hold_igpu_qw)" ]; do sleep 300; done
until grep -q "CHAIN env R3" reports/alfworld/self_evolve3b_rounds/queue.log 2>/dev/null; do sleep 300; done
H=$(hold error_attr_hold_igpu_qw); S="srun --jobid=$H --overlap -n1 -c4 --mem=0"
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
free_mb() { $S nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1; }
guard() { until [ "$(free_mb)" -ge "$1" ] 2>/dev/null; do sleep 60; done; }
D=$SE/proxy_arms/m5_env_top; $P scripts/proxy_arms_se3b.py --arm m5 --rule env_top --k 86 --out $D | tee -a $M
guard 24000; $S $P -u scripts/alfworld_opd_plus.py train --data $D/train.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter $PWD/$STU --lr 1e-5 --epochs 1 --batch 8 --out $D/model > $D/train.log 2>&1 || { log "m5/env_top (hindsight cands): TRAIN FAILED"; exit 1; }
guard 10000; for sp in valid_unseen valid_seen; do $S $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $D/model/adapter --split $sp --budget 50 --temperature 0 --out $D/eval_${sp#valid_}.jsonl --tag px_m5_hind_env >> $D/train.log 2>&1; done
log "3B proxy m5/env_top (M5 steps, HINDSIGHT candidates, env_top86, no rollout): 86 targets | unseen $(grep -c '"won": true' $D/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen.jsonl)/140 (plain candidates: 72/95 = 167; student 56/134, 80/140)"
