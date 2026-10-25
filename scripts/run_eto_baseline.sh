#!/bin/bash
# ETO-style baseline (trajectory-level DPO on same-game success/failure pairs), 3 seeds, 774-game eval; rigorous protocol.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; R=reports/alfworld/rigorous; A=$R/eto_traj_dpo; mkdir -p $A; M=$R/queue.log
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "=== $(date '+%F %T') $*" | tee -a $M reports/alfworld/self_evolve3b/queue.log; }
pick() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge "${1:-10000}" ] 2>/dev/null && echo $best; }
S() { local H=""; until H=$(pick $1); [ -n "$H" ]; do sleep 60; done; echo "srun --jobid=$H --overlap -n1 -c4 --mem=0"; }
until grep -q "DONE prep" $A/prep.log 2>/dev/null; do $(S 12000) $P -u scripts/traj_dpo.py prep --pairs reports/alfworld/self_attr_labels/div/dpo_pairs.jsonl --out $A/refs.jsonl >> $A/prep.log 2>&1 || sleep 60; done
ev() { local D=$1 split=$2 off=$3 n=$4 out=$5 need=$6 tag=$7
  until [ "$(grep -c . $out 2>/dev/null)" -ge $need ]; do $(S 10000) $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $D/model/adapter --split $split ${off:+--game_offset $off} ${n:+--n_games $n} --budget 50 --temperature 0 --out $out --tag $tag >> $D/eval.log 2>&1 || sleep 30; done; }
for k in 1 2 3; do D=$A/seed$k; mkdir -p $D
  until [ -f $D/model/adapter/adapter_config.json ]; do $(S 20000) $P -u scripts/traj_dpo.py train --refs $A/refs.jsonl --seed $k --out $D/model > $D/train.log 2>&1 || { log "eto seed$k: TRAIN FAILED"; sleep 120; }; done
  ev $D valid_unseen "" "" $D/eval_unseen.jsonl 134 rg_eto_s$k; ev $D valid_seen "" "" $D/eval_seen.jsonl 140 rg_eto_s$k; ev $D train 1500 500 $D/eval_heldout.jsonl 500 rg_eto_s$k
  log "rigorous eto_traj_dpo seed$k: unseen $(grep -c '"won": true' $D/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen.jsonl)/140 heldout $(grep -c '"won": true' $D/eval_heldout.jsonl)/500 | 774 total $(( $(grep -c '"won": true' $D/eval_unseen.jsonl) + $(grep -c '"won": true' $D/eval_seen.jsonl) + $(grep -c '"won": true' $D/eval_heldout.jsonl) ))"
done
log "RIGOROUS eto_traj_dpo DONE"
