#!/bin/bash
# Which part of the round-1 bundle costs the 5-10 games: two-layer filter alone, or replay alone, on the attr arm (Model A steps,
# hindsight candidates; env_top reference 172/169).  Trained from the SFT student, eval 274.  GPU = most-free hold.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; SE=reports/alfworld/self_evolve3b; M=$SE/queue.log; STU=reports/alfworld/student3b_low100/final
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
pick() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge "${1:-10000}" ] 2>/dev/null && echo $best; }
S() { local H=""; until H=$(pick $1); [ -n "$H" ]; do sleep 60; done; echo "srun --jobid=$H --overlap -n1 -c4 --mem=0"; }
arm() { local name=$1 data=$2 desc=$3; local D=$SE/proxy_arms/$name; mkdir -p $D
  until [ -f $D/model/adapter/adapter_config.json ]; do $(S 16000) $P -u scripts/alfworld_opd_plus.py train --data $data --student Qwen/Qwen2.5-3B-Instruct --adapter $PWD/$STU --lr 1e-5 --epochs 1 --batch 8 --out $D/model > $D/train.log 2>&1 || { log "factor arm $name: TRAIN FAILED"; sleep 120; }; done
  for sp in valid_unseen valid_seen; do until [ "$(grep -c . $D/eval_${sp#valid_}.jsonl 2>/dev/null)" -ge $([ $sp = valid_unseen ] && echo 134 || echo 140) ]; do
    $(S 10000) $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $D/model/adapter --split $sp --budget 50 --temperature 0 --out $D/eval_${sp#valid_}.jsonl --tag px_$name >> $D/train.log 2>&1 || sleep 30; done; done
  log "3B evidence-weight arm $name ($desc): unseen $(grep -c '"won": true' $D/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen.jsonl)/140 (env_top 172/169; student 56/134, 80/140)"; }
arm attr_env_lex "$SE/proxy_arms/attr_env_lex/train.jsonl:1.0" "lexicographic evidence: progress > goal revealed > novelty, no weights; 73/86 same as env_top" &
sleep 90; arm attr_env_ig "$SE/proxy_arms/attr_env_ig/train.jsonl:1.0" "information-gain evidence in bits (log2 K over unvisited receptacles) + progress; 75/86 same as env_top" &
wait; log "WEIGHT-ROBUSTNESS ARMS DONE"
