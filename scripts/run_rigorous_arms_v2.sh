#!/bin/bash
# Rigorous single-round ablation runner: every arm is trained with SEEDS training seeds from the SFT student and evaluated on the
# large pooled set (valid_unseen 134 + valid_seen 140 + held-out train games 1500-1999 = 774; optionally more held-out games via
# EXTRA_OFFSET/EXTRA_N).  Queue file reports/alfworld/rigorous/queue.txt, lines:  arm_name<TAB>data_spec ("path[:w] path[:w]")
# Output reports/alfworld/rigorous/<arm>/seed<k>/{model,eval_*.jsonl}; one log line per (arm, seed) and per arm (mean over seeds).
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; R=reports/alfworld/rigorous; mkdir -p $R
Q=${RQ:-$R/queue.txt}; M=$R/queue.log; STU=reports/alfworld/student3b_low100/final; SEEDS=${SEEDS:-"1 2 3"}; export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "=== $(date '+%F %T') $*" | tee -a $M reports/alfworld/self_evolve3b/queue.log; }
pick() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge "${1:-10000}" ] 2>/dev/null && echo $best; }
S() { local H=""; until H=$(pick $1); [ -n "$H" ]; do sleep 60; done; echo "srun --jobid=$H --overlap -n1 -c4 --mem=0"; }
ev() { local D=$1 split=$2 off=$3 n=$4 out=$5 need=$6 tag=$7
  until [ "$(grep -c . $out 2>/dev/null)" -ge $need ]; do $(S 10000) $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $D/model/adapter --split $split ${off:+--game_offset $off} ${n:+--n_games $n} --budget 50 --temperature 0 --out $out --tag $tag >> $D/eval.log 2>&1 || sleep 30; done; }
while IFS=$'\t' read -r arm spec; do [ -z "$arm" ] && continue; [ -f $R/$arm/DONE ] && continue; [ -f $R/$arm/CLAIM ] && continue; mkdir -p $R/$arm; touch $R/$arm/CLAIM
  for k in $SEEDS; do D=$R/$arm/seed$k; mkdir -p $D
    until [ -f $D/model/adapter/adapter_config.json ]; do $(S 16000) $P -u scripts/alfworld_opd_plus.py train --data $spec --student Qwen/Qwen2.5-3B-Instruct --adapter $PWD/$STU --lr 1e-5 --epochs 1 --batch 8 --seed $k --out $D/model > $D/train.log 2>&1 || { log "rigorous $arm seed$k: TRAIN FAILED"; sleep 120; }; done
    ev $D valid_unseen "" "" $D/eval_unseen.jsonl 134 rg_${arm}_s$k; ev $D valid_seen "" "" $D/eval_seen.jsonl 140 rg_${arm}_s$k; ev $D train 1500 500 $D/eval_heldout.jsonl 500 rg_${arm}_s$k
    [ -n "${EXTRA_N:-}" ] && ev $D train ${EXTRA_OFFSET:-2000} $EXTRA_N $D/eval_extra.jsonl $EXTRA_N rg_${arm}_s$k
    log "rigorous $arm seed$k: unseen $(grep -c '"won": true' $D/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen.jsonl)/140 heldout $(grep -c '"won": true' $D/eval_heldout.jsonl)/500 | 774 total $(( $(grep -c '"won": true' $D/eval_unseen.jsonl) + $(grep -c '"won": true' $D/eval_seen.jsonl) + $(grep -c '"won": true' $D/eval_heldout.jsonl) ))$([ -n "${EXTRA_N:-}" ] && echo " | extra $(grep -c '"won": true' $D/eval_extra.jsonl)/$EXTRA_N")"
  done
  $P - <<PY | tee -a $M reports/alfworld/self_evolve3b/queue.log
import json, glob, statistics as st
tot=[]
for d in sorted(glob.glob("$R/$arm/seed*")):
    w=sum(sum(1 for l in open(f) if json.loads(l).get("won")) for f in (f"{d}/eval_unseen.jsonl",f"{d}/eval_seen.jsonl",f"{d}/eval_heldout.jsonl")); tot.append(w)
print(f"=== RIGOROUS {'$arm'}: 774-game wins per seed {tot} | mean {st.mean(tot):.1f} sd {st.pstdev(tot):.1f} (student 416)")
PY
  touch $R/$arm/DONE
done < $Q
log "RIGOROUS QUEUE DONE"
