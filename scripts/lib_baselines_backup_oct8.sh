# Shared helpers for the baseline drivers (user-approved Oct 6 21:35: RFT matched, ETO exact, rollout verification with final steps).
# Same collection / trainer / 774-game evaluation as the final method (selfattr_a3bfinal).
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; SA=reports/alfworld/self_attr_rounds; E=reports/alfworld/ext
STU=reports/alfworld/student3b_low100/final; T1=reports/alfworld/student3b_low100_train_b50.jsonl
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}; GPU_MIN=${GPU_MIN:-10000}; TRAIN_MIN=${TRAIN_MIN:-24000}   # student model (Oct 7: 7B runs set MODEL / GPU_MIN / TRAIN_MIN)
log() { echo "=== $(date '+%F %T') $*" | tee -a $SA/queue.log; }
pick() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge ${1:-12000} ] 2>/dev/null && echo $best; }
S() { local H=""; until H=$(pick $1); [ -n "$H" ]; do sleep 60; done; echo "srun --jobid=$H --overlap -n1 -c4 --mem=0"; }
# collect4 DIR ADAPTER TAG : 1500 train games (T=1.0, budget 50) in 4 shards of 375 -> DIR/traj.jsonl (+ release of the failures)
collect4() { local D=$1 AD=$2 TG=$3 OFFS="0 375 750 1125"
  tail -n 3 $D/collect.log 2>/dev/null | grep -q "^DONE" && return
  for off in $OFFS; do ( until [ "$(grep -c . $D/traj_s$off.jsonl 2>/dev/null)" -ge 371 ] && tail -n 3 $D/collect_s$off.log | grep -q "^DONE"; do $(S $GPU_MIN) $P -u scripts/alfworld_collect.py --model $MODEL --adapter $AD --split train --game_offset $off --n_games 375 --budget 50 --temperature 1.0 --out $D/traj_s$off.jsonl --tag $TG --max_episodes_per_run 375 >> $D/collect_s$off.log 2>&1 || sleep 60; done ) & sleep 60; done; wait
  $P - <<PY
import json, sys
sys.path.insert(0,'src'); from error_attr.env.alfworld_env import game_files
D="$D"; order={g:i for i,g in enumerate(game_files('train')[:1500])}; rows={}; rel={}
for s in [int(x) for x in "$OFFS".split()]:
    for l in open(f"{D}/traj_s{s}.jsonl"):
        try: e=json.loads(l)
        except Exception: continue
        if e["game_file"] in order: rows.setdefault(e["game_file"], e)
    for l in open(f"{D}/traj_s{s}.release.jsonl"):
        try: r=json.loads(l); rel.setdefault(r["id"], r)
        except Exception: pass
eps=sorted(rows.values(), key=lambda e: order[e["game_file"]]); fail=[e["id"] for e in eps if not e["won"]]
open(f"{D}/traj.jsonl","w").writelines(json.dumps(e)+"\n" for e in eps); open(f"{D}/traj.release.jsonl","w").writelines(json.dumps(rel[i])+"\n" for i in fail if i in rel)
print(f"merged {len(eps)} games, {sum(e['won'] for e in eps)} won")
PY
  echo "DONE merged 4 shards" >> $D/collect.log; }
# train1 DIR START_ADAPTER "data1:w data2:w" : same trainer / hyper-parameters as the final method
train1() { local D=$1 AD=$2 DATA=$3
  until [ -f $D/model/adapter/adapter_config.json ]; do $(S $TRAIN_MIN) $P -u scripts/alfworld_opd_plus.py train --data $DATA --student $MODEL --adapter $PWD/$AD --lr 1e-5 --epochs 1 --batch 8 --seed ${SEED:-0} --out $D/model > $D/train.log 2>&1 || { log "$D: TRAIN FAILED (retry in 3 min)"; sleep 180; }; done; }
# eval774 DIR TAG : 274 (unseen 134 + seen 140, in DIR) and the 500 held-out train games 1500-1999 (4 shards -> ext/eval_heldout_TAG.jsonl), greedy, all in parallel
eval774() { local D=$1 TG=$2 AD=$1/model/adapter
  for sp in valid_unseen valid_seen; do n=$([ $sp = valid_unseen ] && echo 134 || echo 140); ( until [ "$(grep -c . $D/eval_${sp#valid_}.jsonl 2>/dev/null)" -ge $n ]; do $(S $GPU_MIN) $P -u scripts/alfworld_collect.py --model $MODEL --adapter $AD --split $sp --budget 50 --temperature 0 --out $D/eval_${sp#valid_}.jsonl --tag $TG >> $D/eval_${sp#valid_}.log 2>&1 || sleep 30; done ) & sleep 60; done
  for i in 0 1 2 3; do off=$((1500 + 125 * i)); o=$E/eval_heldout_${TG}_s$i.jsonl; ( until [ "$(grep -c . $o 2>/dev/null)" -ge 125 ]; do $(S $GPU_MIN) $P -u scripts/alfworld_collect.py --model $MODEL --adapter $AD --split train --game_offset $off --n_games 125 --budget 50 --temperature 0 --out $o --tag h500_$TG >> $E/heldout_${TG}_s$i.log 2>&1 || sleep 30; done ) & sleep 60; done; wait
  cat $E/eval_heldout_${TG}_s{0,1,2,3}.jsonl > $E/eval_heldout_$TG.jsonl.tmp && mv $E/eval_heldout_$TG.jsonl.tmp $E/eval_heldout_$TG.jsonl
  local u=$(grep -c '"won": true' $D/eval_unseen.jsonl) s=$(grep -c '"won": true' $D/eval_seen.jsonl) h=$(grep -c '"won": true' $E/eval_heldout_$TG.jsonl)
  log "BASELINE $TG: unseen $u/134 seen $s/140 heldout $h/500 | 274: $((u + s)) | 774 total: $((u + s + h)) (student 136 / 416)"; }
