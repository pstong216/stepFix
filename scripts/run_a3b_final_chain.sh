#!/bin/bash
# User-approved (Oct 6 14:25): FINAL recipe + 3B Model A used BOTH to choose the step and to inform the teacher, 40% corrections + 100 replay.
#   step      = 3B Model A pointer (reports/attributor_no_observations/final, frozen, Who&When Pro)
#   teacher   = hindsight + 3B Model A diagnosis + own analysis
#   filter    = generic one-step environment evidence, top K = 40% of the round's failures; + 100 replay of own wins; one-hot KL
#   R1 starts from the SFT student on the existing R1 trajectories; R2 collects 1500 new games with 3 parallel shards.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; SA=reports/alfworld/self_attr_rounds; CH=selfattr; SUF=_a3bfinal
MA=reports/attributor_no_observations/final; STU=reports/alfworld/student3b_low100/final; A0=reports/alfworld/self_attr/r0/both/adapter
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EVIDENCE=generic STEP=logfix K_FRAC=0.4 REPLAY_N=100 MODELA_PATH=$MA PAR=3
log() { echo "=== $(date '+%F %T') $*" | tee -a $SA/queue.log; }
pick() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge ${1:-12000} ] 2>/dev/null && echo $best; }
S() { local H=""; until H=$(pick $1); [ -n "$H" ]; do sleep 60; done; echo "srun --jobid=$H --overlap -n1 -c4 --mem=0"; }
modelA_pointer() {   # $1=round dir  $2=trajectory file : 3B Model A on the failures -> pointer (step) + diag_modelA (teacher info)
  local D=$1 T=$2
  $P -c "
import json; eps=[json.loads(l) for l in open('$T')][:1000]; open('$D/ids_fail.jsonl','w').writelines(json.dumps({'id':e['id']})+'\n' for e in eps if not e['won'])"
  until grep -q DONE $D/modelA.log 2>/dev/null; do $(S 12000) $P -u scripts/alfworld_attribute.py --attributor $MA --variant no_observations --release ${T%.jsonl}.release.jsonl --out $D/modelA_pred.jsonl >> $D/modelA.log 2>&1 && echo DONE >> $D/modelA.log || sleep 60; done
  $P - <<PY
import json
D="$D"; ids={json.loads(l)["id"] for l in open(f"{D}/ids_fail.jsonl")}; eps={json.loads(l)["id"]:json.loads(l) for l in open("$T")}
pred={r["id"]:r for r in map(json.loads, open(f"{D}/modelA_pred.jsonl"))}; rows=[]
for i in ids:
    if i not in pred or i not in eps: continue
    t=max(0, min(int(pred[i]["t_hat"]), len(eps[i]["steps"])-1)); rows.append({"id":i,"t_hat":t,"type":None})
open(f"{D}/pointer.jsonl","w").writelines(json.dumps({"id":r["id"],"steps":[r["t_hat"]],"mode":None})+"\n" for r in rows)
open(f"{D}/pointer.raw.jsonl","w").writelines(json.dumps(r)+"\n" for r in rows)
open(f"{D}/diag_modelA.raw.jsonl","w").writelines(json.dumps({"id":r["id"],"t_hat":pred[r["id"]]["t_hat"],"type":"Model-A mode "+str(pred[r["id"]]["pred"].get("error_mode"))})+"\n" for r in rows)
ts=[r["t_hat"] for r in rows]; print("3B Model A pointer: %d states, mean t %.2f, t0 %.2f" % (len(rows), sum(ts)/len(ts), sum(t==0 for t in ts)/len(ts)))
PY
}
# ---------- R1 ----------
D=$SA/$CH$SUF/r1; mkdir -p $D/self $D/proxy $D/envlab $D/attr_labels; T1=reports/alfworld/student3b_low100_train_b50.jsonl
[ -s $D/pointer.raw.jsonl ] || modelA_pointer $D $T1 | tee -a $SA/queue.log
[ -f $D/ADAPTER ] || CHAIN_SUFFIX=$SUF SEED=0 bash scripts/self_attr_round.sh $CH 1 $STU $A0 any || { log "a3bfinal R1 FAILED"; exit 1; }
R1A=$(cat $D/ADAPTER); log "a3bfinal R1 done ($R1A)"
# ---------- R2 (parallel collection) ----------
D=$SA/$CH$SUF/r2; mkdir -p $D/self $D/proxy $D/envlab $D/attr_labels; T=$D/traj.jsonl
shard() { local off=$1 out=$D/traj_s$1.jsonl; until [ "$(grep -c . $out 2>/dev/null)" -ge 495 ] && tail -n 3 $D/collect_s$off.log | grep -q "^DONE"; do $(S 10000) $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $R1A --split train --game_offset $off --n_games 500 --budget 50 --temperature 1.0 --out $out --tag ${CH}_r2 --max_episodes_per_run 500 >> $D/collect_s$off.log 2>&1 || sleep 60; done; }
if ! tail -n 3 $D/collect.log 2>/dev/null | grep -q "^DONE"; then
  shard 0 & sleep 45; shard 500 & sleep 45; shard 1000 & wait
  $P - <<PY
import json, sys
sys.path.insert(0,'src'); from error_attr.env.alfworld_env import game_files
D="$D"; order={g:i for i,g in enumerate(game_files('train')[:1500])}; rows={}; rel={}
for s in (0,500,1000):
    for l in open(f"{D}/traj_s{s}.jsonl"):
        try: e=json.loads(l)
        except Exception: continue
        if e["game_file"] in order: rows.setdefault(e["game_file"], e)
    for l in open(f"{D}/traj_s{s}.release.jsonl"):
        try: r=json.loads(l); rel.setdefault(r["id"], r)
        except Exception: pass
eps=sorted(rows.values(), key=lambda e: order[e["game_file"]]); fail=[e["id"] for e in eps if not e["won"]]
open("$T","w").writelines(json.dumps(e)+"\n" for e in eps); open("$T".replace(".jsonl",".release.jsonl"),"w").writelines(json.dumps(rel[i])+"\n" for i in fail if i in rel)
print(f"merged {len(eps)} games, {sum(e['won'] for e in eps)} won")
PY
  echo "DONE merged 3 shards" >> $D/collect.log
fi
[ -s $D/pointer.raw.jsonl ] || modelA_pointer $D $T | tee -a $SA/queue.log
CHAIN_SUFFIX=$SUF SEED=0 bash scripts/self_attr_round.sh $CH 2 $R1A $A0 any || { log "a3bfinal R2 FAILED"; exit 1; }
log "a3bfinal R1->R2 chain DONE"
