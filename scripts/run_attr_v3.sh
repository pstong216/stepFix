#!/bin/bash
# Attributor v3 (AgenTracer-style scale): all counterfactual failures (r0 + scale, unified 5-type vocabulary) + env-evidence labels
# with types; 20% of synthetic GAMES held out; 3 epochs.  Eval: held-out step/+-1/type accuracy vs constant baselines, real-319 distribution.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; L=reports/alfworld/self_attr_labels; A=reports/alfworld/self_attr/v3; mkdir -p $A $L/v3
M=reports/alfworld/self_evolve3b/queue.log; STU=reports/alfworld/student3b_low100/final; export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
pick() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge "${1:-10000}" ] 2>/dev/null && echo $best; }
S() { local H=""; until H=$(pick $1); [ -n "$H" ]; do sleep 60; done; echo "srun --jobid=$H --overlap -n1 -c4 --mem=0"; }
until grep -q "SYNTH SCALE DONE" $M 2>/dev/null; do sleep 300; done
$P - <<PY | tee -a $M
import json, random, glob, collections
L="$L"; rng=random.Random(0)
# every synthetic failure (trajectory + label) from r0 and the scale-up
eps={}; labs=[]
for f in [f"{L}/r0/synth.jsonl"]+sorted(glob.glob(f"{L}/scale/*.jsonl")):
    if f.endswith(".labels.jsonl") or f.endswith(".attempts.jsonl"): continue
    for l in open(f):
        e=json.loads(l); eps[e["id"]]=e; labs.append({"id":e["id"],"t_star":e["inject_t"],"type":e["inject_type"]})
games=sorted({eps[r["id"]]["game_file"] for r in labs}); rng.shuffle(games); held=set(games[:len(games)//5])
tr=[r for r in labs if eps[r["id"]]["game_file"] not in held]; te=[r for r in labs if eps[r["id"]]["game_file"] in held]
open(f"{L}/v3/synth_all.jsonl","w").writelines(json.dumps(eps[r["id"]])+"\n" for r in labs)
open(f"{L}/v3/synth.train.jsonl","w").writelines(json.dumps(r)+"\n" for r in tr); open(f"{L}/v3/synth.heldout.jsonl","w").writelines(json.dumps(r)+"\n" for r in te)
open(f"{L}/v3/heldout_ids.jsonl","w").writelines(json.dumps({"id":r["id"]})+"\n" for r in te)
ts=[r["t_star"] for r in tr]
print(f"=== v3 labels: synthetic {len(tr)} train / {len(te)} held-out (by game) | train t0 {sum(t==0 for t in ts)/len(ts):.2f} t>=5 {sum(t>=5 for t in ts)/len(ts):.2f} | types {dict(collections.Counter(r['type'] for r in tr))} | + env labels 319")
PY
until [ -f $A/adapter/adapter_config.json ]; do $(S 20000) $P -u scripts/alfworld_self_attributor.py train --init $STU --out $A --epochs 3 --lr 1e-4 --labels $L/v3/synth.train.jsonl $L/r0/labels.jsonl --trajs $L/v3/synth_all.jsonl reports/alfworld/student3b_low100_train_b50.jsonl > $A/train.log 2>&1 || { log "attr v3: TRAIN FAILED"; sleep 300; }; done
until [ -s $A/pointer_held.raw.jsonl ]; do $(S 10000) $P -u scripts/alfworld_self_attributor.py pointer --traj $L/v3/synth_all.jsonl --ids $L/v3/heldout_ids.jsonl --adapter $A/adapter --out $A/pointer_held.jsonl > $A/pointer_held.log 2>&1 || sleep 60; done
until [ -s $A/pointer_real319.raw.jsonl ]; do $(S 10000) $P -u scripts/alfworld_self_attributor.py pointer --traj reports/alfworld/student3b_low100_train_b50.jsonl --ids reports/alfworld/self_evolve3b/ids319.jsonl --adapter $A/adapter --out $A/pointer_real319.jsonl > $A/pointer_real319.log 2>&1 || sleep 60; done
$P - <<PY | tee -a $M
import json, statistics as st, collections
A="$A"; L="$L"
U={"wrong_location":"wrong_target","wrong_object":"wrong_target","misplace":"wrong_value","random":"wrong_op","misorder":"misorder","skip":"skip"}
held={json.loads(l)["id"]:json.loads(l) for l in open(f"{L}/v3/synth.heldout.jsonl")}
env={json.loads(l)["id"]:json.loads(l)["t_star"] for l in open(f"{L}/r0/labels.jsonl")}
H=[json.loads(l) for l in open(f"{A}/pointer_held.raw.jsonl")]; R=[json.loads(l) for l in open(f"{A}/pointer_real319.raw.jsonl")]
acc=sum(r["t_hat"]==held[r["id"]]["t_star"] for r in H)/len(H); acc1=sum(abs(r["t_hat"]-held[r["id"]]["t_star"])<=1 for r in H)/len(H)
tacc=sum((r.get("type") or "")==U.get(held[r["id"]]["type"],held[r["id"]]["type"]) for r in H)/len(H)
tt=[held[r["id"]]["t_star"] for r in H]; best_const=max(sum(abs(t-c)<=1 for t in tt) for c in range(0,30))/len(tt)
maj=collections.Counter(U.get(held[r["id"]]["type"]) for r in H).most_common(1)[0][1]/len(H)
ts=[r["t_hat"] for r in R]; ag=sum(abs(r["t_hat"]-env[r["id"]])<=1 for r in R if r["id"] in env)/max(1,sum(r["id"] in env for r in R))
print(f"=== ATTR V3: held-out ({len(H)}) step acc {acc:.2f} (+-1 {acc1:.2f}; best constant +-1 {best_const:.2f}) type acc {tacc:.2f} (majority {maj:.2f}) | real 319: mean t {st.mean(ts):.1f} t0 {sum(t==0 for t in ts)/len(ts):.2f} t>=5 {sum(t>=5 for t in ts)/len(ts):.2f} | +-1 agreement with env labels {ag:.2f} | types {dict(collections.Counter(r.get('type') for r in R).most_common(5))}")
PY
log "ATTR V3 DONE"
