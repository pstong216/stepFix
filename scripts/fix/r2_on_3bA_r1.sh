#!/bin/bash
# User (Oct 6 12:00): R2 built on the previous 3B-Model-A R1 (self_evolve3b/proxy_arms/attr3b_plain_env_top: 3B attributor steps,
# plain student candidates, env_top 86, no replay -> 168/274).  R2 replicates that recipe on the R1 model's own new trajectories:
# collect 1500 train games (T=1) -> 3B Model A pointer on the failures -> plain candidates at those steps -> env top-86 -> train -> eval.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; SA=reports/alfworld/self_attr_rounds
R1A=reports/alfworld/self_evolve3b/proxy_arms/attr3b_plain_env_top/model/adapter; D=$SA/selfattr_plain_a3b/r2; mkdir -p $D/self $D/proxy $D/envlab $D/attr_labels; T=$D/traj.jsonl
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pick() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge ${1:-12000} ] 2>/dev/null && echo $best; }
S() { local H=""; until H=$(pick $1); [ -n "$H" ]; do sleep 60; done; echo "srun --jobid=$H --overlap -n1 -c4 --mem=0"; }
until tail -n 3 $D/collect.log 2>/dev/null | grep -q "^DONE"; do $(S 10000) $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $R1A --split train --n_games 1500 --budget 50 --temperature 1.0 --out $T --tag selfattr_plain_a3b_r2 --max_episodes_per_run 450 >> $D/collect.log 2>&1 || sleep 60; done
$P -c "
import json; eps=[json.loads(l) for l in open('$T')][:1000]; open('$D/ids_fail.jsonl','w').writelines(json.dumps({'id':e['id']})+'\n' for e in eps if not e['won'])"
until grep -q DONE $D/modelA.log 2>/dev/null; do $(S 12000) $P -u scripts/alfworld_attribute.py --attributor reports/attributor_no_observations/final --variant no_observations --release ${T%.jsonl}.release.jsonl --out $D/modelA_pred.jsonl >> $D/modelA.log 2>&1 && echo DONE >> $D/modelA.log || sleep 60; done
$P - <<PY
import json
D="$D"; ids={json.loads(l)["id"] for l in open(f"{D}/ids_fail.jsonl")}; eps={json.loads(l)["id"]:json.loads(l) for l in open("$T")}
rows=[]
for r in map(json.loads, open(f"{D}/modelA_pred.jsonl")):
    if r["id"] not in ids: continue
    t=max(0, min(int(r["t_hat"]), len(eps[r["id"]]["steps"])-1)); rows.append({"id": r["id"], "t_hat": t, "type": None})
open(f"{D}/pointer.jsonl","w").writelines(json.dumps({"id":r["id"],"steps":[r["t_hat"]],"mode":None})+"\n" for r in rows)
open(f"{D}/pointer.raw.jsonl","w").writelines(json.dumps(r)+"\n" for r in rows)
ts=[r["t_hat"] for r in rows]; print("3B Model A pointer:", len(rows), "states, mean t %.2f t0 %.2f" % (sum(ts)/len(ts), sum(t==0 for t in ts)/len(ts)))
PY
K=86 CHAIN_SUFFIX=_a3b SEED=0 bash scripts/self_attr_round.sh selfattr_plain 2 $R1A reports/alfworld/self_attr/r0/both/adapter any
echo "=== $(date '+%F %T') R2 on the previous 3B-Model-A R1 DONE (selfattr_plain_a3b)" | tee -a $SA/queue.log
