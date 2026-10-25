#!/bin/bash
# Speed up the priority R2 (selfattr_plain_a3b): two extra collectors on train games 500-999 and 1000-1499 (separate files), the driver's
# own collector keeps doing 0-499; when all three ranges are covered, stop the driver, merge (one row per game, release rows too),
# mark collection DONE and restart the driver (it resumes after collection).
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; SA=reports/alfworld/self_attr_rounds
D=$SA/selfattr_plain_a3b/r2; T=$D/traj.jsonl; R1A=reports/alfworld/self_evolve3b/proxy_arms/attr3b_plain_env_top/model/adapter
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pick() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge 10000 ] 2>/dev/null && echo $best; }
shard() { local off=$1 out=$D/traj_s$1.jsonl; until [ "$(grep -c . $out 2>/dev/null)" -ge 495 ] && tail -n 3 $D/collect_s$off.log | grep -q "^DONE"; do H=""; until H=$(pick); [ -n "$H" ]; do sleep 60; done
  srun --jobid=$H --overlap -n1 -c4 --mem=0 $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $R1A --split train --game_offset $off --n_games 500 --budget 50 --temperature 1.0 --out $out --tag selfattr_plain_a3b_r2 --max_episodes_per_run 500 >> $D/collect_s$off.log 2>&1 || sleep 60; done; }
shard 500 & sleep 60; shard 1000 & wait
# wait until the driver's own file covers games 0-499
until $P -c "
import json,sys
sys.path.insert(0,'src'); from error_attr.env.alfworld_env import game_files
g=set(game_files('train')[:500]); have={json.loads(l)['game_file'] for l in open('$T')}
sys.exit(0 if len(g-have)<=5 else 1)" 2>/dev/null; do sleep 60; done
tmux kill-session -t r2_on_3bA 2>/dev/null; sleep 2
for p in $(pgrep -f 'r2_on_3bA_r1.sh'); do kill $p 2>/dev/null; done
for p in $(pgrep -f "out $T "); do kill $p 2>/dev/null; done; sleep 5
$P - <<PY
import json, sys
sys.path.insert(0,'src'); from error_attr.env.alfworld_env import game_files
D="$D"; T="$T"; order={g:i for i,g in enumerate(game_files('train')[:1500])}; rows={}; rel={}
for f in [T, D+"/traj_s500.jsonl", D+"/traj_s1000.jsonl"]:
    for l in open(f):
        try: e=json.loads(l)
        except Exception: continue
        if e["game_file"] in order and e["game_file"] not in rows: rows[e["game_file"]]=e
    rf=f.replace(".jsonl",".release.jsonl")
    try:
        for l in open(rf):
            try: r=json.loads(l); rel.setdefault(r["id"], r)
            except Exception: pass
    except FileNotFoundError: pass
eps=sorted(rows.values(), key=lambda e: order[e["game_file"]])
import shutil; shutil.copy(T, T+".driver_part.bak")
open(T,"w").writelines(json.dumps(e)+"\n" for e in eps)
fail={e["id"] for e in eps if not e["won"]}
open(T.replace(".jsonl",".release.jsonl"),"w").writelines(json.dumps(rel[i])+"\n" for i in [e["id"] for e in eps] if i in fail and i in rel)
print(f"merged {len(eps)} games ({sum(e['won'] for e in eps)} won); release rows {sum(1 for i in fail if i in rel)} of {len(fail)} failures")
PY
echo "DONE ${#} merged sharded collection" >> $D/collect.log
echo "=== $(date '+%F %T') selfattr_plain_a3b r2: sharded collection merged, driver restarted" | tee -a $SA/queue.log
tmux new-session -d -s r2_on_3bA "bash scripts/fix/r2_on_3bA_r1.sh"
