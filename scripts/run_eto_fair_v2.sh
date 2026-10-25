#!/bin/bash
# User-approved (Oct 6 21:35) baseline B2, EXACT ETO (Song et al. 2024) on our SFT student, 1 iteration (as in the paper for ALFWorld), seed 1:
#   exploration = GREEDY (T=0) rollouts of the SFT student on train games 0-999; pairs = (expert win, student failure) of the same game;
#   trajectory DPO beta 0.5, lr 1e-6, batch 32, 3 epochs, reference = SFT student.  Same 774 evaluation as the final method.
# Change vs run_eto_fair.sh (paused): exploration first, then planner-expert demos ONLY for the games the student failed (the only ones a
# pair can use) in 12 CPU shards -- the planner expert takes ~70 s/game.  The pairs are identical to doing all 1000 expert games first.
source scripts/lib_baselines.sh; A=reports/alfworld/rigorous/eto_fair; mkdir -p $A; X0=reports/alfworld/rigorous/data/expert_games0_999.jsonl
export ETO_WIN_SRC=$A/expert_used.jsonl
# 1. greedy exploration, 4 GPU shards of 250
n_ex=$(grep -c . $A/explore_greedy.jsonl 2>/dev/null); if [ "${n_ex:-0}" -lt 990 ]; then
  for off in 0 250 500 750; do ( until [ "$(grep -c . $A/explore_s$off.jsonl 2>/dev/null)" -ge 248 ]; do $(S 10000) $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $STU --split train --game_offset $off --n_games 250 --budget 50 --temperature 0 --out $A/explore_s$off.jsonl --tag rg_eto_explore >> $A/explore_s$off.log 2>&1 || sleep 60; done ) & sleep 60; done; wait
  cat $A/explore_s{0,250,500,750}.jsonl > $A/explore_greedy.jsonl
fi
log "eto exact: greedy exploration $(grep -c '"won": true' $A/explore_greedy.jsonl)/$(grep -c . $A/explore_greedy.jsonl) won"
# 2. planner-expert demos for the failed games (12 CPU shards, one game per call; resumable)
$P - <<PY | grep -E "^[0-9]+$" > $A/expert_todo.txt
import json, sys
sys.path.insert(0,'src'); from error_attr.env.alfworld_env import game_files
idx={g:i for i,g in enumerate(game_files('train')[:1000])}; done=set()
import os
for f in ["$X0"]+[f"$A/expert_s{j}.jsonl" for j in range(12)]:
    if os.path.exists(f):
        for l in open(f):
            try: done.add(json.loads(l)["game_file"])
            except Exception: pass
fails=[json.loads(l) for l in open("$A/explore_greedy.jsonl")]
for e in fails:
    if not e["won"] and e["game_file"] in idx and e["game_file"] not in done: print(idx[e["game_file"]])
PY
log "eto exact: $(grep -c . $A/expert_todo.txt) failed games still need an expert demo"
for j in $(seq 0 11); do ( awk -v j=$j 'NR % 12 == j' $A/expert_todo.txt | while read -r g; do $P -u scripts/alfworld_expert_collect.py --n_games 1 --game_offset $g --budget 50 --expert planner --out $A/expert_s$j.jsonl < /dev/null >> $A/expert_s$j.log 2>&1; done ) & done; wait
cat $X0 $A/expert_s*.jsonl > $ETO_WIN_SRC
# 3. pairs: expert win vs greedy student failure of the same game
$P - <<PY | tee -a $SA/queue.log
import json
ex={}
for l in open("$ETO_WIN_SRC"):
    e=json.loads(l)
    if e.get("won"): ex.setdefault(e["game_file"], e)
eps=[json.loads(l) for l in open("$A/explore_greedy.jsonl")]; out=[]
for e in eps:
    if e["won"]: continue
    w=ex.get(e["game_file"])
    if w: out.append({"kind":"dpo","src":"$A/explore_greedy.jsonl","id":e["id"],"t":0,"chosen":w["steps"][0]["action"],"rejected":e["steps"][0]["action"],"win_id":w["id"]})
open("$A/pairs.jsonl","w").writelines(json.dumps(r)+"\n" for r in out)
print(f"=== eto exact pairs: {len(out)} (greedy student failures {sum(not e['won'] for e in eps)} with a successful expert demo of the same game)")
PY
# 4. reference log-probs (SFT student), 5. DPO, 6. 774 evaluation
until grep -q "DONE prep" $A/prep.log 2>/dev/null; do $(S 12000) $P -u scripts/traj_dpo.py prep --pairs $A/pairs.jsonl --src $A/explore_greedy.jsonl --max_steps 50 --out $A/refs.jsonl >> $A/prep.log 2>&1 || sleep 60; done
D=$A/seed1; mkdir -p $D
until [ -f $D/model/adapter/adapter_config.json ]; do $(S 20000) $P -u scripts/traj_dpo.py train --refs $A/refs.jsonl --src $A/explore_greedy.jsonl --max_steps 50 --beta 0.5 --lr 1e-6 --batch 32 --epochs 3 --seed 1 --out $D/model > $D/train.log 2>&1 || { log "eto exact seed1: TRAIN FAILED (retry)"; sleep 180; }; done
[ -s $E/eval_heldout_eto_fair_s1.jsonl ] || eval774 $D eto_fair_s1
log "eto exact seed1 DONE"
