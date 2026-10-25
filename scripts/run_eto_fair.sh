#!/bin/bash
# EXACT ETO replication (Song et al. 2024, arXiv 2403.02502) adapted to our student (Oct 5 22:35): exploration = GREEDY (T=0) rollouts of
# the SFT student on the training instructions that have expert trajectories; pairs = (expert win, student failure) of the same
# instruction, only when rewards differ; DPO beta 0.5 (ALFWorld), lr 1e-6, batch 32, 3 epochs, 1 iteration (ALFWorld), ref = SFT student.
# Deviation (stated): our student is SFT'd on 100 action-only expert demos (ETO: 3321 with GPT-4 CoT), same start as our method.
# Previous note:  The earlier ETO run paired R1 failures with WINS FROM LATER-ROUND CHAINS (env_r2/r3, rollout_r2/r3,
# placebo_r2: 409/409 winners came from stronger, later policies) -> information our R1 never sees.  Fair = original ETO design:
# chosen = expert demonstration of the same game (ALFWorld planner expert, the same kind of data the student was SFT'd on),
# rejected = the student's own R1 failure; the SAME 410 failures our method uses; reference = the R1 student.  3 seeds, 774 eval.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; R=reports/alfworld/rigorous; A=$R/eto_fair; mkdir -p $A; M=$R/queue.log
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ETO_WIN_SRC=reports/alfworld/rigorous/data/expert_games0_999.jsonl
log() { echo "=== $(date '+%F %T') $*" | tee -a $M reports/alfworld/self_evolve3b/queue.log; }
pick() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge ${1:-10000} ] 2>/dev/null && echo $best; }
S() { local H=""; until H=$(pick $1); [ -n "$H" ]; do sleep 60; done; echo "srun --jobid=$H --overlap -n1 -c4 --mem=0"; }
# 1. expert demos for train games 0-999 (CPU, planner expert, budget 50)
until grep -q "DONE expert" $A/expert.log 2>/dev/null; do $P -u scripts/alfworld_expert_collect.py --n_games 1000 --game_offset 0 --budget 50 --expert planner --out $ETO_WIN_SRC >> $A/expert.log 2>&1 || sleep 30; done
log "eto_fair: $(tail -n1 $A/expert.log)"
# 1b. ETO exploration: greedy rollouts of the SFT student on train games 0-999
G=$A/explore_greedy.jsonl
until [ "$(grep -c . $G 2>/dev/null)" -ge 990 ]; do $(S 10000) $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter reports/alfworld/student3b_low100/final --split train --n_games 1000 --budget 50 --temperature 0 --out $G --tag eto_explore --max_episodes_per_run 450 >> $A/explore.log 2>&1 || sleep 60; done
log "eto exact: greedy exploration $(grep -c '"won": true' $G)/$(grep -c . $G) won"
# 2. pairs (ETO exact): expert win vs greedy student failure of the same game (rewards differ)
$P - <<PY | tee -a $M
import json
F="$A/explore_greedy.jsonl"; X="$ETO_WIN_SRC"
ex={}
for l in open(X):
    e=json.loads(l)
    if e.get("won"): ex.setdefault(e["game_file"], e)
eps={json.loads(l)["id"]:json.loads(l) for l in open(F)}; ids=[i for i,e in eps.items() if not e["won"]]; out=[]
for i in ids:
    w=ex.get(eps[i]["game_file"])
    if w: out.append({"kind":"dpo","src":F,"id":i,"t":0,"chosen":w["steps"][0]["action"],"rejected":eps[i]["steps"][0]["action"],"win_id":w["id"]})
open("$A/pairs.jsonl","w").writelines(json.dumps(r)+"\n" for r in out)
print(f"=== eto exact pairs: {len(out)} (greedy student failures {len(ids)} with a successful expert demo of the same game)")
PY
until grep -q "DONE prep" $A/prep.log 2>/dev/null; do $(S 12000) $P -u scripts/traj_dpo.py prep --pairs $A/pairs.jsonl --src $A/explore_greedy.jsonl --max_steps 50 --out $A/refs.jsonl >> $A/prep.log 2>&1 || sleep 60; done
ev() { local D=$1 split=$2 off=$3 n=$4 out=$5 need=$6 tag=$7
  until [ "$(grep -c . $out 2>/dev/null)" -ge $need ]; do $(S 10000) $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $D/model/adapter --split $split ${off:+--game_offset $off} ${n:+--n_games $n} --budget 50 --temperature 0 --out $out --tag $tag >> $D/eval.log 2>&1 || sleep 30; done; }
for k in 1 2 3; do D=$A/seed$k; mkdir -p $D
  until [ -f $D/model/adapter/adapter_config.json ]; do $(S 20000) $P -u scripts/traj_dpo.py train --refs $A/refs.jsonl --src $A/explore_greedy.jsonl --max_steps 50 --beta 0.5 --lr 1e-6 --batch 32 --epochs 3 --seed $k --out $D/model > $D/train.log 2>&1 || { log "eto_fair seed$k: TRAIN FAILED"; sleep 120; }; done
  ev $D valid_unseen "" "" $D/eval_unseen.jsonl 134 rg_etof_s$k; ev $D valid_seen "" "" $D/eval_seen.jsonl 140 rg_etof_s$k; ev $D train 1500 500 $D/eval_heldout.jsonl 500 rg_etof_s$k
  log "rigorous eto_fair seed$k: unseen $(grep -c '"won": true' $D/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen.jsonl)/140 heldout $(grep -c '"won": true' $D/eval_heldout.jsonl)/500 | 774 total $(( $(grep -c '"won": true' $D/eval_unseen.jsonl) + $(grep -c '"won": true' $D/eval_seen.jsonl) + $(grep -c '"won": true' $D/eval_heldout.jsonl) )) | train_s $(grep -o '[0-9]*s$' $D/train.log | tail -n1)"
done
log "RIGOROUS eto_fair DONE"
