#!/bin/bash
# Round-0 self-trained attributors (3B, init = student LoRA) from two self-generated label sources, and their evaluation:
#   synth  : Who&When-style injected failures from the student's own wins (label = injected step + type)
#   envlab : env-evidence labels on the 319 real failures (first step with a clearly better alternative)
#   both   : union
# Eval: (a) synth held-out by game_file (step acc, +-1 acc, type acc); (b) pointer on the 319 real failures vs env labels, M5, Model A;
#       (c) pointer position distribution.  GPU = qianwen hold (by name), memory-guarded.  Resumable per stage.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
L=reports/alfworld/self_attr_labels/r0; A=reports/alfworld/self_attr/r0; mkdir -p $A; M=reports/alfworld/self_evolve3b/queue.log
TRAJ=reports/alfworld/student3b_low100_train_b50.jsonl; STU=reports/alfworld/student3b_low100/final
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
hold() { squeue -u pan00389 -h -o "%i %j %T" | awk '$2=="error_attr_hold_igpu_qw" && $3=="RUNNING"{print $1}' | sort -n | tail -n1; }
S() { local H; until H=$(hold); [ -n "$H" ]; do sleep 120; done; echo "srun --jobid=$H --overlap -n1 -c4 --mem=0"; }
guard() { until [ "$($(S) nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1)" -ge "$1" ] 2>/dev/null; do sleep 60; done; }
until grep -q "DONE synth" $L/synth.log 2>/dev/null && [ -s $L/labels.jsonl ]; do sleep 120; done
# game-level split of the synthetic failures: 80% train / 20% held-out
$P - <<PY
import json, random
rows=[json.loads(l) for l in open("$L/synth.labels.jsonl")]; eps={json.loads(l)["id"]:json.loads(l) for l in open("$L/synth.jsonl")}
games=sorted({eps[r["id"]]["game_file"] for r in rows if r["id"] in eps}); rng=random.Random(0); rng.shuffle(games); held=set(games[:len(games)//5])
tr=[r for r in rows if r["id"] in eps and eps[r["id"]]["game_file"] not in held]; te=[r for r in rows if r["id"] in eps and eps[r["id"]]["game_file"] in held]
open("$L/synth.labels.train.jsonl","w").writelines(json.dumps(r)+"\n" for r in tr); open("$L/synth.labels.heldout.jsonl","w").writelines(json.dumps(r)+"\n" for r in te)
open("$L/synth.heldout_ids.jsonl","w").writelines(json.dumps({"id":r["id"]})+"\n" for r in te)
print(f"synth labels: {len(tr)} train / {len(te)} held-out (by game)")
PY
train() { local name=$1; shift; [ -f $A/$name/adapter/adapter_config.json ] && return
  guard 20000; $(S) $P -u scripts/alfworld_self_attributor.py train --init $STU --out $A/$name --epochs 2 --lr 1e-4 "$@" > $A/$name.train.log 2>&1 || log "attr r0 $name: TRAIN FAILED"; }
train synth  --labels $L/synth.labels.train.jsonl --trajs $L/synth.jsonl
train envlab --labels $L/labels.jsonl --trajs $TRAJ
train both   --labels $L/synth.labels.train.jsonl $L/labels.jsonl --trajs $L/synth.jsonl $TRAJ
point() { local name=$1 traj=$2 ids=$3 tag=$4; [ -s $A/$name/pointer_$tag.raw.jsonl ] && return
  guard 10000; $(S) $P -u scripts/alfworld_self_attributor.py pointer --traj $traj --ids $ids --adapter $A/$name/adapter --out $A/$name/pointer_$tag.jsonl > $A/$name.pointer_$tag.log 2>&1; }
for n in synth envlab both; do [ -f $A/$n/adapter/adapter_config.json ] || continue
  point $n $L/synth.jsonl $L/synth.heldout_ids.jsonl synthheld; point $n $TRAJ reports/alfworld/self_evolve3b/ids319.jsonl real319; done
$P - <<PY | tee -a $M
import json, statistics as st, os
A="$A"; L="$L"
held={json.loads(l)["id"]:json.loads(l) for l in open(f"{L}/synth.labels.heldout.jsonl")}
env={json.loads(l)["id"]:json.loads(l)["t_star"] for l in open(f"{L}/labels.jsonl")}
m5={json.loads(l)["id"]:json.loads(l)["steps"][0] for l in open("reports/alfworld/opd/student3b_low100/progress_nofb/steps.jsonl")}
mA={json.loads(l)["id"]:json.loads(l)["steps"][0] for l in open("reports/alfworld/opd/student3b_low100/attr_nofb/steps.jsonl")}
for n in ("synth","envlab","both"):
    p=f"{A}/{n}/pointer_synthheld.raw.jsonl"; q=f"{A}/{n}/pointer_real319.raw.jsonl"
    if not (os.path.exists(p) and os.path.exists(q)): continue
    H=[json.loads(l) for l in open(p)]; R=[json.loads(l) for l in open(q)]
    acc=sum(r["t_hat"]==held[r["id"]]["t_star"] for r in H)/len(H); acc1=sum(abs(r["t_hat"]-held[r["id"]]["t_star"])<=1 for r in H)/len(H)
    tacc=sum((r.get("type") or "")==held[r["id"]].get("type") for r in H)/len(H)
    ts=[r["t_hat"] for r in R]; ag=lambda ref: sum(abs(r["t_hat"]-ref[r["id"]])<=1 for r in R if r["id"] in ref)/max(1,sum(r["id"] in ref for r in R))
    print(f"=== ATTR R0 {n}: synth held-out ({len(H)}) step acc {acc:.2f} (+-1 {acc1:.2f}) type acc {tacc:.2f} | real 319: mean t {st.mean(ts):.1f} t0 {sum(t==0 for t in ts)/len(ts):.2f} t<=2 {sum(t<=2 for t in ts)/len(ts):.2f} t>=10 {sum(t>=10 for t in ts)/len(ts):.2f} | +-1 agreement: env-labels {ag(env):.2f} M5 {ag(m5):.2f} ModelA {ag(mA):.2f}")
PY
log "ATTR R0 DONE"
