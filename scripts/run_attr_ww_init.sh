#!/bin/bash
# Attributor v2: synth x3 + env labels + balanced divergence labels (see self_attr_labels/v2/SPECS), 1 epoch, init = student LoRA.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
L=reports/alfworld/self_attr_labels; A=reports/alfworld/self_attr/ww_init; M=reports/alfworld/self_evolve3b/queue.log; STU=reports/alfworld/student3b_low100/final
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
pick() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge "${1:-10000}" ] 2>/dev/null && echo $best; }
LAB=$(sed -n 1p $L/v2/SPECS); TRJ=$(sed -n 2p $L/v2/SPECS)
until [ -f $A/adapter/adapter_config.json ]; do H=""; until H=$(pick 20000); [ -n "$H" ]; do sleep 60; done
  srun --jobid=$H --overlap -n1 -c4 --mem=0 $P -u scripts/alfworld_self_attributor.py train --init reports/attributor_no_observations/final --out $A --epochs 2 --lr 1e-4 --labels $L/r0/synth.labels.train.jsonl $L/r0/labels.jsonl --trajs $L/r0/synth.jsonl reports/alfworld/student3b_low100_train_b50.jsonl > $A/train.log 2>&1 || { log "attr ww-init: TRAIN FAILED"; sleep 120; }; done
for tag in synthheld:$L/r0/synth.jsonl:$L/r0/synth.heldout_ids.jsonl real319:reports/alfworld/student3b_low100_train_b50.jsonl:reports/alfworld/self_evolve3b/ids319.jsonl; do IFS=: read -r t traj ids <<< "$tag"
  until [ -s $A/pointer_$t.raw.jsonl ]; do H=""; until H=$(pick 10000); [ -n "$H" ]; do sleep 60; done
    srun --jobid=$H --overlap -n1 -c4 --mem=0 $P -u scripts/alfworld_self_attributor.py pointer --traj $traj --ids $ids --adapter $A/adapter --out $A/pointer_$t.jsonl > $A/pointer_$t.log 2>&1 || sleep 60; done; done
$P - <<PY | tee -a $M
import json, statistics as st
A="$A"; L="$L"
held={json.loads(l)["id"]:json.loads(l) for l in open(f"{L}/r0/synth.labels.heldout.jsonl")}
env={json.loads(l)["id"]:json.loads(l)["t_star"] for l in open(f"{L}/r0/labels.jsonl")}
div={json.loads(l)["id"]:json.loads(l)["t_star"] for l in open(f"{L}/div/labels.jsonl")}
m5={json.loads(l)["id"]:json.loads(l)["steps"][0] for l in open("reports/alfworld/opd/student3b_low100/progress_nofb/steps.jsonl")}
H=[json.loads(l) for l in open(f"{A}/pointer_synthheld.raw.jsonl")]; R=[json.loads(l) for l in open(f"{A}/pointer_real319.raw.jsonl")]
acc=sum(r["t_hat"]==held[r["id"]]["t_star"] for r in H)/len(H); acc1=sum(abs(r["t_hat"]-held[r["id"]]["t_star"])<=1 for r in H)/len(H); tacc=sum((r.get("type") or "")==held[r["id"]].get("type") for r in H)/len(H)
ts=[r["t_hat"] for r in R]; ag=lambda ref: sum(abs(r["t_hat"]-ref[r["id"]])<=1 for r in R if r["id"] in ref)/max(1,sum(r["id"] in ref for r in R))
print(f"=== ATTR WW-INIT (3B Who&When-Pro attributor as init, continued on own R0 labels): synth held-out ({len(H)}) step acc {acc:.2f} (+-1 {acc1:.2f}) type acc {tacc:.2f} | real 319: mean t {st.mean(ts):.1f} t0 {sum(t==0 for t in ts)/len(ts):.2f} t<=2 {sum(t<=2 for t in ts)/len(ts):.2f} t>=5 {sum(t>=5 for t in ts)/len(ts):.2f} | +-1 agreement: env-labels {ag(env):.2f} divergence {ag(div):.2f} M5 {ag(m5):.2f}")
PY
log "ATTR WW-INIT (3B Who&When-Pro attributor as init, continued on own R0 labels) DONE"
