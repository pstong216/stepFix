#!/bin/bash
# Does the log-rule step selection matter?  Same clean R1 failures, same informed teacher, same env filter (top-86), only the STEP:
#   clean_sel_random -- uniformly random step in 0..min(7, len-1)   (no attribution at all)
#   clean_sel_last   -- the failure's last step (budget exhausted there; most naive "where it failed")
# Teacher candidates must be regenerated at the new steps (GPU), then one-step scan (CPU), then env_top 86 -> queue_clean.txt.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; SA=reports/alfworld/self_attr_rounds; C=$SA/selfattr_clean/r1
T=reports/alfworld/student3b_low100_train_b50.jsonl; R=reports/alfworld/rigorous; Q=$R/queue_clean_sel.txt; M=$SA/queue.log; L=$R/data/clean_build.log
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
pick() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge 14000 ] 2>/dev/null && echo $best; }
: > $Q
$P - <<PY
import json, random
C="$C"; T="$T"; rng=random.Random(7)
eps={e["id"]:e for e in map(json.loads,open(T))}; ids=[json.loads(l)["id"] for l in open(f"{C}/ids_fail.jsonl") if json.loads(l)["id"] in eps]
import os
for name, fn in [("random", lambda e: rng.randrange(0, min(8, len(e["steps"])))), ("last", lambda e: len(e["steps"])-1)]:
    d=f"{C}/sel_{name}/self"; os.makedirs(d, exist_ok=True)
    rows=[{"id":i,"steps":[fn(eps[i])],"mode":None} for i in ids]
    open(f"{d}/steps.jsonl","w").writelines(json.dumps(r)+"\n" for r in rows)
    ts=[r["steps"][0] for r in rows]; print(f"sel_{name}: {len(rows)} states, mean t {sum(ts)/len(ts):.1f}, t0 {sum(t==0 for t in ts)/len(ts):.2f}")
PY
for name in random last; do D=$C/sel_$name; mkdir -p $D/proxy; arm=clean_sel_$name
  until grep -q "DONE informed_teacher" $D/teacher.log 2>/dev/null; do H=""; until H=$(pick); [ -n "$H" ]; do sleep 60; done
    srun --jobid=$H --overlap -n1 -c4 --mem=0 $P -u scripts/self_teacher_informed.py --traj $T --steps $D/self/steps.jsonl --diag $C/diag_modelA.raw.jsonl --adapter reports/alfworld/student3b_low100/final --out $D/self/targets.jsonl >> $D/teacher.log 2>&1 || sleep 60; done
  until grep -q "DONE onestep" $D/onestep.log 2>/dev/null; do SE_DIR=$D SE_TRAJ=$T $P -u scripts/proxy_eval_se3b.py onestep --arms self --limit 300 >> $D/onestep.log 2>&1; sleep 3; done
  SE_DIR=$D SE_TRAJ=$T $P scripts/proxy_arms_se3b.py --arm self --rule env_top --k 86 --out $D/arm >> $L 2>&1
  [ -s $D/arm/train.jsonl ] && printf "%s\t%s:1.0\n" "$arm" "$D/arm/train.jsonl" >> $Q && log "clean selector arm $arm queued ($(wc -l < $D/arm/train.jsonl) targets)" || log "clean selector arm $arm: BUILD FAILED"
done
tmux new-session -d -s rig_clean_sel "RQ=$Q SEEDS='1' bash scripts/run_rigorous_arms_v3.sh"
log "clean selector arms launched (random step / last step vs log rule, same teacher+filter)"
