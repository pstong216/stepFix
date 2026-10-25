#!/bin/bash
# User-requested (Oct 9 ~15:40): continue the 7B chain from the ORIGINAL 7B R1 (549) with the 7B Model A and the improved recipe for R2 and
# R3 (= the v3b recipe that gave R3 602 from the original R2): 7B Model A (attributor_7b_no_observations, its no-observation input) picks the
# step; hindsight self-teacher with --full_view; generic one-step evidence; env_top_margin with K = 0.2 x failures among the first 1000;
# fail_weighted replay; same trainer; 774 eval. R2 reuses a7b_final/r2's collection (made by the R1 model, independent of the recipe); R3
# collects 1500 new games with the new R2 model. Tags a7bv3b_r2, a7bv3b_r3.
export MODEL=Qwen/Qwen2.5-7B-Instruct GPU_MIN=20000 TRAIN_MIN=36000 EVIDENCE=generic
source scripts/lib_baselines.sh; C=$SA/a7b_v3b; MA=reports/attributor_7b_no_observations/final
vround() { local R=$1 D=$C/r$1 SRC=$2 START=$3 TAG=a7bv3b_r$1; mkdir -p $D/self $D/proxy; local T=$SRC/traj.jsonl
  [ -f $D/model/adapter/adapter_config.json ] && [ -s $E/eval_heldout_$TAG.jsonl ] && return
  if [ "$SRC" = "$D" ]; then collect4 $D $START a7b_v3b_r$R; fi
  $P -c "import json; eps=[json.loads(l) for l in open('$T')][:1000]; open('$D/ids_fail.jsonl','w').writelines(json.dumps({'id':e['id']})+'\n' for e in eps if not e['won'])"
  log "a7b_v3b r$R: $(grep -c . $T) games, $(grep -c '"won": true' $T) won; $(grep -c . $D/ids_fail.jsonl) failures in the first 1000 (from $START)"
  until grep -q DONE $D/modelA.log 2>/dev/null; do $(S 22000) $P -u scripts/alfworld_attribute.py --attributor $MA --variant no_observations --release ${T%.jsonl}.release.jsonl --out $D/modelA_pred.jsonl >> $D/modelA.log 2>&1 && echo DONE >> $D/modelA.log || sleep 60; done
  [ -s $D/pointer.raw.jsonl ] || $P - <<PY | tee -a $SA/queue.log
import json
D="$D"; ids={json.loads(l)["id"] for l in open(f"{D}/ids_fail.jsonl")}; eps={json.loads(l)["id"]:json.loads(l) for l in open("$T")}
pred={r["id"]:r for r in map(json.loads, open(f"{D}/modelA_pred.jsonl"))}; rows=[]
for i in ids:
    if i not in pred or i not in eps: continue
    t=max(0, min(int(pred[i]["t_hat"] if pred[i]["t_hat"] is not None else 0), len(eps[i]["steps"])-1)); rows.append({"id":i,"t_hat":t,"type":None})
open(f"{D}/pointer.jsonl","w").writelines(json.dumps({"id":r["id"],"steps":[r["t_hat"]],"mode":None})+"\n" for r in rows)
open(f"{D}/pointer.raw.jsonl","w").writelines(json.dumps(r)+"\n" for r in rows)
open(f"{D}/diag_modelA.raw.jsonl","w").writelines(json.dumps({"id":r["id"],"t_hat":pred[r["id"]]["t_hat"],"type":"Model-A mode "+str((pred[r["id"]]["pred"] or {}).get("error_mode"))})+"\n" for r in rows)
ts=[r["t_hat"] for r in rows]; print("=== a7b_v3b r$R 7B Model A pointer: %d states, mean t %.2f, t0 %.2f" % (len(rows), sum(ts)/len(ts), sum(t==0 for t in ts)/len(ts)))
PY
  cp $D/pointer.jsonl $D/self/steps.jsonl
  if ! grep -q "DONE informed_teacher" $D/teacher.log 2>/dev/null; then
    for i in 0 1 2; do ( until grep -q "DONE informed_teacher" $D/teacher_s$i.log 2>/dev/null; do $(S $GPU_MIN) $P -u scripts/self_teacher_informed.py --traj $T --steps $D/self/steps.jsonl --diag $D/diag_modelA.raw.jsonl --adapter $START --model $MODEL --out $D/self/targets_s$i.jsonl --shard $i 3 --full_view >> $D/teacher_s$i.log 2>&1 || sleep 30; done ) & sleep 60; done; wait
    cat $D/self/targets_s*.jsonl > $D/self/targets.jsonl; echo "DONE informed_teacher (merged 3 shards)" >> $D/teacher.log
  fi
  $P -c "
import json; S={(json.loads(l)['id'],json.loads(l)['steps'][0]) for l in open('$D/self/steps.jsonl')}; R=[(json.loads(l)['id'],json.loads(l)['t']) for l in open('$D/self/targets.jsonl')]
print('=== $TAG teacher coverage: states', len(S), 'covered', len(set(R)&S), 'rows', len(R)); assert set(R)==S and len(R)==len(S), 'TEACHER COVERAGE INCOMPLETE'" | tee -a $SA/queue.log || exit 1
  if ! grep -q "DONE onestep" $D/onestep.log 2>/dev/null; then
    for i in $(seq 0 11); do ( until grep -q "DONE onestep" $D/onestep_s$i.log 2>/dev/null; do SE_DIR=$D SE_TRAJ=$T $P -u scripts/proxy_eval_se3b.py onestep --arms self --limit 300 --shard $i 12 >> $D/onestep_s$i.log 2>&1; sleep 3; done ) & done; wait
    cat $D/proxy/onestep_s*.jsonl >> $D/proxy/onestep.jsonl 2>/dev/null; rm -f $D/proxy/onestep_s*.jsonl
    until SE_DIR=$D SE_TRAJ=$T $P -u scripts/proxy_eval_se3b.py onestep --arms self --limit 5000 > $D/onestep_final.log 2>&1 && grep -q "DONE onestep" $D/onestep_final.log; do sleep 3; done
    echo "DONE onestep" >> $D/onestep.log
  fi
  local NF=$(grep -c . $D/ids_fail.jsonl); local K=$($P -c "print(max(1, round(0.2 * $NF)))")
  [ -s $D/targets_env_top_margin/train.jsonl ] || SE_DIR=$D SE_TRAJ=$T $P scripts/proxy_arms_se3b.py --arm self --rule env_top_margin --k $K --out $D/targets_env_top_margin | tee -a $SA/queue.log
  [ -s $D/replay.jsonl ] || $P scripts/make_replay.py --traj $T --n 100 --seed $R --mode fail_weighted --out $D/replay.jsonl | tee -a $SA/queue.log
  train1 $D $START "$D/targets_env_top_margin/train.jsonl:1.0 $D/replay.jsonl:1.0"; [ -s $E/eval_heldout_$TAG.jsonl ] || eval774 $D $TAG
  [ -f $D/model/adapter/adapter_config.json ] || { log "a7b_v3b r$R ended WITHOUT a model"; exit 1; }
  log "a7b_v3b r$R DONE ($TAG)"; }
vround 2 $SA/a7b_final/r2 $SA/a7b_final/r1/model/adapter
vround 3 $C/r3 $C/r2/model/adapter
log "a7b_v3b R2->R3 DONE (7B Model A + full-view teacher + margin + fail_weighted from the original 7B R1)"
