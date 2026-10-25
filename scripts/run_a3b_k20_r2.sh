#!/bin/bash
# User-requested (Oct 7 00:30): R2 of the 20% chain -- final recipe with K = round(0.2 x failures) + 100 replay, from a3bfinal_k20/r1 (537/774).
# Same as selfattr_a3bfinal R2 except K (20% instead of 40%): 1500 train games (4 shards), 3B Model A step + diagnosis, hindsight self-teacher
# (3 shards), generic one-step evidence (12 CPU shards), env top-K, 100 replay, new trainer, 774 evaluation.
# Runs on a GPU-hold node (aga29, job ends ~04:40) so it survives the cpu_tmux node (ends ~02:10) before the 05:00 maintenance.
source scripts/lib_baselines.sh; export EVIDENCE=generic; MA=reports/attributor_no_observations/final
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
R1A=$SA/a3bfinal_k20/r1/model/adapter; D=$SA/a3bfinal_k20/r2; T=$D/traj.jsonl; mkdir -p $D/self $D/proxy
log "a3bfinal_k20 r2: start on $(hostname) (from $R1A)"
collect4 $D $R1A a3bk20_r2
log "a3bfinal_k20 r2: $(grep -c . $T) games, $(grep -c '"won": true' $T) won"
[ -s $D/pointer.raw.jsonl ] || modelA_pointer $D $T | tee -a $SA/queue.log
cp $D/pointer.jsonl $D/self/steps.jsonl
if ! grep -q "DONE informed_teacher" $D/teacher.log 2>/dev/null; then
  for i in 0 1 2; do ( until grep -q "DONE informed_teacher" $D/teacher_s$i.log 2>/dev/null; do $(S 12000) $P -u scripts/self_teacher_informed.py --traj $T --steps $D/self/steps.jsonl --diag $D/diag_modelA.raw.jsonl --adapter $R1A --out $D/self/targets_s$i.jsonl --shard $i 3 >> $D/teacher_s$i.log 2>&1 || sleep 30; done ) & sleep 60; done; wait
  cat $D/self/targets_s*.jsonl > $D/self/targets.jsonl; echo "DONE informed_teacher (merged 3 shards)" >> $D/teacher.log
fi
if ! grep -q "DONE onestep" $D/onestep.log 2>/dev/null; then
  for i in $(seq 0 11); do ( until grep -q "DONE onestep" $D/onestep_s$i.log 2>/dev/null; do SE_DIR=$D SE_TRAJ=$T $P -u scripts/proxy_eval_se3b.py onestep --arms self --limit 300 --shard $i 12 >> $D/onestep_s$i.log 2>&1; sleep 3; done ) & done; wait
  cat $D/proxy/onestep_s*.jsonl >> $D/proxy/onestep.jsonl; rm -f $D/proxy/onestep_s*.jsonl; echo "DONE onestep (merged 12 shards)" >> $D/onestep.log
fi
NF=$(grep -c . $D/ids_fail.jsonl); K=$($P -c "print(max(1, round(0.2 * $NF)))"); log "a3bfinal_k20 r2: K = 0.2 x $NF failures = $K"
[ -s $D/targets_env_top/train.jsonl ] || SE_DIR=$D SE_TRAJ=$T $P scripts/proxy_arms_se3b.py --arm self --rule env_top --k $K --out $D/targets_env_top | tee -a $SA/queue.log
[ -s $D/replay.jsonl ] || $P scripts/make_replay.py --traj $T --n 100 --seed 2 --out $D/replay.jsonl | tee -a $SA/queue.log
train1 $D $R1A "$D/targets_env_top/train.jsonl:1.0 $D/replay.jsonl:1.0"; [ -s $E/eval_heldout_a3bk20_r2.jsonl ] || eval774 $D a3bk20_r2
log "a3bfinal_k20 R1->R2 DONE (compare 40%+replay chain R2 = 572)"
