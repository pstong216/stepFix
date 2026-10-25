#!/bin/bash
# User-approved (Oct 6 21:35) baseline B3 / ablation C1, rollout verification with the FINAL steps: identical to the final method
# (3B Model A step + diagnosis, hindsight self-teacher, its top-2 candidates excluding a-, 40% budget, 100 replay, one-hot KL, same
# trainer and 774 evaluation) except the filter: a candidate is kept if a full rollout (student continuation to the 50-step budget, greedy)
# from the corrected state wins; then K = round(0.2 x failures) random states among those with a winning candidate (all if fewer);
# target = one-hot on the winning candidate with the higher teacher probability.  Verification uses the FIXED start policy of the round
# (alfworld_attr_opd.py distill --verify_only, 4 shards).  R1 reuses the final chain's R1 pointer/teacher outputs (same states, same candidates).
source scripts/lib_baselines.sh; CH=selfattr_rollsub; SUF=_a3b20; C=$SA/$CH$SUF; MA=reports/attributor_no_observations/final
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
# rollout_select DIR START TRAJ ROUND : verify the teacher's candidates by rollout (4 shards), keep K winners, write DIR/targets_rollout/train.jsonl
rollout_select() { local D=$1 AD=$2 T=$3 R=$4
  for i in 0 1 2 3; do ( until grep -q "DONE distill" $D/rollout_s$i.log 2>/dev/null; do $(S 16000) $P -u scripts/alfworld_attr_opd.py distill --traj $T --targets $D/self/targets.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter $AD --out $D/rollout_s$i --verify_k 2 --exclude_a_minus --max_examples_per_run 250 --max_continue_steps 0 --verify_only --shard $i 4 >> $D/rollout_s$i.log 2>&1 || sleep 30; done ) & sleep 60; done; wait
  cat $D/rollout_s{0,1,2,3}/examples.jsonl > $D/rollout_examples.jsonl; mkdir -p $D/targets_rollout
  $P - <<PY | tee -a $SA/queue.log
import json, random
D="$D"; tg={(json.loads(l)['id'],json.loads(l)['t']):json.loads(l) for l in open(f"{D}/self/targets.jsonl")}
nf=sum(1 for _ in open(f"{D}/ids_fail.jsonl")); K=max(1, round(0.2*nf)); rows=[]; ncand=0; nwin=0
for r in map(json.loads, open(f"{D}/rollout_examples.jsonl")):
    T=tg[(r['id'],r['t'])]; cmds=T['admissible']; alts=r.get('alternatives') or {}; ncand+=len(alts)
    wins=[c for c,w in alts.items() if w and c in cmds]; nwin+=len(wins)
    if not wins: continue
    c=max(wins, key=lambda c: T['teacher_logprobs'][c])
    rows.append({'kind':'kl','src':'$T','id':r['id'],'t':r['t'],'cmds':cmds,'target':[1.0 if x==c else 0.0 for x in cmds]})
random.Random($R*100).shuffle(rows); keep=rows[:K]
open(f"{D}/targets_rollout/train.jsonl","w").writelines(json.dumps(x)+"\n" for x in keep)
print(f"=== rollout ($CH$SUF r$R): {len(tg)} states, {ncand} candidates rolled out, {nwin} won; {len(rows)} states with a winner; K = 0.2 x {nf} = {K}; kept {len(keep)}")
PY
}
# (Oct 8, user request) ROLLOUT 20% + 100 replay: identical to the rollout baseline above (selfattr_rollsub_a3b) but K = 20% of the failures,
# the final recipe's budget. R1 reuses the rollout verification of selfattr_rollsub_a3b/r1 (same start policy, same states and candidates);
# only the K = 20% selection, training and evaluation are new. R2 and R3 verify their own candidates. 774 tags roll20_r1..r3.
# ---------- R1 ----------
D=$C/r1; mkdir -p $D/self; F=$SA/selfattr_a3bfinal/r1; RV=$SA/selfattr_rollsub_a3b/r1
for f in ids_fail.jsonl pointer.jsonl pointer.raw.jsonl diag.raw.jsonl diag_modelA.raw.jsonl modelA_pred.jsonl self/steps.jsonl self/targets.jsonl replay.jsonl; do [ -s $D/$f ] || cp $F/$f $D/$f; done
for i in 0 1 2 3; do [ -d $D/rollout_s$i ] || { cp -r $RV/rollout_s$i $D/; cp $RV/rollout_s$i.log $D/; }; done
log "chain $CH$SUF r1: start (states/candidates/replay from $F, rollout verification reused from $RV)"
[ -s $D/targets_rollout/train.jsonl ] || rollout_select $D $STU $T1 1
train1 $D $STU "$D/targets_rollout/train.jsonl:1.0 $D/replay.jsonl:1.0"; [ -s $E/eval_heldout_roll20_r1.jsonl ] || eval774 $D roll20_r1
# ---------- R2, R3 ----------
for R in 2 3; do
R1A=$C/r$((R-1))/model/adapter; D=$C/r$R; mkdir -p $D/self; T=$D/traj.jsonl
collect4 $D $R1A ${CH}${SUF}_r$R
log "chain $CH$SUF r$R: $(grep -c . $T) games, $(grep -c '"won": true' $T) won"
[ -s $D/pointer.raw.jsonl ] || modelA_pointer $D $T | tee -a $SA/queue.log
cp $D/pointer.jsonl $D/self/steps.jsonl
if ! grep -q "DONE informed_teacher" $D/teacher.log 2>/dev/null; then
  for i in 0 1 2; do ( until grep -q "DONE informed_teacher" $D/teacher_s$i.log 2>/dev/null; do $(S 12000) $P -u scripts/self_teacher_informed.py --traj $T --steps $D/self/steps.jsonl --diag $D/diag_modelA.raw.jsonl --adapter $R1A --out $D/self/targets_s$i.jsonl --shard $i 3 >> $D/teacher_s$i.log 2>&1 || sleep 30; done ) & sleep 60; done; wait
  cat $D/self/targets_s*.jsonl > $D/self/targets.jsonl; echo "DONE informed_teacher (merged 3 shards)" >> $D/teacher.log
fi
[ -s $D/targets_rollout/train.jsonl ] || rollout_select $D $R1A $T $R
[ -s $D/replay.jsonl ] || $P scripts/make_replay.py --traj $T --n 100 --seed $R --out $D/replay.jsonl | tee -a $SA/queue.log
train1 $D $R1A "$D/targets_rollout/train.jsonl:1.0 $D/replay.jsonl:1.0"; [ -s $E/eval_heldout_roll20_r$R.jsonl ] || eval774 $D roll20_r$R
[ -f $D/model/adapter/adapter_config.json ] || { log "chain $CH$SUF r$R ended WITHOUT a model"; exit 1; }
done
log "chain $CH$SUF R1->R3 DONE"
