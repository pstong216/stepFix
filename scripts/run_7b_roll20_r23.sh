#!/bin/bash
# User-requested (Oct 9 ~00:50): continue the 7B "rollout verification 20% + replay" ablation to R2 and R3, same protocol as its R1
# (scripts/run_7b_roll20_r1.sh) and as the 3B rollout-20% chain: collect 1500 train games with the previous 7B model, frozen 3B Model A step +
# diagnosis, 7B hindsight self-teacher (top-2 excluding a-), full-rollout verification with the round's FIXED 7B start policy, K = 0.2 x
# failures random states among those with a winner, + default (shortest-wins) 100-step replay, same trainer, 774 eval. Tags a7broll20_r2/r3.
export MODEL=Qwen/Qwen2.5-7B-Instruct GPU_MIN=20000 TRAIN_MIN=36000
source scripts/lib_baselines.sh; export EVIDENCE=generic; C=a7b_roll20; MA=reports/attributor_no_observations/final
modelA_pointer() { local D=$1 T=$2
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
for R in 2 3; do
  PA=$SA/$C/r$((R-1))/model/adapter; D=$SA/$C/r$R; mkdir -p $D/self; T=$D/traj.jsonl
  [ -f $D/model/adapter/adapter_config.json ] && [ -s $E/eval_heldout_a7broll20_r$R.jsonl ] && continue
  collect4 $D $PA ${C}_r$R
  log "$C r$R: $(grep -c . $T) games, $(grep -c '"won": true' $T) won"
  [ -s $D/pointer.raw.jsonl ] || modelA_pointer $D $T | tee -a $SA/queue.log
  cp $D/pointer.jsonl $D/self/steps.jsonl
  if ! grep -q "DONE informed_teacher" $D/teacher.log 2>/dev/null; then
    for i in 0 1 2; do ( until grep -q "DONE informed_teacher" $D/teacher_s$i.log 2>/dev/null; do $(S $GPU_MIN) $P -u scripts/self_teacher_informed.py --traj $T --steps $D/self/steps.jsonl --diag $D/diag_modelA.raw.jsonl --adapter $PA --model $MODEL --out $D/self/targets_s$i.jsonl --shard $i 3 >> $D/teacher_s$i.log 2>&1 || sleep 30; done ) & sleep 60; done; wait
    cat $D/self/targets_s*.jsonl > $D/self/targets.jsonl; echo "DONE informed_teacher (merged 3 shards)" >> $D/teacher.log
  fi
  for i in 0 1 2 3; do ( until grep -q "DONE distill" $D/rollout_s$i.log 2>/dev/null; do $(S $GPU_MIN) $P -u scripts/alfworld_attr_opd.py distill --traj $T --targets $D/self/targets.jsonl --student $MODEL --adapter $PA --out $D/rollout_s$i --verify_k 2 --exclude_a_minus --max_examples_per_run 250 --max_continue_steps 0 --verify_only --shard $i 4 >> $D/rollout_s$i.log 2>&1 || sleep 3; done ) & sleep 60; done; wait
  cat $D/rollout_s{0,1,2,3}/examples.jsonl > $D/rollout_examples.jsonl; mkdir -p $D/targets_rollout
  [ -s $D/targets_rollout/train.jsonl ] || $P - <<PY | tee -a $SA/queue.log
import json, random
D="$D"; tg={(json.loads(l)['id'],json.loads(l)['t']):json.loads(l) for l in open(f"{D}/self/targets.jsonl")}
nf=sum(1 for _ in open(f"{D}/ids_fail.jsonl")); K=max(1, round(0.2*nf)); rows=[]; ncand=0; nwin=0; seen=set()
for r in map(json.loads, open(f"{D}/rollout_examples.jsonl")):
    k=(r['id'],r['t'])
    if k in seen: continue
    seen.add(k); T=tg[k]; cmds=T['admissible']; alts=r.get('alternatives') or {}; ncand+=len(alts)
    wins=[c for c,w in alts.items() if w and c in cmds]; nwin+=len(wins)
    if not wins: continue
    c=max(wins, key=lambda c: T['teacher_logprobs'][c])
    rows.append({'kind':'kl','src':'$T','id':r['id'],'t':r['t'],'cmds':cmds,'target':[1.0 if x==c else 0.0 for x in cmds]})
random.Random($R*100).shuffle(rows); keep=rows[:K]
open(f"{D}/targets_rollout/train.jsonl","w").writelines(json.dumps(x)+"\n" for x in keep)
print(f"=== rollout (a7b_roll20 r$R): {len(tg)} states, {ncand} candidates rolled out, {nwin} won; {len(rows)} states with a winner; K = 0.2 x {nf} = {K}; kept {len(keep)}")
PY
  [ -s $D/replay.jsonl ] || $P scripts/make_replay.py --traj $T --n 100 --seed $R --out $D/replay.jsonl | tee -a $SA/queue.log
  train1 $D $PA "$D/targets_rollout/train.jsonl:1.0 $D/replay.jsonl:1.0"; [ -s $E/eval_heldout_a7broll20_r$R.jsonl ] || eval774 $D a7broll20_r$R
  [ -f $D/model/adapter/adapter_config.json ] || { log "$C r$R ended WITHOUT a model"; exit 1; }
done
log "$C R2->R3 DONE"
