#!/bin/bash
# One round of the FINAL recipe (user decision Oct 7 19:20: 20% corrections + 100 replay), parameterized:
#   bash scripts/run_k20_round.sh <chain dir under self_attr_rounds> <round> <start adapter> <774 tag> <replay: 1|0>
# 1500 train games (4 shards, T=1.0) -> 3B Model A step + diagnosis -> hindsight self-teacher (3 shards) -> generic one-step evidence (12 CPU
# shards) -> env top-K with K = round(0.2 x failures among the first 1000 games) -> [+100 replay of own shortest wins] -> new trainer -> 774 eval.
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
CH=$1; R=$2; START=$3; TAG=$4; REPLAY=$5; RULE=${6:-env_top}; D=$SA/$CH/r$R; T=$D/traj.jsonl; mkdir -p $D/self $D/proxy
log "$CH r$R: start on $(hostname) (from $START, replay=$REPLAY)"
collect4 $D $START ${CH}_r$R
log "$CH r$R: $(grep -c . $T) games, $(grep -c '"won": true' $T) won"
[ -s $D/pointer.raw.jsonl ] || modelA_pointer $D $T | tee -a $SA/queue.log
cp $D/pointer.jsonl $D/self/steps.jsonl
if ! grep -q "DONE informed_teacher" $D/teacher.log 2>/dev/null; then
  for i in 0 1 2; do ( until grep -q "DONE informed_teacher" $D/teacher_s$i.log 2>/dev/null; do $(S $GPU_MIN) $P -u scripts/self_teacher_informed.py --traj $T --steps $D/self/steps.jsonl --diag $D/diag_modelA.raw.jsonl --adapter $START --model $MODEL --out $D/self/targets_s$i.jsonl --shard $i 3 >> $D/teacher_s$i.log 2>&1 || sleep 30; done ) & sleep 60; done; wait
  cat $D/self/targets_s*.jsonl > $D/self/targets.jsonl; echo "DONE informed_teacher (merged 3 shards)" >> $D/teacher.log
fi
if ! grep -q "DONE onestep" $D/onestep.log 2>/dev/null; then
  for i in $(seq 0 11); do ( until grep -q "DONE onestep" $D/onestep_s$i.log 2>/dev/null; do SE_DIR=$D SE_TRAJ=$T $P -u scripts/proxy_eval_se3b.py onestep --arms self --limit 300 --shard $i 12 >> $D/onestep_s$i.log 2>&1; sleep 3; done ) & done; wait
  cat $D/proxy/onestep_s*.jsonl >> $D/proxy/onestep.jsonl; rm -f $D/proxy/onestep_s*.jsonl; echo "DONE onestep (merged 12 shards)" >> $D/onestep.log
fi
if [ "$RULE" = "env_top" ]; then
  NF=$(grep -c . $D/ids_fail.jsonl); K=$($P -c "print(max(1, round(0.2 * $NF)))"); log "$CH r$R: K = 0.2 x $NF failures = $K"
  [ -s $D/targets_env_top/train.jsonl ] || SE_DIR=$D SE_TRAJ=$T $P scripts/proxy_arms_se3b.py --arm self --rule env_top --k $K --out $D/targets_env_top | tee -a $SA/queue.log
elif [ "$RULE" = "rand20" ]; then   # ablation (Oct 8): RANDOM 20% of the failures, teacher top-1 at the Model-A state, no environment check
  [ -s $D/targets_all_top1/train.jsonl ] || SE_DIR=$D SE_TRAJ=$T $P scripts/proxy_arms_se3b.py --arm self --rule all_top1 --out $D/targets_all_top1 | tee -a $SA/queue.log
  NF=$(grep -c . $D/ids_fail.jsonl); K=$($P -c "print(max(1, round(0.2 * $NF)))"); mkdir -p $D/targets_rand20
  [ -s $D/targets_rand20/train.jsonl ] || $P -c "import random; L=open('$D/targets_all_top1/train.jsonl').readlines(); random.Random(100 * $R + ${SEED:-0}).shuffle(L); open('$D/targets_rand20/train.jsonl','w').writelines(L[:$K])"
  log "$CH r$R: random 20% = $(grep -c . $D/targets_rand20/train.jsonl) of $(grep -c . $D/targets_all_top1/train.jsonl) teacher top-1 targets (K = 0.2 x $NF)"
else   # e.g. all_top1 = no filter: the teacher's top-1 at every attributed failure state (baseline, Oct 7)
  [ -s $D/targets_$RULE/train.jsonl ] || SE_DIR=$D SE_TRAJ=$T $P scripts/proxy_arms_se3b.py --arm self --rule $RULE --out $D/targets_$RULE | tee -a $SA/queue.log
fi
DATA="$D/targets_$RULE/train.jsonl:1.0"
if [ "$REPLAY" = "1" ]; then [ -s $D/replay.jsonl ] || $P scripts/make_replay.py --traj $T --n 100 --seed $((R + 1000 * ${SEED:-0})) --out $D/replay.jsonl | tee -a $SA/queue.log; DATA="$DATA $D/replay.jsonl:1.0"; fi
train1 $D $START "$DATA"; [ -s $E/eval_heldout_$TAG.jsonl ] || eval774 $D $TAG
log "$CH r$R DONE ($TAG)"
