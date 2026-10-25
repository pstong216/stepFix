#!/bin/bash
# Oct 9 (user autonomy note, 7B R3): R3 from the ORIGINAL 7B R2 (576) on the same R3 collection (a7b_final/r3/traj.jsonl), with
#   (a) Model A input = FULL trajectory (variant full: every observation + the initial room description), same frozen 3B weights,
#       unless MODELA_VARIANT is set (decided from the gold-label check in reports/alfworld/modelA_fullinfo_check);
#   (b) hindsight self-teacher with --full_view (complete prefix, every future step, full observations);
#   (c) env_top_margin (a correction must strictly beat the taken action's one-step evidence), K = 0.2 x failures;
#   (d) fail_weighted replay. Same trainer, 774 eval. Tag a7b_r3v3 (or $TAG).
export MODEL=Qwen/Qwen2.5-7B-Instruct GPU_MIN=20000 TRAIN_MIN=36000 EVIDENCE=generic
source scripts/lib_baselines.sh; S3=$SA/a7b_final/r3; D=$SA/a7b_final/${OUTDIR:-r3_v3}; T=$S3/traj.jsonl; MA=${MA:-reports/attributor_no_observations/final}
VAR=${MODELA_VARIANT:-full}; TAG=${TAG:-a7b_r3v3}; START=$SA/a7b_final/r2/model/adapter; mkdir -p $D/self $D/proxy
cp $S3/ids_fail.jsonl $D/ids_fail.jsonl
until grep -q DONE $D/modelA.log 2>/dev/null; do $(S ${MA_MIN:-14000}) $P -u scripts/alfworld_attribute.py --attributor $MA --variant $VAR --release ${T%.jsonl}.release.jsonl --out $D/modelA_pred.jsonl >> $D/modelA.log 2>&1 && echo DONE >> $D/modelA.log || sleep 60; done
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
old={json.loads(l)["id"]:json.loads(l)["t_hat"] for l in open("$S3/pointer.raw.jsonl")}
ts=[r["t_hat"] for r in rows]; same=sum(old.get(r["id"])==r["t_hat"] for r in rows)
print("=== $TAG Model A ($VAR): %d states, mean t %.2f, t0 %.2f; same step as the standard pointer in %d/%d" % (len(rows), sum(ts)/len(ts), sum(t==0 for t in ts)/len(ts), same, len(rows)))
PY
cp $D/pointer.jsonl $D/self/steps.jsonl
if ! grep -q "DONE informed_teacher" $D/teacher.log 2>/dev/null; then
  for i in 0 1 2; do ( until grep -q "DONE informed_teacher" $D/teacher_s$i.log 2>/dev/null; do $(S $GPU_MIN) $P -u scripts/self_teacher_informed.py --traj $T --steps $D/self/steps.jsonl --diag $D/diag_modelA.raw.jsonl --adapter $START --model $MODEL --out $D/self/targets_s$i.jsonl --shard $i 3 ${TEACHER_FLAGS---full_view} >> $D/teacher_s$i.log 2>&1 || sleep 30; done ) & sleep 60; done; wait
  cat $D/self/targets_s*.jsonl > $D/self/targets.jsonl; echo "DONE informed_teacher (merged 3 shards)" >> $D/teacher.log
fi
$P -c "
import json; S={(json.loads(l)['id'],json.loads(l)['steps'][0]) for l in open('$D/self/steps.jsonl')}; R=[(json.loads(l)['id'],json.loads(l)['t']) for l in open('$D/self/targets.jsonl')]
print('=== $TAG teacher coverage: states', len(S), 'covered', len(set(R)&S), 'rows', len(R)); assert set(R)==S and len(R)==len(S), 'TEACHER COVERAGE INCOMPLETE'" | tee -a $SA/queue.log || exit 1
[ -s $D/proxy/onestep.jsonl ] || cp $S3/proxy/onestep.jsonl $D/proxy/onestep.jsonl   # cache: identical (state, candidate) executions are reused
if ! grep -q "DONE onestep" $D/onestep.log 2>/dev/null; then   # 12 CPU shards as in run_k20_round.sh, then a single-process pass that must find nothing left
  for i in $(seq 0 11); do ( until grep -q "DONE onestep" $D/onestep_s$i.log 2>/dev/null; do SE_DIR=$D SE_TRAJ=$T $P -u scripts/proxy_eval_se3b.py onestep --arms self --limit 300 --shard $i 12 >> $D/onestep_s$i.log 2>&1; sleep 3; done ) & done; wait
  cat $D/proxy/onestep_s*.jsonl >> $D/proxy/onestep.jsonl 2>/dev/null; rm -f $D/proxy/onestep_s*.jsonl
  until SE_DIR=$D SE_TRAJ=$T $P -u scripts/proxy_eval_se3b.py onestep --arms self --limit 5000 > $D/onestep_final.log 2>&1 && grep -q "DONE onestep" $D/onestep_final.log; do sleep 3; done
  echo "DONE onestep (12 shards + final pass: $(head -n 1 $D/onestep_final.log))" >> $D/onestep.log
fi
NF=$(grep -c . $D/ids_fail.jsonl); K=$($P -c "print(max(1, round(0.2 * $NF)))")
[ -s $D/targets_env_top_margin/train.jsonl ] || SE_DIR=$D SE_TRAJ=$T $P scripts/proxy_arms_se3b.py --arm self --rule env_top_margin --k $K --out $D/targets_env_top_margin | tee -a $SA/queue.log
[ -s $D/replay.jsonl ] || $P scripts/make_replay.py --traj $T --n 100 --seed 3 --mode fail_weighted --out $D/replay.jsonl | tee -a $SA/queue.log
train1 $D $START "$D/targets_env_top_margin/train.jsonl:1.0 $D/replay.jsonl:1.0"; [ -s $E/eval_heldout_$TAG.jsonl ] || eval774 $D $TAG
log "a7b $TAG DONE (Model A $VAR + full-view teacher + margin + fail_weighted replay; R2 = 576, R3 = 540, R3 v2 = see a7b_r3v2)"
