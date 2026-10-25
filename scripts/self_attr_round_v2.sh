#!/bin/bash
# One round of the SELF-ATTRIBUTION chains, v2 (from round 2 on).  Change vs v1: the step selector of the main chain is the
# MODEL-FREE env-evidence attribution (for each failure, the student's top-2 alternatives at each of the first 8 steps are executed
# one step in the environment; t* = first step with a clearly better alternative) -- the trained log-reading attributor collapsed to
# the step-0/1 prior (ATTR R0/V2), so it is kept only as a per-round distillation side product whose agreement is logged.
#   self_attr_round_v2.sh <chain> <round> <start adapter> <prev attributor adapter | M5 estimator> <hold job name>
# chains: selfattr (env-evidence steps + informed teacher + two-layer + replay) | selfattr_plain (env-evidence steps + plain teacher
#         + env_top) | m5_informed (frozen M5 steps + informed teacher + two-layer + replay)
set -u; CH=$1; R=$2; START=$3; PREV=$4; HOLD=$5
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
B=reports/alfworld/self_attr_rounds; D=$B/$CH/r$R; mkdir -p $D/self $D/proxy $D/envlab; M=$B/queue.log
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
hold() { squeue -u pan00389 -h -o "%i %j %T" | awk -v n="$HOLD" '$2==n && $3=="RUNNING"{print $1}' | sort -n | tail -n1; }
S() { local H; until H=$(hold); [ -n "$H" ]; do sleep 120; done; echo "srun --jobid=$H --overlap -n1 -c4 --mem=0"; }
guard() { until [ "$($(S) nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1)" -ge "$1" ] 2>/dev/null; do sleep 60; done; }
log "chain $CH round $R (v2): start (from $START)"
# 1. trajectories
if [ "$R" = "1" ]; then T=reports/alfworld/student3b_low100_train_b50.jsonl
else T=$D/traj.jsonl
  until tail -n 3 $D/collect.log 2>/dev/null | grep -q "^DONE"; do
    guard 10000; $(S) $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $START --split train --n_games 1500 --budget 50 --temperature 1.0 --out $T --tag ${CH}_r$R --max_episodes_per_run 450 >> $D/collect.log 2>&1 || sleep 60
  done
fi
NW=$(grep -c '"won": true' $T); NF=$(grep -c '"won": false' $T); log "chain $CH r$R: $(wc -l < $T) games, $NW won, $NF failed"
$P - <<PY
import json
eps=[json.loads(l) for l in open("$T")][:1000]
open("$D/ids_fail.jsonl","w").writelines(json.dumps({"id":e["id"]})+"\n" for e in eps if not e["won"])
PY
# 2.+3. step selection
if [ "$CH" = "m5_informed" ]; then
  guard 10000; [ -s $D/m5_scores.jsonl ] || $(S) $P -u scripts/alfworld_progress_model.py score --traj $T --adapter $PREV --ids_from $D/ids_fail.jsonl --out $D/m5_scores.jsonl > $D/m5.log 2>&1
  $P scripts/alfworld_progress_model.py pointer --scores $D/m5_scores.jsonl --out $D/m5_pointer.jsonl --delta 1.0 >> $D/m5.log 2>&1
  $P - <<PY
import json
rows=[json.loads(l) for l in open("$D/m5_pointer.jsonl")]
open("$D/self/steps.jsonl","w").writelines(json.dumps({"id":r["id"],"steps":[r["t_hat"]],"mode":None})+"\n" for r in rows)
open("$D/diag.raw.jsonl","w").writelines(json.dumps({"id":r["id"],"t_hat":r["t_hat"],"type":None})+"\n" for r in rows)
PY
  ATTR=$PREV
else
  # env-evidence attribution: GPU scoring of the student's top-2 alternatives at the first 8 steps, CPU one-step execution (4 shards), label
  until grep -q "DONE score" $D/envlab/score.log 2>/dev/null; do guard 10000; $(S) $P -u scripts/attr_labels_env.py score --traj $T --ids $D/ids_fail.jsonl --out_dir $D/envlab --k 8 --adapter $START >> $D/envlab/score.log 2>&1 || sleep 30; done
  for i in 0 1 2 3; do ( until grep -q "DONE onestep" $D/envlab/onestep_s$i.log 2>/dev/null; do $P -u scripts/attr_labels_env.py onestep --traj $T --ids $D/ids_fail.jsonl --out_dir $D/envlab --k 8 --shard $i 4 >> $D/envlab/onestep_s$i.log 2>&1; sleep 3; done ) & done; wait
  $P scripts/attr_labels_env.py label --traj $T --ids $D/ids_fail.jsonl --out_dir $D/envlab --k 8 2>&1 | tee -a $M
  cp $D/envlab/steps.jsonl $D/self/steps.jsonl
  $P - <<PY
import json
rows=[json.loads(l) for l in open("$D/envlab/labels.jsonl")]
open("$D/diag.raw.jsonl","w").writelines(json.dumps({"id":r["id"],"t_hat":r["t_star"],"type":r.get("type")})+"\n" for r in rows)
PY
  ATTR=$PREV
fi
PSTAT=$($P - <<PY
import json, statistics as st, collections
R=[json.loads(l) for l in open("$D/diag.raw.jsonl")]; ts=[r["t_hat"] for r in R]; ty=collections.Counter(r.get("type") for r in R)
print(f"{len(ts)} states, mean t {st.mean(ts):.1f}, t0 {sum(t==0 for t in ts)/len(ts):.2f}, t>=5 {sum(t>=5 for t in ts)/len(ts):.2f}, types {dict(ty.most_common(4))}")
PY
)
log "chain $CH r$R: pointer $PSTAT"
# 4. teacher candidates
case $CH in selfattr_plain) TOPT="--no_hind --no_diag --no_analysis" ;; *) TOPT="" ;; esac
until grep -q "DONE informed_teacher" $D/teacher.log 2>/dev/null; do
  guard 12000; $(S) $P -u scripts/self_teacher_informed.py --traj $T --steps $D/self/steps.jsonl --diag $D/diag.raw.jsonl --adapter $START --out $D/self/targets.jsonl $TOPT >> $D/teacher.log 2>&1 || sleep 30; done
# 5. one-step evidence for the teacher's candidates (CPU) and the filter
until grep -q "DONE onestep" $D/onestep.log 2>/dev/null; do SE_DIR=$D SE_TRAJ=$T $P -u scripts/proxy_eval_se3b.py onestep --arms self --limit 300 >> $D/onestep.log 2>&1; sleep 3; done
case $CH in selfattr_plain) RULE=env_top ;; *) RULE=two_layer ;; esac
SE_DIR=$D SE_TRAJ=$T $P scripts/proxy_arms_se3b.py --arm self --rule $RULE --k 86 --out $D/targets_$RULE | tee -a $M
DATA="$D/targets_$RULE/train.jsonl:1.0"
if [ "$CH" != "selfattr_plain" ]; then $P scripts/make_replay.py --traj $T --n 86 --seed $R --out $D/replay.jsonl | tee -a $M; DATA="$DATA $D/replay.jsonl:1.0"; fi
# 6. train, 7. eval
guard 24000; [ -f $D/model/adapter/adapter_config.json ] || $(S) $P -u scripts/alfworld_opd_plus.py train --data $DATA --student Qwen/Qwen2.5-3B-Instruct --adapter $PWD/$START --lr 1e-5 --epochs 1 --batch 8 --out $D/model > $D/train.log 2>&1 || { log "chain $CH r$R: TRAIN FAILED"; exit 1; }
ADAPTER=$D/model/adapter
for sp in valid_unseen valid_seen; do until [ "$(grep -c . $D/eval_${sp#valid_}.jsonl 2>/dev/null)" -ge $([ $sp = valid_unseen ] && echo 134 || echo 140) ]; do
  guard 10000; $(S) $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $ADAPTER --split $sp --budget 50 --temperature 0 --out $D/eval_${sp#valid_}.jsonl --tag ${CH}_r$R >> $D/eval.log 2>&1 || sleep 30; done; done
echo "$ADAPTER" > $D/ADAPTER; echo "$ATTR" > $D/ATTR
log "CHAIN $CH R$R: train-game wins $NW/$((NW+NF)) | pointer $PSTAT | $(wc -l < $D/targets_$RULE/train.jsonl) targets ($RULE) | unseen $(grep -c '"won": true' $D/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen.jsonl)/140 (student 56/134, 80/140; env chain R1 167, R2 178)"
# 8. side product: distil the env-evidence labels (+ this round's divergence labels) into the log-reading attributor; log agreement only
if [ "$CH" = "selfattr" ] && [ "$R" != "1" ]; then
  $P scripts/divergence_labels.py --trajs $T reports/alfworld/student3b_low100_train_b50.jsonl reports/alfworld/self_evolve3b_rounds/env/r2/traj.jsonl reports/alfworld/self_evolve3b_rounds/env/r3/traj.jsonl --out_dir $D/divlab > $D/divlab.log 2>&1
  guard 20000; [ -f $D/attr/adapter/adapter_config.json ] || $(S) $P -u scripts/alfworld_self_attributor.py train --labels $D/envlab/labels.jsonl --trajs $T --init $PREV --epochs 1 --lr 1e-4 --out $D/attr > $D/attr.train.log 2>&1
  guard 10000; [ -s $D/attr_pointer.raw.jsonl ] || $(S) $P -u scripts/alfworld_self_attributor.py pointer --traj $T --ids $D/ids_fail.jsonl --adapter $D/attr/adapter --out $D/attr_pointer.jsonl > $D/attr.pointer.log 2>&1
  echo "$D/attr/adapter" > $D/ATTR
  $P - <<PY | tee -a $M
import json
env={json.loads(l)["id"]:json.loads(l)["t_star"] for l in open("$D/envlab/labels.jsonl")}; R=[json.loads(l) for l in open("$D/attr_pointer.raw.jsonl")]
print(f"=== chain $CH r$R: distilled attributor vs env-evidence labels: exact {sum(r['t_hat']==env.get(r['id']) for r in R)/len(R):.2f}, +-1 {sum(abs(r['t_hat']-env.get(r['id'],99))<=1 for r in R)/len(R):.2f}, t0 {sum(r['t_hat']==0 for r in R)/len(R):.2f}")
PY
fi
