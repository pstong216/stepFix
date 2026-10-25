#!/bin/bash
# Self-evolving local correction from an UNTRAINED Qwen2.5-7B-Instruct (student = teacher, teacher sees the full failed log).
# Usage: run_self_evolve7b.sh <gpu job 1> <gpu job 2>   (collection already done by run_self_evolve7b_collect)
J1=$1; J2=$2; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S1="srun --jobid=$J1 --overlap -n1 -c4 --mem=0"; S2="srun --jobid=$J2 --overlap -n1 -c4 --mem=0"; SC="srun --jobid=2239109 --overlap -n1 -c2 --mem=24G"
O=reports/alfworld/self_evolve7b; M=$O/queue.log; T=$O/r1_train.jsonl
export SE_DIR=$O SE_TRAJ=$T SE_MODEL=Qwen/Qwen2.5-7B-Instruct SE_ADAPTER= SE_ATTR=$O/attr7b.jsonl
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
# failures -> 60 ids, their release rows for Model A
$P - <<'PY'
import json, random
O = "reports/alfworld/self_evolve7b"; E = [json.loads(l) for l in open(f"{O}/r1_train.jsonl")]; F = [e for e in E if not e["won"]]
random.Random(0).shuffle(F); ids = {e["id"] for e in F[:60]}
open(f"{O}/ids60.jsonl", "w").writelines(json.dumps({"id": e["id"]}) + "\n" for e in F[:60])
rel = [l for l in open(f"{O}/r1_train.release.jsonl") if json.loads(l)["id"] in ids]; open(f"{O}/fail60.release.jsonl", "w").writelines(rel)
print(f"{len(E)} episodes, {len(F)} failures ({len(F) / len(E):.0%}); chose 60; release rows {len(rel)}")
PY
log "Model A pointers + hindsight scoring"
$S1 $P -u scripts/alfworld_attribute.py --attributor reports/attributor_7b_no_observations/final --variant no_observations --release $O/fail60.release.jsonl --out $O/attr7b.jsonl > $O/attr.log 2>&1 &
head -30 $O/ids60.jsonl > $O/ids_a.jsonl; tail -30 $O/ids60.jsonl > $O/ids_b.jsonl
$S1 $P -u scripts/self_evolve.py hindsight --ids $O/ids_a.jsonl --out hindsight.jsonl > $O/hindsight.log 2>&1 &
$S2 $P -u scripts/self_evolve.py hindsight --ids $O/ids_b.jsonl --out hindsight_b.jsonl > $O/hindsight_b.log 2>&1 &
# self-trained attributor (A100): progress estimator trained only on the 7B's OWN winning trajectories -> pointer on the 60 failures
SA="srun --jobid=2433165 --overlap -n1 -c4 --mem=0"
( $SA $P -u scripts/alfworld_progress_model.py train --traj $T --student Qwen/Qwen2.5-7B-Instruct --adapter "" --n 2000 --out $O/selfprog_model > $O/selfprog.log 2>&1 \
  && $SA $P -u scripts/alfworld_progress_model.py score --traj $T --student Qwen/Qwen2.5-7B-Instruct --adapter $O/selfprog_model/adapter --ids_from $O/ids60.jsonl --out $O/selfprog_scores.jsonl >> $O/selfprog.log 2>&1 \
  && $P scripts/alfworld_progress_model.py pointer --scores $O/selfprog_scores.jsonl --out $O/selfprog_pointer.jsonl --delta 1.0 >> $O/selfprog.log 2>&1 \
  || echo "=== selfprog FAILED (arm skipped)" >> $M ) &
wait
$S1 $P -u scripts/self_evolve.py selfattr --ids $O/ids60.jsonl > $O/selfattr.log 2>&1 || log "selfattr FAILED (arm skipped)"
log "hindsight done: $(cat $O/hindsight*.jsonl | wc -l)/60; Model A rows $(wc -l < $O/attr7b.jsonl)"
$P scripts/self_evolve.py choose >> $M 2>&1 && $P scripts/self_evolve.py targets >> $M 2>&1 || { log "choose/targets FAILED"; exit 1; }
ARMS="rand_early entropy attr hkl"; [ -s $O/selfattr.jsonl ] && ARMS="$ARMS selfattr"; [ -s $O/selfprog_pointer.jsonl ] && ARMS="$ARMS selfprog"
for arm in $ARMS; do
  until grep -q "DONE onestep" $O/$arm/onestep.log 2>/dev/null; do $SC $P -u scripts/env_evidence_filter.py --teacher_dir $O/$arm --traj $T onestep >> $O/$arm/onestep.log 2>&1; sleep 3; done
done
$P scripts/self_evolve.py select >> $M 2>&1 || { log "select FAILED"; exit 1; }
log "training + eval"
i=0
for arm in $ARMS; do
  S=$S1; [ $((i % 2)) -eq 1 ] && S=$S2; i=$((i + 1))
  ( $S $P -u scripts/alfworld_opd_plus.py train --data $O/$arm/train_proxy.jsonl --student Qwen/Qwen2.5-7B-Instruct --adapter "" --lr 1e-4 --epochs 3 --out $O/$arm/model > $O/$arm/train.log 2>&1 \
    && $S $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-7B-Instruct --adapter $O/$arm/model/adapter --split valid_unseen --budget 50 --temperature 0 --out $O/$arm/eval_unseen.jsonl --tag se7b_$arm >> $O/$arm/train.log 2>&1 ;
    echo "=== $(date '+%F %T') $arm: trained on $(wc -l < $O/$arm/train_proxy.jsonl) proxy-selected corrections, unseen $(grep -c '"won": true' $O/$arm/eval_unseen.jsonl)/$(wc -l < $O/$arm/eval_unseen.jsonl) (7B zero-shot 60/134)" >> $M ) &
  [ $((i % 2)) -eq 0 ] && wait
done
wait; log "SELF-EVOLVE 7B ROUND 1 DONE"
