#!/bin/bash
# One round of a self-evolution CHAIN (pre-registered protocol, docs/self_evolve_proxy_filters.md §7).
#   self_evolve_round.sh <chain: env|rollout|placebo> <round r> <start adapter> <prev estimator adapter> <hold job id>
# Every chain, every round, identical protocol; only the action filter differs:
#   1. collect  : the round's student plays the same 1500 train games, T=1.0, budget 50 (same as the original b50 data)
#   2. estimator: refresh the M5 progress estimator on THIS student's own wins (continues from the previous estimator)
#   3. pointer  : M5 rule (first step whose next state is predicted farther from success, delta 1.0), failures of the first 1000 games
#   4. candidates: the student's own top-2 admissible commands at the pointer (plain prompt, no hindsight), a- excluded
#   5. filter   : env     = one-step env evidence, top-86 states by env_score (no rollout)
#                 rollout = student continuation <=20 steps, keep winners (verified KL, as the external-teacher arms)
#                 placebo = 86 random pointer states, a uniformly random admissible action != a-
#   6. train    : KL on the targets, lr 1e-5, 1 epoch, LoRA continued from the start adapter; 7. eval 274 games greedy.
set -u; CH=$1; R=$2; START=$3; EST_PREV=$4; H=$5
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
B=reports/alfworld/self_evolve3b_rounds; D=$B/$CH/r$R; mkdir -p $D/self $D/proxy; M=$B/queue.log; T=$D/traj.jsonl
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S="srun --jobid=$H --overlap -n1 -c4 --mem=0"
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
free_mb() { $S nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1; }
guard() { until [ "$(free_mb)" -ge "$1" ] 2>/dev/null; do sleep 60; done; }
log "chain $CH round $R: start"
# 1. collect (host RAM leak -> chunks; resume until DONE)
until tail -n 3 $D/collect.log 2>/dev/null | grep -q "^DONE"; do
  guard 10000; $S $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $START --split train --n_games 1500 --budget 50 --temperature 1.0 --out $T --tag ${CH}_r$R --max_episodes_per_run 450 >> $D/collect.log 2>&1 || sleep 60
done
NW=$(grep -c '"won": true' $T); NF=$(grep -c '"won": false' $T); log "chain $CH r$R: collected $(wc -l < $T) games, $NW won, $NF failed"
# 2. estimator refresh on this student's own wins; 3. pointer on failures of the first 1000 games
guard 20000; [ -f $D/m5/adapter/adapter_config.json ] || $S $P -u scripts/alfworld_progress_model.py train --traj $T --adapter $EST_PREV --n 4000 --out $D/m5 > $D/m5.log 2>&1
$P - <<PY
import json
eps=[json.loads(l) for l in open("$T")][:1000]
open("$D/ids_fail.jsonl","w").writelines(json.dumps({"id":e["id"]})+"\n" for e in eps if not e["won"])
PY
guard 10000; $S $P -u scripts/alfworld_progress_model.py score --traj $T --adapter $D/m5/adapter --ids_from $D/ids_fail.jsonl --out $D/m5/scores.jsonl >> $D/m5.log 2>&1
$P scripts/alfworld_progress_model.py pointer --scores $D/m5/scores.jsonl --out $D/m5/pointer.jsonl --delta 1.0 >> $D/m5.log 2>&1
$P - <<PY
import json
rows=[json.loads(l) for l in open("$D/m5/pointer.jsonl")]
open("$D/self/steps.jsonl","w").writelines(json.dumps({"id":r["id"],"steps":[r["t_hat"]],"mode":None})+"\n" for r in rows)
print("pointer states:", len(rows))
PY
# 4. candidates: plain student distribution at the pointer (hindsight_steps scores plain + hindsight; we keep plain)
export SE_DIR=$D SE_TRAJ=$T SE_MODEL=Qwen/Qwen2.5-3B-Instruct SE_ADAPTER=$PWD/$START
guard 10000; $S $P -u scripts/self_evolve.py hindsight_steps --steps $D/self/steps.jsonl > $D/cands.log 2>&1
$P - <<PY
import json
rows=[json.loads(l) for l in open("$D/self/targets.jsonl")]
with open("$D/self/targets.jsonl","w") as f:
    for r in rows: r["hind_logprobs"]=r["teacher_logprobs"]; r["teacher_logprobs"]=r["plain_logprobs"]; r["mode"]="plain"; f.write(json.dumps(r)+"\n")
print("plain targets:", len(rows))
PY
NS=$(wc -l < $D/self/steps.jsonl)
# 5.+6. filter and train
case $CH in
  env)
    until grep -q "DONE onestep" $D/onestep.log 2>/dev/null; do srun --jobid=2239109 --overlap -n1 -c2 --mem=20G $P -u scripts/proxy_eval_se3b.py onestep --arms self --limit 250 >> $D/onestep.log 2>&1; sleep 5; done
    $P scripts/proxy_arms_se3b.py --arm self --rule env_top --k 86 --out $D/targets_env | tee -a $M; DATA=$D/targets_env/train.jsonl ;;
  placebo)
    $P - <<PY
import json, random
rng=random.Random($R); tg=[json.loads(l) for l in open("$D/self/targets.jsonl")]; rng.shuffle(tg)
with open("$D/placebo_train.jsonl","w") as f:
    for T in tg[:86]:
        cmds=T["admissible"]; c=rng.choice([x for x in cmds if x!=T["a_minus"]])
        f.write(json.dumps({"kind":"kl","src":"$T","id":T["id"],"t":T["t"],"cmds":cmds,"target":[1.0 if x==c else 0.0 for x in cmds]})+"\n")
print("placebo targets: 86 random states, uniform random action")
PY
    DATA=$D/placebo_train.jsonl ;;
  rollout)
    until grep -q "DONE distill" $D/distill.log 2>/dev/null; do
      guard 24000; $S $P -u scripts/alfworld_attr_opd.py distill --traj $T --targets $D/self/targets.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter $START --out $D/model_rollout --verify_k 2 --exclude_a_minus --max_examples_per_run 250 >> $D/distill.log 2>&1 || sleep 30; done
    ADAPTER=$D/model_rollout/adapter; NT="$(wc -l < $D/model_rollout/examples.jsonl) verified" ;;
esac
if [ "$CH" != "rollout" ]; then
  guard 24000; $S $P -u scripts/alfworld_opd_plus.py train --data $DATA --student Qwen/Qwen2.5-3B-Instruct --adapter $PWD/$START --lr 1e-5 --epochs 1 --batch 8 --out $D/model > $D/train.log 2>&1 || { log "chain $CH r$R: TRAIN FAILED"; exit 1; }
  ADAPTER=$D/model/adapter; NT="$(wc -l < $DATA) targets"
fi
# 7. eval
guard 10000; for sp in valid_unseen valid_seen; do $S $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $ADAPTER --split $sp --budget 50 --temperature 0 --out $D/eval_${sp#valid_}.jsonl --tag ${CH}_r$R >> $D/eval.log 2>&1; done
echo "$ADAPTER" > $D/ADAPTER; echo "$D/m5/adapter" > $D/ESTIMATOR
log "CHAIN $CH R$R: train-game wins $NW/$((NW+NF)) | pointer states $NS (failures of first 1000 games) | $NT | unseen $(grep -c '"won": true' $D/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen.jsonl)/140 (student 56/134, 80/140)"
