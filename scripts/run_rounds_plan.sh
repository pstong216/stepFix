#!/bin/bash
# Multi-round self-evolution plan (Oct 2): three chains from the SAME SFT student, same protocol every round, only the action
# filter differs (env one-step top-86 | rollout verification | placebo).  Round 1 of the env chain = m5_plain/env_top (run by
# run_step_x_filter.sh); round 1 of the rollout and placebo chains are produced here on the same m5_plain states/candidates.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
SE=reports/alfworld/self_evolve3b; B=reports/alfworld/self_evolve3b_rounds; M=$B/queue.log; mkdir -p $B/rollout/r1 $B/placebo/r1 $B/env/r1
STU=reports/alfworld/student3b_low100/final; EST1=reports/alfworld/progress_model/model/adapter; T1=reports/alfworld/student3b_low100_train_b50.jsonl
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HA=2644179; HB=2644180; SA="srun --jobid=$HA --overlap -n1 -c4 --mem=0"; SB="srun --jobid=$HB --overlap -n1 -c4 --mem=0"
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
free_mb() { $1 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1; }
guard() { until [ "$(free_mb "$1")" -ge "$2" ] 2>/dev/null; do sleep 60; done; }
evalit() { local S=$1 AD=$2 D=$3 TAG=$4; guard "$S" 10000
  for sp in valid_unseen valid_seen; do $S $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $AD --split $sp --budget 50 --temperature 0 --out $D/eval_${sp#valid_}.jsonl --tag $TAG >> $D/eval.log 2>&1; done
  echo "unseen $(grep -c '"won": true' $D/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen.jsonl)/140"; }
# ---------------- hold A: rollout chain R1 (verified KL on the m5_plain candidates) then R2
( until grep -q "YIELD ARMS DONE" $SE/queue.log 2>/dev/null; do sleep 120; done
  D=$B/rollout/r1; log "chain rollout R1: verify the m5_plain candidates from the SFT student"
  until grep -q "DONE distill" $D/distill.log 2>/dev/null; do
    guard "$SA" 24000; $SA $P -u scripts/alfworld_attr_opd.py distill --traj $T1 --targets $SE/m5_plain/targets.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter $STU --out $D/model_rollout --verify_k 2 --exclude_a_minus --max_examples_per_run 250 >> $D/distill.log 2>&1 || sleep 30; done
  R=$(evalit "$SA" $D/model_rollout/adapter $D rollout_r1); echo "$D/model_rollout/adapter" > $D/ADAPTER; echo "$EST1" > $D/ESTIMATOR
  log "CHAIN rollout R1: $(wc -l < $D/model_rollout/examples.jsonl) verified / 319 | $R (student 56/134, 80/140)"
  bash scripts/self_evolve_round.sh rollout 2 $(cat $D/ADAPTER) $EST1 $HA ) &
# ---------------- hold B: placebo chain R1, then env chain R2 (from m5_plain/env_top R1), then placebo R2
( until grep -q "STEP X FILTER ARMS DONE" $SE/queue.log 2>/dev/null; do sleep 120; done
  D=$B/placebo/r1; log "chain placebo R1: 86 random m5_plain states, uniform random action"
  $P - <<PY
import json, random
rng=random.Random(1); tg=[json.loads(l) for l in open("$SE/m5_plain/targets.jsonl")]; rng.shuffle(tg)
with open("$D/placebo_train.jsonl","w") as f:
    for T in tg[:86]:
        cmds=T["admissible"]; c=rng.choice([x for x in cmds if x!=T["a_minus"]])
        f.write(json.dumps({"kind":"kl","src":"$T1","id":T["id"],"t":T["t"],"cmds":cmds,"target":[1.0 if x==c else 0.0 for x in cmds]})+"\n")
PY
  guard "$SB" 24000; $SB $P -u scripts/alfworld_opd_plus.py train --data $D/placebo_train.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter $PWD/$STU --lr 1e-5 --epochs 1 --batch 8 --out $D/model > $D/train.log 2>&1
  R=$(evalit "$SB" $D/model/adapter $D placebo_r1); echo "$D/model/adapter" > $D/ADAPTER; echo "$EST1" > $D/ESTIMATOR
  log "CHAIN placebo R1: 86 targets | $R (student 56/134, 80/140)"
  # env chain R1 = m5_plain/env_top produced by run_step_x_filter.sh
  E1=$SE/proxy_arms/m5_plain_env_top; D=$B/env/r1; cp $E1/eval_unseen.jsonl $E1/eval_seen.jsonl $D/ 2>/dev/null; echo "$E1/model/adapter" > $D/ADAPTER; echo "$EST1" > $D/ESTIMATOR
  log "CHAIN env R1 = m5_plain/env_top: $(grep 'm5_plain/env_top' $SE/queue.log | tail -n1 | grep -o 'unseen.*140')"
  bash scripts/self_evolve_round.sh env 2 $E1/model/adapter $EST1 $HB
  bash scripts/self_evolve_round.sh placebo 2 $(cat $B/placebo/r1/ADAPTER) $EST1 $HB ) &
wait; log "ROUNDS PLAN (R1 + R2 for env/rollout/placebo) DONE"
