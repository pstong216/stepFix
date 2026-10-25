#!/bin/bash
# WebShop round 1 (mirrors sciworld_round.sh): student = LoRA SFT on 100 oracle demos; collect train sessions at T=1 ->
# failures -> env-evidence scan (first 10 steps, 2 candidates each, one-step replay on the env server) -> labels (step-back not
# needed: labels come from the scan) -> filters {env_top | conf_top | placebo | all} at K -> continued LoRA -> dev eval (200
# sessions, greedy, budget 15): success (reward 1) + mean reward.   webshop_round.sh [K]
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; O=reports/webshop; R=$O/round1; mkdir -p $R; M=$O/queue.log
K=${1:-100}; STU=$O/student100/final; export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WEBSHOP_SERVER=http://acn216:38300
log() { echo "=== $(date '+%F %T') $*" | tee -a $M reports/alfworld/self_evolve3b/queue.log; }
pick() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge ${1:-10000} ] 2>/dev/null && echo $best; }
S() { local H=""; until H=$(pick $1); [ -n "$H" ]; do sleep 60; done; echo "srun --jobid=$H --overlap -n1 -c4 --mem=0"; }
until grep -q "DONE demos" $O/demos.log 2>/dev/null; do sleep 60; done
log "webshop demos: $(tail -n1 $O/demos.log) | oracle on test: $(tail -n1 $O/oracle_test.log 2>/dev/null)"
# 0. student SFT on 100 oracle demos + greedy dev eval
until [ -f $STU/adapter_config.json ]; do $(S 20000) $P -u scripts/webshop_agent.py sft --n_demos 100 --out $O/student100 > $O/sft_student100.log 2>&1 || { log "webshop sft FAILED"; sleep 120; }; done
until grep -q "DONE collect" $R/eval_student.log 2>/dev/null; do $(S 10000) $P -u scripts/webshop_agent.py collect --adapter $STU --split dev --budget 15 --temperature 0 --out $R/eval_dev_student.jsonl --tag ws_student >> $R/eval_student.log 2>&1 || sleep 60; done
log "WEBSHOP student100: dev $(grep DONE $R/eval_student.log | tail -n1 | sed 's/DONE collect //')"
# 1. collect on 600 train sessions at T=1.0
T=$R/traj_train.jsonl
until grep -q "DONE collect" $R/collect.log 2>/dev/null; do $(S 10000) $P -u scripts/webshop_agent.py collect --adapter $STU --split train --n 600 --budget 15 --temperature 1.0 --out $T --tag ws_r1 >> $R/collect.log 2>&1 || sleep 60; done
log "webshop r1 collect: $(grep DONE $R/collect.log | tail -n1)"
# 2. env-evidence scan of the first 10 steps of every failure (2 shards)
for i in 0 1; do ( until grep -q "DONE scan" $R/scan_s$i.log 2>/dev/null; do $(S 12000) $P -u scripts/webshop_evidence.py scan --traj $T --out_dir $R --k 10 --n_samples 6 --adapter $STU --shard $i 2 >> $R/scan_s$i.log 2>&1 || sleep 60; done ) & sleep 90; done; wait
$P scripts/webshop_evidence.py rescore --traj $T --out_dir $R 2>&1 | tail -n 1 | tee -a $M
$P scripts/webshop_evidence.py label --out_dir $R 2>&1 | tee -a $M reports/alfworld/self_evolve3b/queue.log
# replay of the student's own wins (final recipe: K corrections + K replay steps), same target format
$P - <<PYR | tee -a $M
import json, random; rng=random.Random(1); T="$T"; K=$K
wins=[e for e in map(json.loads, open(T)) if e["won"]]; states=[(e["id"],t,e["steps"][t]["action"]) for e in wins for t in range(len(e["steps"]))]; rng.shuffle(states)
open("$R/replay.jsonl","w").writelines(json.dumps({"src":T,"id":i,"t":t,"action":a,"rank_score":0.0})+"\\n" for i,t,a in states[:K])
print(f"=== webshop replay: {min(K,len(states))} steps from {len(wins)} winning train sessions")
PYR
# 3. filters at t*, same K, train, dev eval
for rule in env_top conf_top placebo all; do D=$R/arm_$rule; mkdir -p $D
  $P scripts/webshop_evidence.py filter --out_dir $R --steps $R/steps.jsonl --traj $T --rule $rule --k $K --out $D/train.jsonl | tee -a $M
  cat $D/train.jsonl $R/replay.jsonl > $D/train_replay.jsonl
  until [ -f $D/model/adapter/adapter_config.json ]; do $(S 16000) $P -u scripts/webshop_evidence.py train --data $D/train_replay.jsonl --adapter $STU --out $D/model > $D/train.log 2>&1 || { log "webshop arm $rule: TRAIN FAILED"; sleep 120; }; done
  until grep -q "DONE collect" $D/eval.log 2>/dev/null; do $(S 10000) $P -u scripts/webshop_agent.py collect --adapter $D/model/adapter --split dev --budget 15 --temperature 0 --out $D/eval_dev.jsonl --tag ws_$rule >> $D/eval.log 2>&1 || sleep 60; done
  log "WEBSHOP r1 arm $rule: $(wc -l < $D/train.jsonl) targets | dev $(grep DONE $D/eval.log | tail -n1 | sed 's/DONE collect //')"
done
log "WEBSHOP ROUND 1 DONE"
