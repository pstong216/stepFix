#!/bin/bash
# ScienceWorld round 1 of the method + main-table-3 baselines (generic evidence; no task parsing):
#   student (3 gold demos / task) replays TRAIN variations at T=1.0 -> failures -> env-evidence attribution (scan first 8 steps)
#   -> candidates at t* (policy samples) -> one-step generic evidence -> filters {env_top | conf_top | placebo | all} (same K)
#   -> LoRA continued from the student -> greedy dev eval (all 7 variations x 30 tasks = 210 episodes, budget 50): mean score + success.
#   sciworld_round.sh [K]
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; O=reports/sciworld; R=$O/round1; mkdir -p $R; M=$O/queue.log
K=${1:-50}; STU=$O/student3/final; export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "=== $(date '+%F %T') $*" | tee -a $M reports/alfworld/self_evolve3b/queue.log; }
pick() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge "${1:-10000}" ] 2>/dev/null && echo $best; }
S() { local H=""; until H=$(pick $1); [ -n "$H" ]; do sleep 60; done; echo "srun --jobid=$H --overlap -n1 -c4 --mem=0"; }
until [ -f $STU/adapter_config.json ]; do sleep 120; done
# 1. collect on train variations (14 per task; 10 per task x 30 tasks = 300 episodes), T=1.0, budget 50
T=$R/traj_train.jsonl
until grep -q "DONE collect" $R/collect.log 2>/dev/null; do $(S 10000) $P -u scripts/sciworld_agent.py collect --adapter $STU --split train --per_task 10 --budget 50 --temperature 1.0 --out $T --tag sw_r1 >> $R/collect.log 2>&1 || sleep 60; done
log "sciworld r1: $(grep DONE $R/collect.log | tail -n1) | mean final score $($P -c "import json;r=[json.loads(l) for l in open('$T')];print(round(sum(e['final_score'] for e in r)/len(r),1))")"
# 2. env-evidence scan of the first 8 steps of every failure (GPU sampling + env replay)
rm -f $R/scan.log
until grep -q "DONE scan" $R/scan.log 2>/dev/null; do $(S 12000) $P -u scripts/sciworld_evidence.py scan --traj $T --out_dir $R --k 50 --n_samples 6 --adapter $STU >> $R/scan.log 2>&1 || sleep 60; done
$P scripts/sciworld_evidence.py label --out_dir $R 2>&1 | tee -a $M reports/alfworld/self_evolve3b/queue.log
# 3. filters at t*, same K, train, eval
for rule in env_top conf_top placebo all; do D=$R/arm_$rule; mkdir -p $D
  $P scripts/sciworld_evidence.py filter --out_dir $R --steps $R/steps.jsonl --traj $T --rule $rule --k $K --out $D/train.jsonl | tee -a $M
  until [ -f $D/model/adapter/adapter_config.json ]; do $(S 16000) $P -u scripts/sciworld_evidence.py train --data $D/train.jsonl --adapter $STU --out $D/model > $D/train.log 2>&1 || { log "sciworld arm $rule: TRAIN FAILED"; sleep 120; }; done
  until grep -q "DONE collect" $D/eval.log 2>/dev/null; do $(S 10000) $P -u scripts/sciworld_agent.py collect --adapter $D/model/adapter --split dev --per_task 7 --budget 50 --temperature 0 --out $D/eval_dev.jsonl --tag sw_$rule >> $D/eval.log 2>&1 || sleep 60; done
  log "SCIWORLD r1 arm $rule: $(wc -l < $D/train.jsonl) targets | dev $(grep DONE $D/eval.log | tail -n1 | sed 's/DONE collect //') success | mean score $($P -c "import json;r=[json.loads(l) for l in open('$D/eval_dev.jsonl')];print(round(sum(e['final_score'] for e in r)/len(r),1))")"
done
# student reference on the same dev protocol (budget 50)
until grep -q "DONE collect" $R/eval_student.log 2>/dev/null; do $(S 10000) $P -u scripts/sciworld_agent.py collect --adapter $STU --split dev --per_task 7 --budget 50 --temperature 0 --out $R/eval_dev_student.jsonl --tag sw_student >> $R/eval_student.log 2>&1 || sleep 60; done
log "SCIWORLD r1 student: dev $(grep DONE $R/eval_student.log | tail -n1 | sed 's/DONE collect //') success | mean score $($P -c "import json;r=[json.loads(l) for l in open('$R/eval_dev_student.jsonl')];print(round(sum(e['final_score'] for e in r)/len(r),1))")"
log "SCIWORLD ROUND 1 DONE"
