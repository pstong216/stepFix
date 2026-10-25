#!/bin/bash
# Relaunched Oct 8 ~17:25 (user: "从R0开始就把一共15个limit，当前是第几步了，记得买东西 加上"), on the seeded-goal server (identical goals for every worker / restart).
# WebShop with student prompt v2 (Oct 8, user: tell the student to buy before the budget runs out; budget 15 everywhere).
# R0 = Qwen2.5-3B + LoRA SFT on the same 100 planner demos with prompt v2; R1-R3 = FINAL recipe (frozen 3B Model A step + diagnosis,
# no_observations input as on ALFWorld; hindsight self-teacher, 6 samples -> top-2; one-step v2 goal-progress evidence; top 20% + 100
# replay; continued LoRA). Each round collects 600 train sessions (700-1299) with the current model; every model is evaluated on
# test 500 + dev 200 (greedy, budget 15).
export WEBSHOP_PROMPT=v2 WEBSHOP_BUDGET=15
source scripts/lib_baselines.sh; export WEBSHOP_SERVER=${WEBSHOP_SERVER:-http://aga34:38300}   # Oct 8 17:20: seeded-goal server (scripts/webshop_server.py GOAL_SEED=0) on hold 3080458
W=reports/webshop; C=$W/final_p2; STU=$C/r0/model/final; MA=reports/attributor_no_observations/final; mkdir -p $C/r0
wlog() { echo "=== $(date '+%F %T') $*" | tee -a $W/queue.log $SA/queue.log; }
ev700() { local D=$1 AD=$2 TG=$3
  for spec in "test 0" "test 100" "test 200" "test 300" "test 400" "dev 0" "dev 100"; do set -- $spec; o=$D/eval_${1}_$2.jsonl
    ( until grep -q "DONE collect" $o.log 2>/dev/null; do $(S 10000) $P -u scripts/webshop_agent.py collect --adapter $AD --split $1 --offset $2 --n 100 --budget 15 --temperature 0 --out $o --tag ${TG}_$1$2 >> $o.log 2>&1 || sleep 60; done ) & sleep 20; done; wait
  wlog "WEBSHOP P2 $TG: $($P -c "import json,glob; R=[json.loads(l) for f in sorted(glob.glob('$D/eval_*_*.jsonl')) for l in open(f)]; n=len(R); s=sum(r['won'] for r in R); print('700-eval %d/%d success (%.1f%%), mean reward %.3f, no purchase %d' % (s, n, 100*s/max(1,n), sum(r['final_score'] for r in R)/100/max(1,n), sum(not any(st['action']=='click[buy now]' for st in r['steps']) for r in R)))")"; }
# R0: SFT with prompt v2 + eval
until [ -f $STU/adapter_config.json ]; do $(S 20000) $P -u scripts/webshop_agent.py sft --n_demos 100 --out $C/r0/model > $C/r0/sft.log 2>&1 || sleep 120; done
[ -s $C/r0/eval.done ] || { ev700 $C/r0 $STU p2_r0; touch $C/r0/eval.done; }
wround() { local R=$1 START=$2 D=$C/r$1; mkdir -p $D; local T=$D/traj_train.jsonl
  until grep -q "DONE collect" $D/collect.log 2>/dev/null; do $(S 10000) $P -u scripts/webshop_agent.py collect --adapter $START --split train --n 600 --budget 15 --temperature 1.0 --out $T --tag p2_r$R >> $D/collect.log 2>&1 || sleep 60; done
  [ -s $D/release.jsonl ] || $P scripts/webshop_final.py release --traj $T --out $D/release.jsonl --ids $D/ids_fail.jsonl --first 600 | tee -a $W/queue.log
  until grep -q DONE $D/modelA.log 2>/dev/null; do $(S 12000) $P -u scripts/alfworld_attribute.py --attributor $MA --variant no_observations --release $D/release.jsonl --out $D/modelA_pred.jsonl >> $D/modelA.log 2>&1 || sleep 60; done
  [ -s $D/pointer.jsonl ] || $P scripts/webshop_final.py pointer --traj $T --pred $D/modelA_pred.jsonl --out $D/pointer.jsonl | tee -a $W/queue.log
  for i in 0 1 2; do ( until grep -q "DONE fscan" $D/scan_s$i.log 2>/dev/null; do $(S 12000) $P -u scripts/webshop_final.py scan --traj $T --pointer $D/pointer.jsonl --out $D/scan_s$i.jsonl --adapter $START --shard $i 3 >> $D/scan_s$i.log 2>&1 || sleep 30; done ) & sleep 30; done; wait
  [ -s $D/targets.jsonl ] || $P scripts/webshop_final.py filter --scan_dir $D --ids $D/ids_fail.jsonl --traj $T --out $D/targets.jsonl --frac 0.2 | tee -a $W/queue.log $SA/queue.log
  [ -s $D/replay.jsonl ] || $P scripts/webshop_final.py replay --traj $T --out $D/replay.jsonl --n 100 --seed $R | tee -a $W/queue.log
  cat $D/targets.jsonl $D/replay.jsonl > $D/train_replay.jsonl
  until [ -f $D/model/adapter/adapter_config.json ]; do $(S 16000) $P -u scripts/webshop_evidence.py train --data $D/train_replay.jsonl --adapter $START --out $D/model > $D/train.log 2>&1 || sleep 120; done
  [ -s $D/eval.done ] || { ev700 $D $D/model/adapter p2_r$R; touch $D/eval.done; }; }
wround 1 $STU
wround 2 $C/r1/model/adapter
wround 3 $C/r2/model/adapter
wlog "webshop P2 R0->R3 DONE"
