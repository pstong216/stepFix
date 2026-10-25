#!/bin/bash
# Second dataset (autonomous, Oct 7 23:10, user authorized paper-needed runs): WebShop with the FINAL recipe, except the step selector:
# the frozen 3B Model A collapses to step 0 on WebShop (95% of failures, with or without page text), so the step comes from ENVIRONMENT
# ADJUDICATION (the same rule as ALFWorld's STEP_POOL: first-step + stall-entry proposals, keep the one with the larger verified
# improvement). Teacher = hindsight + own analysis (no diagnosis), 6 samples -> top-2 candidates, ONE step each, graded goal-progress
# evidence (scripts/webshop_final.py ev_goal, v2 fixes), K = 20% of failures + 100 replay (shortest wins, 2 steps each), continued LoRA
# SFT (same as ALFWorld's one-hot), dev-200 eval (greedy, budget 15). R1 uses the existing 600-session collection of the SFT student;
# R2 collects the same 600 train sessions with the R1 model and runs only if R1 is not worse than the student on dev (86 / 0.598).
source scripts/lib_baselines.sh; export WEBSHOP_SERVER=http://agd07:38300
W=reports/webshop; WSTU=$W/student100/final; C=$W/final_adj; LOGW=$W/queue.log
wlog() { echo "=== $(date '+%F %T') $*" | tee -a $LOGW $SA/queue.log; }
devstat() { $P -c "import json; R=[json.loads(l) for f in '$1'.split() for l in open(f)]; print('dev %d/%d success, mean reward %.3f' % (sum(r['won'] for r in R), len(R), sum(r['final_score'] for r in R)/100/max(1,len(R))))"; }
wround() { local R=$1 START=$2 T=$3 D=$C/r$1; mkdir -p $D
  if [ "$R" != "1" ]; then until grep -q "DONE collect" $D/collect.log 2>/dev/null; do $(S 10000) $P -u scripts/webshop_agent.py collect --adapter $START --split train --n 600 --budget 15 --temperature 1.0 --out $D/traj_train.jsonl --tag wsf_r$R >> $D/collect.log 2>&1 || sleep 60; done; T=$D/traj_train.jsonl; fi
  $P -c "import json; open('$D/ids_fail.jsonl','w').writelines(json.dumps({'id':e['id']})+'\n' for e in map(json.loads, open('$T')) if not e['won'])"
  wlog "webshop final r$R: start, $(grep -c . $T) sessions, $(grep -c . $D/ids_fail.jsonl) failures (from $START)"
  [ -s $D/pointer_adj.jsonl ] || $P scripts/webshop_final.py pointer_adj --traj $T --ids $D/ids_fail.jsonl --out $D/pointer_adj.jsonl | tee -a $LOGW
  for i in 0 1 2; do ( until grep -q "DONE fscan" $D/scan_s$i.log 2>/dev/null; do $(S 12000) $P -u scripts/webshop_final.py scan --traj $T --pointer $D/pointer_adj.jsonl --out $D/scan_s$i.jsonl --adapter $START --shard $i 3 --no_diag >> $D/scan_s$i.log 2>&1 || sleep 30; done ) & sleep 30; done; wait
  [ -s $D/targets.jsonl ] || $P scripts/webshop_final.py filter_pool --scan_dir $D --ids $D/ids_fail.jsonl --traj $T --out $D/targets.jsonl --frac 0.2 | tee -a $LOGW $SA/queue.log
  [ -s $D/replay.jsonl ] || $P scripts/webshop_final.py replay --traj $T --out $D/replay.jsonl --n 100 --seed $R | tee -a $LOGW
  cat $D/targets.jsonl $D/replay.jsonl > $D/train_replay.jsonl
  until [ -f $D/model/adapter/adapter_config.json ]; do $(S 16000) $P -u scripts/webshop_evidence.py train --data $D/train_replay.jsonl --adapter $START --out $D/model > $D/train.log 2>&1 || { wlog "webshop final r$R: TRAIN FAILED (retry)"; sleep 120; }; done
  for i in 0 1; do ( until grep -q "DONE collect" $D/eval_s$i.log 2>/dev/null; do $(S 10000) $P -u scripts/webshop_agent.py collect --adapter $D/model/adapter --split dev --offset $((100 * i)) --n 100 --budget 15 --temperature 0 --out $D/eval_dev_s$i.jsonl --tag wsf_r${R}_s$i >> $D/eval_s$i.log 2>&1 || sleep 60; done ) & sleep 30; done; wait
  wlog "WEBSHOP FINAL R$R: $(grep -c . $D/targets.jsonl) corrections + 100 replay | $(devstat "$D/eval_dev_s0.jsonl $D/eval_dev_s1.jsonl") (student dev 86/200, 0.598)"; }
wround 1 $WSTU $W/round1/traj_train.jsonl
OK=$($P -c "import json; R=[json.loads(l) for f in ['$C/r1/eval_dev_s0.jsonl','$C/r1/eval_dev_s1.jsonl'] for l in open(f)]; s=sum(r['won'] for r in R); m=sum(r['final_score'] for r in R)/100/len(R); print(int(s >= 86 or m >= 0.598))")
if [ "$OK" = "1" ]; then wround 2 $C/r1/model/adapter ""; wlog "webshop final R1->R2 DONE"; else wlog "webshop final: R1 worse than the student on dev -> R2 not started"; fi
