#!/bin/bash
# WebShop R1 -> R2, FINAL recipe with Model A step selection, after fixing the release-format bug (Oct 8 01:50: step_number was a string,
# so Model A saw an EMPTY transcript and always answered step 1). Model A input = no_observations, exactly as on ALFWorld.
# Model A step + diagnosis -> hindsight self-teacher (6 samples -> top-2) -> one step each, v2 goal-progress evidence -> top 20% + 100
# replay -> continued LoRA -> dev 200 (greedy, budget 15). R2: recollect 600 train sessions with the R1 model, same pipeline.
source scripts/lib_baselines.sh; export WEBSHOP_SERVER=http://agd07:38300
W=reports/webshop; WSTU=$W/student100/final; C=$W/final_modelA; MA=reports/attributor_no_observations/final
wlog() { echo "=== $(date '+%F %T') $*" | tee -a $W/queue.log $SA/queue.log; }
wround() { local R=$1 START=$2 T=$3 D=$C/r$1; mkdir -p $D
  if [ "$R" != "1" ]; then until grep -q "DONE collect" $D/collect.log 2>/dev/null; do $(S 10000) $P -u scripts/webshop_agent.py collect --adapter $START --split train --n 600 --budget 15 --temperature 1.0 --out $D/traj_train.jsonl --tag wsA_r$R >> $D/collect.log 2>&1 || sleep 60; done; T=$D/traj_train.jsonl; fi
  [ -s $D/release.jsonl ] || $P scripts/webshop_final.py release --traj $T --out $D/release.jsonl --ids $D/ids_fail.jsonl --first 600 | tee -a $W/queue.log
  until grep -q DONE $D/modelA.log 2>/dev/null; do $(S 12000) $P -u scripts/alfworld_attribute.py --attributor $MA --variant no_observations --release $D/release.jsonl --out $D/modelA_pred.jsonl >> $D/modelA.log 2>&1 || sleep 60; done
  [ -s $D/pointer.jsonl ] || $P scripts/webshop_final.py pointer --traj $T --pred $D/modelA_pred.jsonl --out $D/pointer.jsonl | tee -a $W/queue.log
  for i in 0 1 2; do ( until grep -q "DONE fscan" $D/scan_s$i.log 2>/dev/null; do $(S 12000) $P -u scripts/webshop_final.py scan --traj $T --pointer $D/pointer.jsonl --out $D/scan_s$i.jsonl --adapter $START --shard $i 3 >> $D/scan_s$i.log 2>&1 || sleep 30; done ) & sleep 30; done; wait
  [ -s $D/targets.jsonl ] || $P scripts/webshop_final.py filter --scan_dir $D --ids $D/ids_fail.jsonl --traj $T --out $D/targets.jsonl --frac 0.2 | tee -a $W/queue.log $SA/queue.log
  [ -s $D/replay.jsonl ] || $P scripts/webshop_final.py replay --traj $T --out $D/replay.jsonl --n 100 --seed $R | tee -a $W/queue.log
  cat $D/targets.jsonl $D/replay.jsonl > $D/train_replay.jsonl
  until [ -f $D/model/adapter/adapter_config.json ]; do $(S 16000) $P -u scripts/webshop_evidence.py train --data $D/train_replay.jsonl --adapter $START --out $D/model > $D/train.log 2>&1 || sleep 120; done
  for i in 0 1; do ( until grep -q "DONE collect" $D/eval_s$i.log 2>/dev/null; do $(S 10000) $P -u scripts/webshop_agent.py collect --adapter $D/model/adapter --split dev --offset $((100 * i)) --n 100 --budget 15 --temperature 0 --out $D/eval_dev_s$i.jsonl --tag wsAf_r${R}_s$i >> $D/eval_s$i.log 2>&1 || sleep 60; done ) & sleep 30; done; wait
  wlog "WEBSHOP MODEL-A (fixed) R$R: $(grep -c . $D/targets.jsonl) corrections + 100 replay | $($P -c "import json; R=[json.loads(l) for f in ['$D/eval_dev_s0.jsonl','$D/eval_dev_s1.jsonl'] for l in open(f)]; print('dev %d/%d success, mean reward %.3f' % (sum(r['won'] for r in R), len(R), sum(r['final_score'] for r in R)/100/len(R)))") (student 86/200, 0.598)"; }
wround 1 $WSTU $W/round1/traj_train.jsonl
wround 2 $C/r1/model/adapter ""
wlog "webshop Model-A (fixed) R1->R2 DONE"
