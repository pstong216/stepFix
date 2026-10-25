#!/bin/bash
# WebShop R1 with the FINAL recipe and the SAME step selector as ALFWorld (user Oct 8 00:30: "cannot change the step selector"):
# frozen 3B Model A step + diagnosis -> hindsight self-teacher scan (reports/webshop/final_r1/scan_s*.jsonl) re-scored offline with the
# v2 goal-progress evidence (banner fix, option click = sat, graded results-page loc) -> top 20% (81) + 100 replay -> train -> dev 200.
source scripts/lib_baselines.sh; export WEBSHOP_SERVER=http://agd07:38300
W=reports/webshop; D=$W/final_r1/v2; T=$W/round1/traj_train.jsonl; WSTU=$W/student100/final
[ -s $D/replay.jsonl ] || $P scripts/webshop_final.py replay --traj $T --out $D/replay.jsonl --n 100 --seed 1
cat $D/targets.jsonl $D/replay.jsonl > $D/train_replay.jsonl
until [ -f $D/model/adapter/adapter_config.json ]; do $(S 16000) $P -u scripts/webshop_evidence.py train --data $D/train_replay.jsonl --adapter $WSTU --out $D/model > $D/train.log 2>&1 || sleep 120; done
for i in 0 1; do ( until grep -q "DONE collect" $D/eval_s$i.log 2>/dev/null; do $(S 10000) $P -u scripts/webshop_agent.py collect --adapter $D/model/adapter --split dev --offset $((100 * i)) --n 100 --budget 15 --temperature 0 --out $D/eval_dev_s$i.jsonl --tag wsA_r1_s$i >> $D/eval_s$i.log 2>&1 || sleep 60; done ) & sleep 30; done; wait
R=$($P -c "import json; R=[json.loads(l) for f in ['$D/eval_dev_s0.jsonl','$D/eval_dev_s1.jsonl'] for l in open(f)]; print('dev %d/%d success, mean reward %.3f' % (sum(r['won'] for r in R), len(R), sum(r['final_score'] for r in R)/100/len(R)))")
echo "=== $(date '+%F %T') WEBSHOP MODEL-A R1: 81 corrections + 100 replay | $R (student 86/200, 0.598)" | tee -a $W/queue.log $SA/queue.log
