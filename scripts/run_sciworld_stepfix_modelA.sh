#!/bin/bash
# Model A (frozen 3B, no_observations) on a round's failures, 3 shards -> D/modelA_pred_s*.jsonl -> pointer
source scripts/lib_sciworld_stepfix.sh; D=$1
for i in 0 1 2; do ( until grep -q "^DONE" $D/modelA_s$i.log 2>/dev/null; do wait_free "--out $D/modelA_pred_s$i.jsonl"; $(SG $GMEM) $P -u scripts/alfworld_attribute.py --attributor reports/attributor_no_observations/final --variant no_observations --release $D/rel_s$i.jsonl --out $D/modelA_pred_s$i.jsonl >> $D/modelA_s$i.log 2>&1 || sleep 30; done ) & sleep 25; done; wait
$P scripts/sciworld_stepfix.py pointer --dir $D | tee -a $SF/queue.log
