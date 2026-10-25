#!/bin/bash
# User-approved (Oct 7 22:10) WebShop feasibility test of the FINAL recipe on the existing round-1 collection (600 sessions, SFT student):
# Model A pointer -> hindsight + diagnosis teacher scan (graded goal-progress evidence) -> 20% filter.  NO training.
source scripts/lib_baselines.sh; export WEBSHOP_SERVER=http://agd07:38300
W=reports/webshop; D=$W/final_r1; T=$W/round1/traj_train.jsonl; WSTU=$W/student100/final; MA=reports/attributor_no_observations/final; mkdir -p $D
[ -s $D/release.jsonl ] || $P scripts/webshop_final.py release --traj $T --out $D/release.jsonl --ids $D/ids_fail.jsonl --first 600 | tee -a $W/queue.log
until grep -q DONE $D/modelA.log 2>/dev/null; do $(S 12000) $P -u scripts/alfworld_attribute.py --attributor $MA --variant no_observations --release $D/release.jsonl --out $D/modelA_pred.jsonl >> $D/modelA.log 2>&1 || sleep 60; done
$P scripts/webshop_final.py pointer --traj $T --pred $D/modelA_pred.jsonl --out $D/pointer.jsonl | tee -a $W/queue.log
for i in 0 1 2; do ( until grep -q "DONE fscan" $D/scan_s$i.log 2>/dev/null; do $(S 12000) $P -u scripts/webshop_final.py scan --traj $T --pointer $D/pointer.jsonl --out $D/scan_s$i.jsonl --adapter $WSTU --shard $i 3 >> $D/scan_s$i.log 2>&1 || sleep 30; done ) & sleep 30; done; wait
$P scripts/webshop_final.py filter --scan_dir $D --ids $D/ids_fail.jsonl --traj $T --out $D/targets_k20.jsonl --frac 0.2 | tee -a $W/queue.log
echo "=== $(date '+%F %T') webshop final feasibility DONE" | tee -a $W/queue.log
