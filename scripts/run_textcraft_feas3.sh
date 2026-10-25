#!/bin/bash
# TextCraft feasibility part 3: Model A with observations (variant full; out of its training distribution) on the same 50 failures.
source scripts/lib_baselines.sh
T=reports/textcraft; MA=reports/attributor_no_observations/final
until grep -q DONE $T/modelA/pred_full.log 2>/dev/null; do $(S 12000) $P -u scripts/alfworld_attribute.py --attributor $MA --variant full --release $T/modelA/release.jsonl --out $T/modelA/pred_full.jsonl >> $T/modelA/pred_full.log 2>&1 || sleep 60; done
echo "=== $(date '+%F %T') TEXTCRAFT FEAS3 (Model A full) DONE" >> $T/queue.log
