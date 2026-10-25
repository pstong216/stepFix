#!/bin/bash
# Side job for R1_v3: the v3 Model A configuration (--variant full, complete task statement as the query) on the same 50 feasibility
# failures that were hand-judged before (reports/textcraft/modelA/release_qcmd.jsonl already carries the crafting commands in its query).
source scripts/lib_baselines.sh
M=reports/textcraft/modelA; MA=reports/attributor_no_observations/final
until $P scripts/textcraft_round.py cover --glob $M/pred_full_qcmd.jsonl --key id --ids $M/release_qcmd.jsonl > /dev/null 2>&1; do $(S 16000) $P -u scripts/alfworld_attribute.py --attributor $MA --variant full --release $M/release_qcmd.jsonl --out $M/pred_full_qcmd.jsonl >> $M/pred_full_qcmd.log 2>&1 || sleep 60; done
echo "=== $(date '+%F %T') v3 Model A config on the 50 feasibility failures DONE" >> reports/textcraft/final/queue.log
