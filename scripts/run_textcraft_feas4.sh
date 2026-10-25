#!/bin/bash
# TextCraft feasibility part 4: informed self-teacher candidates at Model A's pointed states (standard no_observations pointer, and the
# with-observations pointer) -> one-step goal-progress evidence.  50 failures each.
source scripts/lib_baselines.sh
T=reports/textcraft; STU=$T/student100/final
sc() { until grep -q "DONE scan" $T/modelA/scan_$1.log 2>/dev/null; do $(S 12000) $P -u scripts/textcraft_evidence.py scan --traj $T/eval_train300.jsonl --pointer $T/modelA/pointer_$2.jsonl --out $T/modelA/scan_$1.jsonl --adapter $STU >> $T/modelA/scan_$1.log 2>&1 || sleep 60; done; }
sc noobs "" & sleep 20
sc full full & wait
echo "=== $(date '+%F %T') TEXTCRAFT FEAS4 (scan) DONE" >> $T/queue.log
