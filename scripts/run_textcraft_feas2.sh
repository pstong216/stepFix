#!/bin/bash
# TextCraft feasibility part 2: Model A on 50 greedy train failures (standard no_observations input; variant with the crafting commands in
# the task text) + a T=1.0 collection on 300 train-pool tasks (offsets 378-677 = distractor seed 1, no demo task) to size R1.
source scripts/lib_baselines.sh
T=reports/textcraft; STU=$T/student100; MA=reports/attributor_no_observations/final
tlog() { echo "=== $(date '+%F %T') $*" | tee -a $T/queue.log; }
ma() { until grep -q DONE $T/modelA/$2.log 2>/dev/null; do $(S 12000) $P -u scripts/alfworld_attribute.py --attributor $MA --variant no_observations --release $T/modelA/$1 --out $T/modelA/$2.jsonl >> $T/modelA/$2.log 2>&1 || sleep 60; done; }
co() { local O=$T/collect_t1_s$1.jsonl; until grep -q "DONE collect" ${O%.jsonl}.log 2>/dev/null; do $(S 10000) $P -u scripts/textcraft_agent.py collect --adapter $STU/final --split train --offset $2 --n 150 --temperature 1.0 --out $O --tag stu100t1 >> ${O%.jsonl}.log 2>&1 || sleep 60; done; }
ma release.jsonl pred & sleep 20
ma release_qcmd.jsonl pred_qcmd & sleep 20
co 0 378 & sleep 20
co 1 528 & wait
cat $T/collect_t1_s0.jsonl $T/collect_t1_s1.jsonl > $T/collect_t1_300.jsonl
$P -u scripts/textcraft_agent.py report $T/collect_t1_300.jsonl | tee -a $T/queue.log
tlog "TEXTCRAFT FEAS2 DONE"
