#!/bin/bash
# plan3 R0: greedy 810-task TEST eval of the plan3 SFT student (same protocol as plan2 R0); report vs plan2 R0 (446) and old R0 (454).
export HOLD_NAME_RE='^(tcded_gpu|error_attr_hold_.*)$'
source scripts/lib_baselines.sh; set -o pipefail
T3=reports/textcraft/plan3; R0=$T3/r0; STU=$T3/sft/final; Q=$T3/queue.log; mkdir -p $R0
OLD0=reports/textcraft/final/r0/eval_test.jsonl; P2R0=reports/textcraft/plan/r0/eval_test.jsonl
tlog() { echo "=== $(date '+%F %T') $*" | tee -a $Q; }
cover() { $P scripts/textcraft_round.py cover "$@" > /dev/null 2>&1; }
until [ -f $STU/adapter_config.json ]; do sleep 60; done
tlog "plan3 R0 eval start"
k=0; for spec in 0:203 203:203 406:202 608:202; do off=${spec%:*}; n=${spec#*:}; o=$R0/eval_test_s$k.jsonl
  ( until cover --glob $o --key task --split test810 --offset $off --n $n; do att=$((${att:-0}+1)); $(S 28000) $P -u scripts/textcraft_plan3.py collect --adapter $STU --split test810 --offset $off --n $n --batch 40 --temperature 0 --plan_temperature 0 --out $o --tag tcp3_r0 >> ${o%.jsonl}.log 2>&1 || { tlog "plan3 R0 eval shard $o attempt failed: $(grep -o "OutOfMemoryError\|Error: [^ ]*" ${o%.jsonl}.log | tail -n 1)"; sleep 60; }; done ) &
  k=$((k + 1)); sleep 15; done; wait
$P scripts/textcraft_round.py cover --glob "$R0/eval_test_s*.jsonl" --key task --split test810 --out $R0/eval_test.jsonl | tee -a $Q || { tlog "plan3 R0 eval coverage FAILED"; exit 1; }
$P scripts/textcraft_agent.py report $OLD0 $P2R0 $R0/eval_test.jsonl | tee -a $Q
$P scripts/textcraft_plan_diag.py eval $P2R0 $R0/eval_test.jsonl | tee -a $Q
for ref in "$P2R0:R0_plan2" "$OLD0:R0_old"; do $P scripts/textcraft_round.py compare --r0 ${ref%:*} --r1 $R0/eval_test.jsonl --names ${ref#*:},R0_plan3 | tee -a $Q; done
tlog "PLAN3 R0 EVAL DONE"
