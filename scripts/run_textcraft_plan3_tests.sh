#!/bin/bash
# plan3 tests A (teacher with derivation/verification/few-shot, masked and unmasked; no training) and B (Model A hint 2, both variants),
# on the plan2 R1 failures; at most 2 GPU slots, sharing with the plan2 R1 eval.
export HOLD_NAME_RE='^(tcded_gpu|error_attr_hold_.*)$'
source scripts/lib_baselines.sh
T3=reports/textcraft/plan3; R1=reports/textcraft/plan/r1; STU=reports/textcraft/plan/sft2/final; MA=reports/attributor_no_observations/final; Q=$T3/queue.log
tlog() { echo "=== $(date '+%F %T') $*" | tee -a $Q; }
cover() { $P scripts/textcraft_round.py cover "$@" > /dev/null 2>&1; }
tlog "plan3 tests A/B start"
slot() { local MASK=$1 V=$2
  until grep -q "DONE teacher_test" $T3/testA/mask$MASK.log 2>/dev/null; do $(S 16000) $P -u scripts/textcraft_plan3.py teacher_test --traj $R1/traj.jsonl --src_dir $R1 --states $T3/testA/states.jsonl --out $T3/testA/mask$MASK.jsonl --adapter $STU --mask $MASK >> $T3/testA/mask$MASK.log 2>&1 || sleep 60; done
  tlog "test A mask=$MASK done"
  mkdir -p $T3/testB/modelA_$V; for i in 0 1; do o=$T3/testB/modelA_$V/modelA_pred_s$i.jsonl; until cover --glob $o --key id --ids $T3/testB/release_s$i.jsonl; do $(S 16000) $P -u scripts/alfworld_attribute.py --attributor $MA --variant $V --release $T3/testB/release_s$i.jsonl --out $o >> ${o%.jsonl}.log 2>&1 || sleep 60; done; done
  tlog "test B $V done"; }
slot 1 no_observations & sleep 20
slot 0 full & wait
$P scripts/textcraft_plan3.py test_report --traj $R1/traj.jsonl --src_dir $R1 --states $T3/testA/states.jsonl --files "$T3/testA/mask1.jsonl" "$T3/testA/mask0.jsonl" --names masked unmasked | tee -a $Q
$P scripts/textcraft_plan3.py hint2_report --traj $R1/traj.jsonl --src_dir $R1 --out_dir $T3/testB | tee -a $Q
tlog "plan3 tests A/B DONE"
