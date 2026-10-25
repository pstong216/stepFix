#!/bin/bash
# TextCraft plan-at-every-step student (option B): SFT on the same 100 TRAIN demos (plan SFT + action SFT), same hyper-parameters as student100.
export HOLD_NAME_RE='^(tcded_gpu|error_attr_hold_.*)$'
source scripts/lib_baselines.sh
PL=reports/textcraft/plan; Q=$PL/queue.log; mkdir -p $PL/sft2
tlog() { echo "=== $(date '+%F %T') $*" | tee -a $Q; }
tlog "plan2 SFT start"
until [ -f $PL/sft2/final/adapter_config.json ]; do $(S 24000) $P -u scripts/textcraft_plan2.py sft --traj $PL/gold_train_plan2.jsonl --n_demos 100 --out $PL/sft2 > $PL/sft2/sft.log 2>&1 || { tlog "plan2 SFT FAILED (retry in 2 min)"; sleep 120; }; done
tlog "plan2 SFT done: $(grep -o "{'train_runtime[^}]*}" $PL/sft2/sft.log)"
