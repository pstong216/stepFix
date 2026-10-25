#!/bin/bash
# plan3 student SFT (derivation + check at t=0, re-derivation after failed crafts, verification augmentation); same hyper-parameters as student100.
export HOLD_NAME_RE='^(tcded_gpu|error_attr_hold_.*)$'
source scripts/lib_baselines.sh
T3=reports/textcraft/plan3; Q=$T3/queue.log; mkdir -p $T3/sft
tlog() { echo "=== $(date '+%F %T') $*" | tee -a $Q; }
tlog "plan3 SFT start ($(grep -c . $T3/sft_pairs.jsonl) pairs)"
att=0; until [ -f $T3/sft/final/adapter_config.json ]; do att=$((att+1)); $(S 34000) $P -u scripts/textcraft_plan3.py sft3 --pairs $T3/sft_pairs.jsonl --out $T3/sft > $T3/sft/sft_attempt_r$att.log 2>&1 || { tlog "plan3 SFT FAILED (restart attempt $att: $(tr "\\r" "\\n" < $T3/sft/sft_attempt_r$att.log | grep -o "PREEMPTION\|OutOfMemoryError\|Error: .*" | tail -n 1); retry in 60 s)"; sleep 60; }; done
tlog "plan3 SFT done: $(grep -o "{'train_runtime[^}]*}" $T3/sft/sft.log)"
