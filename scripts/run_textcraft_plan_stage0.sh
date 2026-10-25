#!/bin/bash
# TextCraft plan pipeline, stage 0 (user-approved Oct 9): plan-then-act SFT student (same 100 TRAIN demos, same SFT hyper-parameters as
# student100) + a pre-check of the Model A hint variants on the OLD R1 training failures (no plan there; final choice is made again on the
# new student's training failures).
export HOLD_NAME_RE='^(tcded_gpu|error_attr_hold_.*)$'
source scripts/lib_baselines.sh; set -o pipefail
PL=reports/textcraft/plan; Q=$PL/queue.log; MA=reports/attributor_no_observations/final; mkdir -p $PL/sft $PL/varcheck
tlog() { echo "=== $(date '+%F %T') $*" | tee -a $Q; }
cover() { $P scripts/textcraft_round.py cover "$@" > /dev/null 2>&1; }
tlog "plan stage 0 start"
( until [ -f $PL/sft/final/adapter_config.json ]; do $(S 24000) $P -u scripts/textcraft_plan.py sft --traj $PL/gold_train_plan.jsonl --n_demos 100 --out $PL/sft > $PL/sft/sft.log 2>&1 || { tlog "plan SFT FAILED (retry)"; sleep 120; }; done; tlog "plan SFT done: $(grep -o "{'train_runtime[^}]*}" $PL/sft/sft.log)" ) &
# Model A hint pre-check on the old R1 failures: release = full task statement + hint as query (old episodes have no plan)
$P - <<PY
import json, sys
sys.path.insert(0, "scripts")
from textcraft_plan import HINT
from textcraft_agent import to_release
eps = {json.loads(l)["id"]: json.loads(l) for l in open("reports/textcraft/final/r1/traj.jsonl")}
ids = [json.loads(l)["id"] for l in open("reports/textcraft/final/r1/ids_fail.jsonl")]
rows = []
for i in ids:
    r = to_release(eps[i]); r["task"]["query"] = eps[i]["initial_obs"] + "\n\n" + HINT; rows.append(r)
for k in range(2): open(f"$PL/varcheck/release_s{k}.jsonl", "w").writelines(json.dumps(r) + "\n" for r in rows[k::2])
open("$PL/varcheck/ids_fail.jsonl", "w").writelines(json.dumps({"id": i}) + "\n" for i in ids)
print("varcheck release:", len(rows), "old R1 failures")
PY
for v in no_observations full; do mkdir -p $PL/varcheck/modelA_$v; for i in 0 1; do ( o=$PL/varcheck/modelA_$v/modelA_pred_s$i.jsonl; until cover --glob $o --key id --ids $PL/varcheck/release_s$i.jsonl; do $(S 16000) $P -u scripts/alfworld_attribute.py --attributor $MA --variant $v --release $PL/varcheck/release_s$i.jsonl --out $o >> ${o%.jsonl}.log 2>&1 || sleep 60; done ) & sleep 15; done; done
wait
for v in no_observations full; do $P scripts/textcraft_round.py cover --glob "$PL/varcheck/modelA_$v/modelA_pred_s*.jsonl" --key id --ids $PL/varcheck/ids_fail.jsonl | tee -a $Q; done
$P scripts/textcraft_plan.py variant --traj reports/textcraft/final/r1/traj.jsonl --out_dir $PL/varcheck | tee -a $Q
tlog "plan stage 0 DONE"
