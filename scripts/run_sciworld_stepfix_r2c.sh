#!/bin/bash
# Oct 10 16:30 (user-approved): (I2) CUMULATIVE corrections at R2: from r1b, train on r1b env_top (62) + r2h env_top (63) + r2h replay_fw (100).
source scripts/lib_sciworld_stepfix.sh; D=$SF/r2c; mkdir -p $D; export GMEM=9000
slog "r2c: start (r1b 62 + r2h 63 corrections + r2h replay_fw 100; from r1b)"
$P - <<PY
import json
a=[json.loads(l) for l in open("$SF/r1b/targets_env_top.jsonl")]; b=[json.loads(l) for l in open("$SF/r2h/targets_env_top.jsonl")]
kb={(r["src"],r["id"],r["t"]) for r in b}; a=[r for r in a if (r["src"],r["id"],r["t"]) not in kb]   # a state in both: keep the R2 version
open("$D/targets_cum.jsonl","w").writelines(json.dumps(r)+"\n" for r in a+b); print("r2c cumulative corrections:", len(a), "+", len(b))
PY
until [ -f $D/model/adapter/adapter_config.json ]; do echo "### attempt $(date '+%F %T')" >> $D/train.log
  wait_free "train --data .* --out $D/model "; $(SG 20000) $P -u scripts/sciworld_stepfix.py train --data $D/targets_cum.jsonl $SF/r2h/replay_fw.jsonl --adapter $SF/r1b/stepfix/adapter --out $D/model --seed 2 >> $D/train.log 2>&1 || { slog "r2c: train attempt failed (retry)"; sleep 60; }
  for k in $(seq 30); do [ -f $D/model/adapter/adapter_config.json ] && break; sleep 2; done; done
slog "r2c: trained ($(tail -1 $D/model/train_log.jsonl))"
eval_sw $D/eval $D/model/adapter r2c; slog "r2c: 848 eval DONE"
for rf in "R0=$SF/r0/eval_test.jsonl,reports/sciworld/refix/eval_dev_safe.jsonl" "r1b=$SF/r1b/eval_stepfix/eval_test.jsonl,$SF/r1b/eval_stepfix/eval_dev.jsonl" "r2h=$SF/r2h/eval/eval_test.jsonl,$SF/r2h/eval/eval_dev.jsonl"; do
  $P scripts/sciworld_stepfix_compare.py --a $rf --b r2c=$D/eval/eval_test.jsonl,$D/eval/eval_dev.jsonl | head -1 | tee -a $SF/queue.log; done
slog "r2c DONE"
