#!/bin/bash
# TextCraft R1_v2 (user-approved Oct 9): REUSES the R1 collection, Model A pointer, teacher candidates and one-step evidence in
# reports/textcraft/final/r1; redoes selection (env_top_margin: candidate F must STRICTLY exceed the taken action's F, K = 76) ->
# failure-weighted replay (100 steps split over crafting depth by the training-collection failures, complete shortest wins) ->
# the same trainer / hyper-parameters from R0's LoRA -> the same 810-task greedy TEST eval.
source scripts/lib_baselines.sh; set -o pipefail
F=reports/textcraft/final; R0=$F/r0; R1=$F/r1; V=$F/r1_v2; STU=reports/textcraft/student100/final; mkdir -p $V; Q=$F/queue.log
tlog() { echo "=== $(date '+%F %T') $*" | tee -a $Q; }
cover() { $P scripts/textcraft_round.py cover "$@" > /dev/null 2>&1; }
tlog "TextCraft R1_v2 driver start"
[ -s $V/targets_env_top_margin/train.jsonl ] || $P scripts/textcraft_round.py select --traj $R1/traj.jsonl --out_dir $R1 --k 76 --rule env_top_margin --dest $V | tee -a $Q
[ -s $V/replay.jsonl ] || $P scripts/textcraft_round.py replay_fw --traj $R1/traj.jsonl --out $V/replay.jsonl --n 100 | tee -a $Q
until [ -f $V/model/adapter/adapter_config.json ]; do $(S 24000) $P -u scripts/textcraft_round.py train --data $V/targets_env_top_margin/train.jsonl:1.0 $V/replay.jsonl:1.0 --adapter $PWD/$STU --lr 1e-5 --epochs 1 --batch 8 --seed 0 --out $V/model > $V/train.log 2>&1 && { for k in $(seq 60); do ls $V/model/adapter > /dev/null 2>&1; [ -f $V/model/adapter/adapter_config.json ] && break; sleep 2; done; } || { tlog "R1_v2 TRAIN FAILED (retry in 3 min)"; sleep 180; }; done
tail -n 1 $V/train.log | tee -a $Q
k=0; for spec in 0:270 270:270 540:270; do off=${spec%:*}; n=${spec#*:}; o=$V/eval_test_s$k.jsonl
  ( until cover --glob $o --key task --split test810 --offset $off --n $n; do $(S 12000) $P -u scripts/textcraft_agent.py collect --adapter $V/model/adapter --split test810 --offset $off --n $n --temperature 0 --out $o --tag tcf_r1v2 >> ${o%.jsonl}.log 2>&1 || sleep 60; done ) &
  k=$((k + 1)); sleep 20; done; wait
$P scripts/textcraft_round.py cover --glob "$V/eval_test_s*.jsonl" --key task --split test810 --out $V/eval_test.jsonl | tee -a $Q || { tlog "R1_v2 eval coverage FAILED -- stopping"; exit 1; }
$P scripts/textcraft_agent.py report $R0/eval_test.jsonl $R1/eval_test.jsonl $V/eval_test.jsonl | tee -a $Q
$P scripts/textcraft_round.py compare --r0 $R0/eval_test.jsonl --r1 $V/eval_test.jsonl --names R0,R1_v2 | tee -a $Q
$P scripts/textcraft_round.py compare --r0 $R1/eval_test.jsonl --r1 $V/eval_test.jsonl --names R1_old,R1_v2 | tee -a $Q
tlog "TEXTCRAFT R1_v2 DONE"
