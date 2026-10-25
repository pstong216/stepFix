#!/bin/bash
# TextCraft R1_v3 (user-approved Oct 9).  REUSES the R1 collection (reports/textcraft/final/r1/traj.jsonl: R0 at T=1.0 on the 1512 train
# tasks); failures = the same 379 among the first 1000, K = 76.  Four changes vs R1, everything else unchanged:
#   1. Model A (same weights, attributor_no_observations) sees the FULL trajectory: --variant full (every [observation] block kept) and the
#      complete task statement (all crafting commands + goal) as its query; integer step_number, reasoning None, is_final_answer False, no score
#   2. the informed teacher sees the whole failed trajectory: full prefix, every remaining step with its full observation, outcome, task
#      (textcraft_round.py teacher --full_view); Model A diagnosis + the student's own analysis as before; candidates = top-2 excluding a-
#   3. margin rule: env_top_margin (candidate F strictly above the taken action's F), K = 76
#   4. replay: 100 steps split by crafting depth in proportion to the training-collection failures, shortest wins per depth, unfilled quota
#      moved to the next shallower depth with wins (textcraft_round.py replay_fw --carry)
# Same trainer / hyper-parameters from R0's LoRA, same 810-task greedy TEST eval.  Shard before resume, one file per shard, coverage checks.
source scripts/lib_baselines.sh; set -o pipefail
F=reports/textcraft/final; R0=$F/r0; R1=$F/r1; V2=$F/r1_v2; V=$F/r1_v3; T=$R1/traj.jsonl
STU=reports/textcraft/student100/final; MA=reports/attributor_no_observations/final; mkdir -p $V; Q=$F/queue.log
tlog() { echo "=== $(date '+%F %T') $*" | tee -a $Q; }
cover() { $P scripts/textcraft_round.py cover "$@" > /dev/null 2>&1; }
tlog "TextCraft R1_v3 driver start"
# ---- 1. release (complete task statement as the query) for the SAME 379 failures; Model A --variant full
[ -s $V/K.json ] || $P scripts/textcraft_round.py fails --traj $T --out_dir $V --first 1000 --k_frac 0.2 --n_shards 2 --query_full | tee -a $Q
cmp -s $V/ids_fail.jsonl $R1/ids_fail.jsonl || { tlog "v3 failure ids differ from R1 -- stopping"; exit 1; }
for i in 0 1; do ( until cover --glob $V/modelA_pred_s$i.jsonl --key id --ids $V/release_s$i.jsonl; do $(S 16000) $P -u scripts/alfworld_attribute.py --attributor $MA --variant full --release $V/release_s$i.jsonl --out $V/modelA_pred_s$i.jsonl >> $V/modelA_s$i.log 2>&1 || sleep 60; done ) & sleep 20; done; wait
$P scripts/textcraft_round.py cover --glob "$V/modelA_pred_s*.jsonl" --key id --ids $V/ids_fail.jsonl | tee -a $Q || { tlog "v3 Model A coverage FAILED -- stopping"; exit 1; }
[ -s $V/pointer.jsonl ] || $P scripts/textcraft_round.py pointer --traj $T --out_dir $V | tee -a $Q
# ---- 2. informed teacher with the whole failed trajectory (3 GPU shards)
for i in 0 1 2; do ( until grep -q "DONE informed_teacher" $V/teacher_s$i.log 2>/dev/null; do $(S 16000) $P -u scripts/textcraft_round.py teacher --traj $T --out_dir $V --adapter $STU --shard $i 3 --full_view >> $V/teacher_s$i.log 2>&1 || sleep 60; done ) & sleep 20; done; wait
$P scripts/textcraft_round.py cover --glob "$V/targets_s*.jsonl" --key idt --ids $V/ids_fail.jsonl | tee -a $Q || { tlog "v3 teacher coverage FAILED -- stopping"; exit 1; }
# ---- one-step evidence (4 CPU shards on this node), F unchanged
for i in 0 1 2 3; do ( until grep -q "DONE onestep" $V/onestep_s$i.log 2>/dev/null; do $P -u scripts/textcraft_round.py onestep --traj $T --out_dir $V --shard $i 4 >> $V/onestep_s$i.log 2>&1 || sleep 10; done ) & done; wait
$P scripts/textcraft_round.py cover --glob "$V/onestep_s*.jsonl" --key idt --ids $V/ids_fail.jsonl | tee -a $Q || { tlog "v3 one-step coverage FAILED -- stopping"; exit 1; }
# ---- 3. margin selection, 4. failure-weighted replay with carry-over
$P scripts/textcraft_round.py select --traj $T --out_dir $V --rule env_top_margin | tee -a $Q
$P scripts/textcraft_round.py replay_fw --traj $T --out $V/replay.jsonl --n 100 --carry | tee -a $Q
# ---- train (same trainer / hyper-parameters, from R0's LoRA)
until [ -f $V/model/adapter/adapter_config.json ]; do $(S 24000) $P -u scripts/textcraft_round.py train --data $V/targets_env_top_margin/train.jsonl:1.0 $V/replay.jsonl:1.0 --adapter $PWD/$STU --lr 1e-5 --epochs 1 --batch 8 --seed 0 --out $V/model > $V/train.log 2>&1 && { for k in $(seq 60); do ls $V/model/adapter > /dev/null 2>&1; [ -f $V/model/adapter/adapter_config.json ] && break; sleep 2; done; } || { tlog "R1_v3 TRAIN FAILED (retry in 3 min)"; sleep 180; }; done
tail -n 1 $V/train.log | tee -a $Q
# ---- TEST eval (810, greedy)
k=0; for spec in 0:270 270:270 540:270; do off=${spec%:*}; n=${spec#*:}; o=$V/eval_test_s$k.jsonl
  ( until cover --glob $o --key task --split test810 --offset $off --n $n; do $(S 12000) $P -u scripts/textcraft_agent.py collect --adapter $V/model/adapter --split test810 --offset $off --n $n --temperature 0 --out $o --tag tcf_r1v3 >> ${o%.jsonl}.log 2>&1 || sleep 60; done ) &
  k=$((k + 1)); sleep 20; done; wait
$P scripts/textcraft_round.py cover --glob "$V/eval_test_s*.jsonl" --key task --split test810 --out $V/eval_test.jsonl | tee -a $Q || { tlog "R1_v3 eval coverage FAILED -- stopping"; exit 1; }
$P scripts/textcraft_agent.py report $R0/eval_test.jsonl $R1/eval_test.jsonl $V/eval_test.jsonl | tee -a $Q
$P scripts/textcraft_round.py compare --r0 $R0/eval_test.jsonl --r1 $V/eval_test.jsonl --names R0,R1_v3 | tee -a $Q
$P scripts/textcraft_round.py compare --r0 $R1/eval_test.jsonl --r1 $V/eval_test.jsonl --names R1_old,R1_v3 | tee -a $Q
[ -s $V2/eval_test.jsonl ] && $P scripts/textcraft_round.py compare --r0 $V2/eval_test.jsonl --r1 $V/eval_test.jsonl --names R1_v2,R1_v3 | tee -a $Q
wait
tlog "TEXTCRAFT R1_v3 DONE"
