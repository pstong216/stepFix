#!/bin/bash
# TextCraft R1 with the ALFWorld FINAL recipe (user-approved Oct 8: R1 ONLY, no R2/R3).
#   R0 = reports/textcraft/student100/final (100-demo SFT student)
#   TEST = 162 goals (108 test + 54 dev) x 5 distractor seeds = 810 tasks, greedy, per-task budget 2 x oracle (min 6)
#   R1: collect the 1512 TRAIN tasks with R0 at T=1.0 (fixed seeded order train_r1) -> failures among the first 1000 ->
#       K = round(0.2 x failures) -> Model A (standard: attributor_no_observations, variant no_observations) step + diagnosis ->
#       informed self-teacher (hindsight + Model A diagnosis + own analysis), top-2 candidates excluding a- -> one-step evidence F ->
#       env_top (veto no-info, rank (F, hindsight p), top K) + 100 replay (make_replay.py default: shortest wins, 2 steps each, seed 1)
#       -> alfworld_opd_plus.py train objective (lr 1e-5, batch 8, 1 epoch, seed 0) continuing R0's LoRA -> greedy TEST eval.
# Every sharded stage: shard before resume, one file per shard, coverage check before the next stage.
source scripts/lib_baselines.sh; set -o pipefail
F=reports/textcraft/final; R0=$F/r0; R1=$F/r1; STU=reports/textcraft/student100/final; MA=reports/attributor_no_observations/final
mkdir -p $R0 $R1; Q=$F/queue.log
tlog() { echo "=== $(date '+%F %T') $*" | tee -a $Q; }
cover() { $P scripts/textcraft_round.py cover "$@" > /dev/null 2>&1; }
# collect_shards SPLIT ADAPTER TEMP DIR PREFIX TAG "OFF:N ..."   -> DIR/PREFIX_s<k>.jsonl, each loop ends only when its slice is covered
collect_shards() { local SP=$1 AD=$2 TE=$3 DD=$4 PF=$5 TG=$6 k=0
  for spec in $7; do local off=${spec%:*} n=${spec#*:} o=$DD/${PF}_s$k.jsonl
    ( until cover --glob $o --key task --split $SP --offset $off --n $n; do $(S 12000) $P -u scripts/textcraft_agent.py collect --adapter $AD --split $SP --offset $off --n $n --temperature $TE --out $o --tag $TG >> ${o%.jsonl}.log 2>&1 || sleep 60; done ) &
    k=$((k + 1)); sleep 20; done; wait; }
tlog "TextCraft R1 driver start"
# ---- R0 TEST eval (810) and R1 collection (1512, T=1.0), in parallel
( [ -s $R0/eval_test.jsonl ] || { collect_shards test810 $STU 0 $R0 eval_test tcf_r0 "0:270 270:270 540:270"
    $P scripts/textcraft_round.py cover --glob "$R0/eval_test_s*.jsonl" --key task --split test810 --out $R0/eval_test.jsonl | tee -a $Q; } ) &
( [ -s $R1/traj.jsonl ] || { collect_shards train_r1 $STU 1.0 $R1 traj tcf_r1c "0:378 378:378 756:378 1134:378"
    $P scripts/textcraft_round.py cover --glob "$R1/traj_s*.jsonl" --key task --split train_r1 --out $R1/traj.jsonl | tee -a $Q; } ) &
wait
[ -s $R0/eval_test.jsonl ] && [ -s $R1/traj.jsonl ] || { tlog "COVERAGE FAILED after collection -- stopping"; exit 1; }
$P scripts/textcraft_agent.py report $R0/eval_test.jsonl $R1/traj.jsonl | tee -a $Q
T=$R1/traj.jsonl
# ---- failures among the first 1000, K, Model A release (2 shard files)
[ -s $R1/K.json ] || $P scripts/textcraft_round.py fails --traj $T --out_dir $R1 --first 1000 --k_frac 0.2 --n_shards 2 | tee -a $Q
for i in 0 1; do ( until cover --glob $R1/modelA_pred_s$i.jsonl --key id --ids $R1/release_s$i.jsonl; do $(S 12000) $P -u scripts/alfworld_attribute.py --attributor $MA --variant no_observations --release $R1/release_s$i.jsonl --out $R1/modelA_pred_s$i.jsonl >> $R1/modelA_s$i.log 2>&1 || sleep 60; done ) & sleep 20; done; wait
$P scripts/textcraft_round.py cover --glob "$R1/modelA_pred_s*.jsonl" --key id --ids $R1/ids_fail.jsonl | tee -a $Q || { tlog "Model A coverage FAILED -- stopping"; exit 1; }
[ -s $R1/pointer.jsonl ] || $P scripts/textcraft_round.py pointer --traj $T --out_dir $R1 | tee -a $Q
# ---- informed self-teacher (3 GPU shards)
for i in 0 1 2; do ( until grep -q "DONE informed_teacher" $R1/teacher_s$i.log 2>/dev/null; do $(S 12000) $P -u scripts/textcraft_round.py teacher --traj $T --out_dir $R1 --adapter $STU --shard $i 3 >> $R1/teacher_s$i.log 2>&1 || sleep 60; done ) & sleep 20; done; wait
$P scripts/textcraft_round.py cover --glob "$R1/targets_s*.jsonl" --key idt --ids $R1/ids_fail.jsonl | tee -a $Q || { tlog "teacher coverage FAILED -- stopping"; exit 1; }
# ---- one-step evidence (4 CPU shards on this node)
for i in 0 1 2 3; do ( until grep -q "DONE onestep" $R1/onestep_s$i.log 2>/dev/null; do $P -u scripts/textcraft_round.py onestep --traj $T --out_dir $R1 --shard $i 4 >> $R1/onestep_s$i.log 2>&1 || sleep 10; done ) & done; wait
$P scripts/textcraft_round.py cover --glob "$R1/onestep_s*.jsonl" --key idt --ids $R1/ids_fail.jsonl | tee -a $Q || { tlog "one-step coverage FAILED -- stopping"; exit 1; }
# ---- env_top selection + replay
$P scripts/textcraft_round.py select --traj $T --out_dir $R1 | tee -a $Q
$P scripts/make_replay.py --traj $T --n 100 --seed 1 --out $R1/replay.jsonl | tee -a $Q
# ---- train (continuing R0's LoRA)
until [ -f $R1/model/adapter/adapter_config.json ]; do $(S 24000) $P -u scripts/textcraft_round.py train --data $R1/targets_env_top/train.jsonl:1.0 $R1/replay.jsonl:1.0 --adapter $PWD/$STU --lr 1e-5 --epochs 1 --batch 8 --seed 0 --out $R1/model > $R1/train.log 2>&1 && { for k in $(seq 60); do ls $R1/model/adapter > /dev/null 2>&1; [ -f $R1/model/adapter/adapter_config.json ] && break; sleep 2; done; } || { tlog "R1 TRAIN FAILED (retry in 3 min)"; sleep 180; }; done
tail -n 1 $R1/train.log | tee -a $Q
# ---- R1 TEST eval
collect_shards test810 $R1/model/adapter 0 $R1 eval_test tcf_r1 "0:270 270:270 540:270"
$P scripts/textcraft_round.py cover --glob "$R1/eval_test_s*.jsonl" --key task --split test810 --out $R1/eval_test.jsonl | tee -a $Q || { tlog "R1 eval coverage FAILED -- stopping"; exit 1; }
$P scripts/textcraft_agent.py report $R0/eval_test.jsonl $R1/eval_test.jsonl | tee -a $Q
$P scripts/textcraft_round.py compare --r0 $R0/eval_test.jsonl --r1 $R1/eval_test.jsonl | tee -a $Q
tlog "TEXTCRAFT R1 DONE"
