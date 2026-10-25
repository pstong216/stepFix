#!/bin/bash
# TextCraft R2_v4 = final recipe at 20%, round 2 (user-approved Oct 9).  Start = the R1_v4 model (470/810).
#   collect the 1512 TRAIN tasks with R1_v4 at T=1.0 in the same seeded order as R1 (train_r1) -> failures among the first 1000 ->
#   K = round(0.2 x failures) -> standard Model A pointer (attributor_no_observations, variant no_observations, same release format) ->
#   FullHindsight full-view teacher (the R1_v4 model) -> one-step evidence -> env_top_margin -> fail_weighted replay by depth with
#   carry-over from the R2 collection -> same trainer / hyper-parameters continuing the R1_v4 adapter -> 810-task greedy TEST eval.
# Shard before resume, one file per shard, coverage checks before every next stage.
source scripts/lib_baselines.sh; set -o pipefail
F=reports/textcraft/final; R0=$F/r0; V4=$F/r1_v4; V=$F/r2_v4; START=$V4/model/adapter; MA=reports/attributor_no_observations/final
mkdir -p $V; Q=$F/queue.log
tlog() { echo "=== $(date '+%F %T') $*" | tee -a $Q; }
cover() { $P scripts/textcraft_round.py cover "$@" > /dev/null 2>&1; }
[ -f $START/adapter_config.json ] || { tlog "R2_v4: start adapter $START missing -- stopping"; exit 1; }
tlog "TextCraft R2_v4 driver start (start = $START)"
# ---- collection: 1512 train tasks, T=1.0, 4 shards
k=0; for spec in 0:378 378:378 756:378 1134:378; do off=${spec%:*}; n=${spec#*:}; o=$V/traj_s$k.jsonl
  ( until cover --glob $o --key task --split train_r1 --offset $off --n $n; do $(S 12000) $P -u scripts/textcraft_agent.py collect --adapter $START --split train_r1 --offset $off --n $n --temperature 1.0 --out $o --tag tcf_r2c >> ${o%.jsonl}.log 2>&1 || sleep 60; done ) &
  k=$((k + 1)); sleep 20; done; wait
$P scripts/textcraft_round.py cover --glob "$V/traj_s*.jsonl" --key task --split train_r1 --out $V/traj.jsonl | tee -a $Q || { tlog "R2_v4 collection coverage FAILED -- stopping"; exit 1; }
T=$V/traj.jsonl
$P scripts/textcraft_agent.py report $T | sed 's/^== /== [R2_v4 collection] /' | tee -a $Q
# ---- failures among the first 1000, K, standard release (2 shard files), standard Model A
[ -s $V/K.json ] || $P scripts/textcraft_round.py fails --traj $T --out_dir $V --first 1000 --k_frac 0.2 --n_shards 2 | sed 's/^fails:/[R2_v4] fails:/' | tee -a $Q
for i in 0 1; do ( until cover --glob $V/modelA_pred_s$i.jsonl --key id --ids $V/release_s$i.jsonl; do $(S 12000) $P -u scripts/alfworld_attribute.py --attributor $MA --variant no_observations --release $V/release_s$i.jsonl --out $V/modelA_pred_s$i.jsonl >> $V/modelA_s$i.log 2>&1 || sleep 60; done ) & sleep 20; done; wait
$P scripts/textcraft_round.py cover --glob "$V/modelA_pred_s*.jsonl" --key id --ids $V/ids_fail.jsonl | tee -a $Q || { tlog "R2_v4 Model A coverage FAILED -- stopping"; exit 1; }
[ -s $V/pointer.jsonl ] || $P scripts/textcraft_round.py pointer --traj $T --out_dir $V | sed 's/^Model A pointer/[R2_v4] Model A pointer/' | tee -a $Q
# ---- full-view informed teacher = the R1_v4 model (3 GPU shards)
for i in 0 1 2; do ( until grep -q "DONE informed_teacher" $V/teacher_s$i.log 2>/dev/null; do $(S 16000) $P -u scripts/textcraft_round.py teacher --traj $T --out_dir $V --adapter $START --shard $i 3 --full_view >> $V/teacher_s$i.log 2>&1 || sleep 60; done ) & sleep 20; done; wait
$P scripts/textcraft_round.py cover --glob "$V/targets_s*.jsonl" --key idt --ids $V/ids_fail.jsonl | tee -a $Q || { tlog "R2_v4 teacher coverage FAILED -- stopping"; exit 1; }
# ---- one-step evidence (4 CPU shards on this node)
for i in 0 1 2 3; do ( until grep -q "DONE onestep" $V/onestep_s$i.log 2>/dev/null; do $P -u scripts/textcraft_round.py onestep --traj $T --out_dir $V --shard $i 4 >> $V/onestep_s$i.log 2>&1 || sleep 10; done ) & done; wait
$P scripts/textcraft_round.py cover --glob "$V/onestep_s*.jsonl" --key idt --ids $V/ids_fail.jsonl | tee -a $Q || { tlog "R2_v4 one-step coverage FAILED -- stopping"; exit 1; }
# ---- selection + replay from the R2 collection
$P scripts/textcraft_round.py select --traj $T --out_dir $V --rule env_top_margin | sed 's/^=== textcraft/=== [R2_v4] textcraft/; s/^margin rule/[R2_v4] margin rule/' | tee -a $Q
$P scripts/textcraft_round.py replay_fw --traj $T --out $V/replay.jsonl --n 100 --carry | sed 's/^/[R2_v4] /' | tee -a $Q
# ---- train, continuing the R1_v4 adapter
att=0
until [ -f $V/model/adapter/adapter_config.json ]; do
  att=$((att + 1)); CH=$([ $att -le 3 ] && echo 8 || echo 2)
  $(S 24000) env TC_LP_CHUNK=$CH $P -u scripts/textcraft_round.py train --data $V/targets_env_top_margin/train.jsonl:1.0 $V/replay.jsonl:1.0 --adapter $PWD/$START --lr 1e-5 --epochs 1 --batch 8 --seed 0 --out $V/model > $V/train.log 2>&1 && { echo $CH > $V/model_chunk.txt; for k in $(seq 60); do ls $V/model/adapter > /dev/null 2>&1; [ -f $V/model/adapter/adapter_config.json ] && break; sleep 2; done; } || { tlog "R2_v4 TRAIN FAILED (attempt $att, chunk $CH; retry in 3 min)"; sleep 180; }
done
tlog "R2_v4 trained with command-scoring chunk $(cat $V/model_chunk.txt): $(tail -n 1 $V/train.log)"
# ---- TEST eval (810, greedy)
k=0; for spec in 0:270 270:270 540:270; do off=${spec%:*}; n=${spec#*:}; o=$V/eval_test_s$k.jsonl
  ( until cover --glob $o --key task --split test810 --offset $off --n $n; do $(S 12000) $P -u scripts/textcraft_agent.py collect --adapter $V/model/adapter --split test810 --offset $off --n $n --temperature 0 --out $o --tag tcf_r2v4 >> ${o%.jsonl}.log 2>&1 || sleep 60; done ) &
  k=$((k + 1)); sleep 20; done; wait
$P scripts/textcraft_round.py cover --glob "$V/eval_test_s*.jsonl" --key task --split test810 --out $V/eval_test.jsonl | tee -a $Q || { tlog "R2_v4 eval coverage FAILED -- stopping"; exit 1; }
$P scripts/textcraft_agent.py report $R0/eval_test.jsonl $V4/eval_test.jsonl $V/eval_test.jsonl | tee -a $Q
for ref in "$R0:R0" "$V4:R1_v4"; do $P scripts/textcraft_round.py compare --r0 ${ref%:*}/eval_test.jsonl --r1 $V/eval_test.jsonl --names ${ref#*:},R2_v4 | tee -a $Q; done
tlog "TEXTCRAFT R2_v4 DONE"
