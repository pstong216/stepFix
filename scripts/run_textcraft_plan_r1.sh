#!/bin/bash
# TextCraft plan pipeline (user-approved Oct 9): new R0 = plan-then-act SFT student (reports/textcraft/plan/sft/final); R0 TEST eval (810,
# greedy plan + greedy actions) and the R1 round with the final recipe:
#   collect the 1512 TRAIN tasks (train_r1 order) with R0: plan greedy, actions at T=1.0 -> failures among the first 1000 -> K = 20% ->
#   Model A (same weights; query = complete task statement + hint; plan = step-1 reasoning) in BOTH variants (no_observations, full) ->
#   variant chosen on these TRAINING failures by agreement with the oracle first_waste -> pointer -> analysis-first full-view teacher (plan +
#   whole trajectory + diagnosis, 256-token analysis) -> one-step evidence F -> env_top_margin -> fail_weighted replay by depth with carry-over
#   (+ plan-generation targets for t=0 rows) -> same trainer from R0's LoRA (corrections = action at the pointed state with the plan in
#   context) -> 810-task TEST eval; McNemar + goal-level sign test vs the new R0, the old R0 (454) and the old R1_v4 (470).
# Shard before resume, one file per shard, coverage checks before every next stage.
export HOLD_NAME_RE='^(tcded_gpu|error_attr_hold_.*)$'
source scripts/lib_baselines.sh; set -o pipefail
PL=reports/textcraft/plan; R0=$PL/r0; V=$PL/r1; STU=$PL/sft/final; MA=reports/attributor_no_observations/final; Q=$PL/queue.log
OLD0=reports/textcraft/final/r0/eval_test.jsonl; OLDV4=reports/textcraft/final/r1_v4/eval_test.jsonl; mkdir -p $R0 $V
tlog() { echo "=== $(date '+%F %T') $*" | tee -a $Q; }
cover() { $P scripts/textcraft_round.py cover "$@" > /dev/null 2>&1; }
pcollect() { local SP=$1 AD=$2 TE=$3 DD=$4 PF=$5 TG=$6 k=0   # pcollect SPLIT ADAPTER T DIR PREFIX TAG "OFF:N ..."
  for spec in $7; do local off=${spec%:*} n=${spec#*:} o=$DD/${PF}_s$k.jsonl
    ( until cover --glob $o --key task --split $SP --offset $off --n $n; do $(S 12000) $P -u scripts/textcraft_plan.py collect --adapter $AD --split $SP --offset $off --n $n --temperature $TE --out $o --tag $TG >> ${o%.jsonl}.log 2>&1 || sleep 60; done ) &
    k=$((k + 1)); sleep 20; done; wait; }
[ -f $STU/adapter_config.json ] || { tlog "plan R0 adapter missing -- stopping"; exit 1; }
tlog "plan R0/R1 driver start"
# ---- new R0 TEST eval and the R1 collection, in parallel
( [ -s $R0/eval_test.jsonl ] || { pcollect test810 $STU 0 $R0 eval_test tcp_r0 "0:270 270:270 540:270"
    $P scripts/textcraft_round.py cover --glob "$R0/eval_test_s*.jsonl" --key task --split test810 --out $R0/eval_test.jsonl | tee -a $Q || exit 1
    $P scripts/textcraft_agent.py report $OLD0 $R0/eval_test.jsonl | tee -a $Q; $P scripts/textcraft_plan.py plan_report $R0/eval_test.jsonl | tee -a $Q
    $P scripts/textcraft_round.py compare --r0 $OLD0 --r1 $R0/eval_test.jsonl --names R0_old,R0_plan | tee -a $Q
    $P scripts/textcraft_round.py compare --r0 $OLDV4 --r1 $R0/eval_test.jsonl --names R1_v4_old,R0_plan | tee -a $Q
    tlog "PLAN R0 EVAL DONE"; } ) &
( [ -s $V/traj.jsonl ] || { pcollect train_r1 $STU 1.0 $V traj tcp_r1c "0:378 378:378 756:378 1134:378"
    $P scripts/textcraft_round.py cover --glob "$V/traj_s*.jsonl" --key task --split train_r1 --out $V/traj.jsonl | tee -a $Q; } ) &
wait
[ -s $V/traj.jsonl ] || { tlog "plan R1 collection coverage FAILED -- stopping"; exit 1; }
T=$V/traj.jsonl
$P scripts/textcraft_agent.py report $T | sed 's/^== /== [plan R1 collection] /' | tee -a $Q; $P scripts/textcraft_plan.py plan_report $T | tee -a $Q
# ---- failures, K, release, Model A in both variants, variant choice on these TRAINING failures, pointer
[ -s $V/K.json ] || $P scripts/textcraft_plan.py fails --traj $T --out_dir $V --first 1000 --k_frac 0.2 --n_shards 2 | tee -a $Q
for v in no_observations full; do mkdir -p $V/modelA_$v; for i in 0 1; do ( o=$V/modelA_$v/modelA_pred_s$i.jsonl; until cover --glob $o --key id --ids $V/release_s$i.jsonl; do $(S 16000) $P -u scripts/alfworld_attribute.py --attributor $MA --variant $v --release $V/release_s$i.jsonl --out $o >> ${o%.jsonl}.log 2>&1 || sleep 60; done ) & sleep 15; done; done; wait
for v in no_observations full; do $P scripts/textcraft_round.py cover --glob "$V/modelA_$v/modelA_pred_s*.jsonl" --key id --ids $V/ids_fail.jsonl | tee -a $Q || { tlog "plan R1 Model A ($v) coverage FAILED -- stopping"; exit 1; }; done
[ -s $V/modelA_variant.json ] || $P scripts/textcraft_plan.py variant --traj $T --out_dir $V | tee -a $Q
[ -s $V/pointer.jsonl ] || $P scripts/textcraft_round.py pointer --traj $T --out_dir $V | sed 's/^Model A pointer/[plan R1] Model A pointer/' | tee -a $Q
# ---- analysis-first full-view teacher (3 GPU shards)
for i in 0 1 2; do ( until grep -q "DONE informed_teacher" $V/teacher_s$i.log 2>/dev/null; do $(S 16000) $P -u scripts/textcraft_plan.py teacher --traj $T --out_dir $V --adapter $STU --shard $i 3 >> $V/teacher_s$i.log 2>&1 || sleep 60; done ) & sleep 20; done; wait
$P scripts/textcraft_round.py cover --glob "$V/targets_s*.jsonl" --key idt --ids $V/ids_fail.jsonl | tee -a $Q || { tlog "plan R1 teacher coverage FAILED -- stopping"; exit 1; }
# ---- one-step evidence (4 CPU shards on this node)
for i in 0 1 2 3; do ( until grep -q "DONE onestep" $V/onestep_s$i.log 2>/dev/null; do $P -u scripts/textcraft_round.py onestep --traj $T --out_dir $V --shard $i 4 >> $V/onestep_s$i.log 2>&1 || sleep 10; done ) & done; wait
$P scripts/textcraft_round.py cover --glob "$V/onestep_s*.jsonl" --key idt --ids $V/ids_fail.jsonl | tee -a $Q || { tlog "plan R1 one-step coverage FAILED -- stopping"; exit 1; }
# ---- selection + replay (with plan-generation targets for t=0 rows)
[ -s $V/targets_env_top_margin/train.jsonl ] || $P scripts/textcraft_round.py select --traj $T --out_dir $V --rule env_top_margin | sed 's/^=== textcraft/=== [plan R1] textcraft/; s/^margin rule/[plan R1] margin rule/' | tee -a $Q
[ -s $V/replay.jsonl ] || { $P scripts/textcraft_round.py replay_fw --traj $T --out $V/replay_actions.jsonl --n 100 --carry | sed 's/^/[plan R1] /' | tee -a $Q
  $P scripts/textcraft_plan.py replay_plan --replay $V/replay_actions.jsonl --traj $T --out $V/replay.jsonl | sed 's/^/[plan R1] /' | tee -a $Q; }
# ---- train from R0's LoRA (command-scoring chunk 8 for the first 3 attempts, 2 after that: memory only)
att=0
until [ -f $V/model/adapter/adapter_config.json ]; do
  att=$((att + 1)); CH=$([ $att -le 3 ] && echo 8 || echo 2)
  $(S 24000) env TC_LP_CHUNK=$CH $P -u scripts/textcraft_plan.py train --data $V/targets_env_top_margin/train.jsonl:1.0 $V/replay.jsonl:1.0 --adapter $PWD/$STU --lr 1e-5 --epochs 1 --batch 8 --seed 0 --out $V/model > $V/train.log 2>&1 && { echo $CH > $V/model_chunk.txt; for k in $(seq 60); do ls $V/model/adapter > /dev/null 2>&1; [ -f $V/model/adapter/adapter_config.json ] && break; sleep 2; done; } || { tlog "plan R1 TRAIN FAILED (attempt $att, chunk $CH; retry in 3 min)"; sleep 180; }
done
tlog "plan R1 trained with command-scoring chunk $(cat $V/model_chunk.txt): $(tail -n 1 $V/train.log)"
# ---- R1 TEST eval
pcollect test810 $V/model/adapter 0 $V eval_test tcp_r1 "0:270 270:270 540:270"
$P scripts/textcraft_round.py cover --glob "$V/eval_test_s*.jsonl" --key task --split test810 --out $V/eval_test.jsonl | tee -a $Q || { tlog "plan R1 eval coverage FAILED -- stopping"; exit 1; }
$P scripts/textcraft_agent.py report $OLD0 $OLDV4 $R0/eval_test.jsonl $V/eval_test.jsonl | tee -a $Q; $P scripts/textcraft_plan.py plan_report $R0/eval_test.jsonl $V/eval_test.jsonl | tee -a $Q
for ref in "$R0/eval_test.jsonl:R0_plan" "$OLD0:R0_old" "$OLDV4:R1_v4_old"; do $P scripts/textcraft_round.py compare --r0 ${ref%:*} --r1 $V/eval_test.jsonl --names ${ref#*:},R1_plan | tee -a $Q; done
tlog "PLAN R1 DONE"
