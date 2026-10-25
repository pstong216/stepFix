#!/bin/bash
# Smoke test of every new piece of the TextCraft plan pipeline on a handful of tasks (no retry loops; any error stops the script):
#   plan-then-act collect (greedy test + T=1.0 train) -> plan report -> release (task + hint, plan as step-1 reasoning) -> Model A (both
#   variants) -> variant choice -> pointer -> analysis-first full-view teacher -> one-step evidence -> env_top_margin -> replay + plan rows
#   -> train (kl + action sft + plan sft rows) -> eval with the trained adapter.
set -e
export HOLD_NAME_RE='^(tcded_gpu|error_attr_hold_.*)$'
source scripts/lib_baselines.sh
PL=reports/textcraft/plan; SM=$PL/smoke; STU=$PL/sft/final; MA=reports/attributor_no_observations/final; rm -rf $SM; mkdir -p $SM
$(S 12000) $P -u scripts/textcraft_plan.py collect --adapter $STU --split test810 --offset 0 --n 4 --temperature 0 --out $SM/eval_s0.jsonl --tag smk_p0
$(S 12000) $P -u scripts/textcraft_plan.py collect --adapter $STU --split train_r1 --offset 0 --n 10 --temperature 1.0 --out $SM/traj_s0.jsonl --tag smk_pc
$P scripts/textcraft_round.py cover --glob "$SM/traj_s*.jsonl" --key task --split train_r1 --offset 0 --n 10 --out $SM/traj.jsonl
$P scripts/textcraft_plan.py plan_report $SM/eval_s0.jsonl $SM/traj.jsonl
$P scripts/textcraft_plan.py fails --traj $SM/traj.jsonl --out_dir $SM --first 10 --k_frac 0.2 --n_shards 2
for v in no_observations full; do mkdir -p $SM/modelA_$v; for i in 0 1; do [ -s $SM/release_s$i.jsonl ] && $(S 16000) $P -u scripts/alfworld_attribute.py --attributor $MA --variant $v --release $SM/release_s$i.jsonl --out $SM/modelA_$v/modelA_pred_s$i.jsonl; done; done
$P scripts/textcraft_plan.py variant --traj $SM/traj.jsonl --out_dir $SM
$P scripts/textcraft_round.py pointer --traj $SM/traj.jsonl --out_dir $SM
$(S 16000) $P -u scripts/textcraft_plan.py teacher --traj $SM/traj.jsonl --out_dir $SM --adapter $STU --shard 0 1
$P scripts/textcraft_round.py cover --glob "$SM/targets_s*.jsonl" --key idt --ids $SM/ids_fail.jsonl
$P -u scripts/textcraft_round.py onestep --traj $SM/traj.jsonl --out_dir $SM --shard 0 1
$P scripts/textcraft_round.py select --traj $SM/traj.jsonl --out_dir $SM --rule env_top_margin
$P scripts/textcraft_round.py replay_fw --traj $SM/traj.jsonl --out $SM/replay_actions.jsonl --n 6 --carry
$P scripts/textcraft_plan.py replay_plan --replay $SM/replay_actions.jsonl --traj $SM/traj.jsonl --out $SM/replay.jsonl
$(S 24000) $P -u scripts/textcraft_plan.py train --data $SM/targets_env_top_margin/train.jsonl:1.0 $SM/replay.jsonl:1.0 --adapter $PWD/$STU --lr 1e-5 --epochs 1 --batch 8 --seed 0 --out $SM/model
$(S 12000) $P -u scripts/textcraft_plan.py collect --adapter $SM/model/adapter --split test810 --offset 0 --n 2 --temperature 0 --out $SM/eval_r1_s0.jsonl --tag smk_p1
echo "PLAN SMOKE OK"
