#!/bin/bash
# Smoke test of the TextCraft R1 pipeline on a handful of tasks (collect -> fails/release -> Model A -> pointer -> teacher -> one-step
# evidence -> env_top -> replay -> train -> eval -> compare).  No retry loops: any error stops the script.
set -e
source scripts/lib_baselines.sh
F=reports/textcraft/final/smoke; STU=reports/textcraft/student100/final; MA=reports/attributor_no_observations/final; mkdir -p $F/r0 $F/r1
[ -s $F/r1/traj_s0.jsonl ] || exit 1
[ -s $F/r1/traj_s1.jsonl ] || exit 1
$P scripts/textcraft_round.py cover --glob "$F/r1/traj_s*.jsonl" --key task --split train_r1 --offset 0 --n 12 --out $F/r1/traj.jsonl
$P scripts/textcraft_round.py fails --traj $F/r1/traj.jsonl --out_dir $F/r1 --first 10 --k_frac 0.2 --n_shards 2
for i in 0 1; do $(S 12000) $P -u scripts/alfworld_attribute.py --attributor $MA --variant no_observations --release $F/r1/release_s$i.jsonl --out $F/r1/modelA_pred_s$i.jsonl; done
$P scripts/textcraft_round.py cover --glob "$F/r1/modelA_pred_s*.jsonl" --key id --ids $F/r1/ids_fail.jsonl
$P scripts/textcraft_round.py pointer --traj $F/r1/traj.jsonl --out_dir $F/r1
for i in 0 1; do $(S 12000) $P -u scripts/textcraft_round.py teacher --traj $F/r1/traj.jsonl --out_dir $F/r1 --adapter $STU --shard $i 2; done
$P scripts/textcraft_round.py cover --glob "$F/r1/targets_s*.jsonl" --key idt --ids $F/r1/ids_fail.jsonl
for i in 0 1; do $P -u scripts/textcraft_round.py onestep --traj $F/r1/traj.jsonl --out_dir $F/r1 --shard $i 2; done
$P scripts/textcraft_round.py cover --glob "$F/r1/onestep_s*.jsonl" --key idt --ids $F/r1/ids_fail.jsonl
$P scripts/textcraft_round.py select --traj $F/r1/traj.jsonl --out_dir $F/r1
$P scripts/make_replay.py --traj $F/r1/traj.jsonl --n 4 --seed 1 --out $F/r1/replay.jsonl
$(S 24000) $P -u scripts/textcraft_round.py train --data $F/r1/targets_env_top/train.jsonl:1.0 $F/r1/replay.jsonl:1.0 --adapter $PWD/$STU --lr 1e-5 --epochs 1 --batch 8 --seed 0 --out $F/r1/model
$(S 12000) $P -u scripts/textcraft_agent.py collect --adapter $STU --split test810 --offset 0 --n 4 --temperature 0 --out $F/r0/eval_s0.jsonl --tag smk_r0
$(S 12000) $P -u scripts/textcraft_agent.py collect --adapter $F/r1/model/adapter --split test810 --offset 0 --n 4 --temperature 0 --out $F/r1/eval_s0.jsonl --tag smk_r1
$P scripts/textcraft_round.py compare --r0 $F/r0/eval_s0.jsonl --r1 $F/r1/eval_s0.jsonl
echo "SMOKE OK"
