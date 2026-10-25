#!/bin/bash
# Low-data student chain on hold $1: SFT ($2 trajectories, $3 epochs) -> greedy eval on unseen -> T=1 failure collection (train games 0-1499, budget 50) -> Model A no_obs -> progress oracle (CPU).
J=$1; NG=$2; EP=$3; TAG=student3b_low$NG; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; L=reports/alfworld/chain_$TAG.log; SR="srun --jobid=$J --overlap -n1 -c4 --mem=0"
echo "=== $(date) [1] SFT $TAG on $NG trajectories, $EP epochs (hold $J, $(squeue -j $J -h -o %N))" | tee -a $L
$SR $P -u scripts/alfworld_sft.py --traj reports/alfworld/expert_all_train_full.jsonl --max_games $NG --epochs $EP --model Qwen/Qwen2.5-3B-Instruct --out reports/alfworld/$TAG >> $L 2>&1 || { echo "SFT FAILED" | tee -a $L; exit 1; }
echo "=== $(date) [2] greedy eval on valid_unseen" | tee -a $L
$SR $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter reports/alfworld/$TAG/final --split valid_unseen --budget 50 --temperature 0 --out reports/alfworld/eval_unseen_$TAG.jsonl --tag $TAG >> $L 2>&1
echo "=== $(date) [3] T=1 failure collection, train games 0-1499" | tee -a $L
$SR $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter reports/alfworld/$TAG/final --split train --n_games 1500 --budget 50 --temperature 1.0 --out reports/alfworld/${TAG}_train_b50.jsonl --tag $TAG >> $L 2>&1
echo "=== $(date) [4] Model A (7B no_obs) on failures; progress oracle on CPU" | tee -a $L
sbatch --export=ALL,TRAJ=reports/alfworld/${TAG}_train_b50.jsonl,OUT=reports/alfworld/progress_${TAG}_train_b50.jsonl,N=100000,EXTRA="--progress" slurm/cpu_oracle.sbatch | tee -a $L
sbatch --export=ALL,TRAJ=reports/alfworld/${TAG}_train_b50.jsonl,OUT=reports/alfworld/oracle_${TAG}_train_b50.jsonl,N=300 slurm/cpu_oracle.sbatch | tee -a $L
$SR $P -u scripts/alfworld_attribute.py --attributor reports/attributor_7b_no_observations/final --variant no_observations --release reports/alfworld/${TAG}_train_b50.release.jsonl --out reports/alfworld/attr7bnoobs_${TAG}_train_b50.jsonl >> $L 2>&1
echo "=== $(date) CHAIN_DONE $TAG" | tee -a $L
