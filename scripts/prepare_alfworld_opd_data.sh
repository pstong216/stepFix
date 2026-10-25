#!/bin/bash
# After the SFT chain: Model A (7B no_obs) on the SFT student's failures (GPU hold $1), progress oracle + t* (CPU sbatch), step-A2 report.
J=$1; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=reports/alfworld/prepare_opd.log; T=reports/alfworld/sft3b_train_b50.jsonl
until grep -q "SFT chain finished" reports/alfworld/sft_chain.log 2>/dev/null; do sleep 60; done
echo "=== $(date) failures: $(grep -c '"won": false' $T) of $(wc -l < $T) episodes" | tee -a $L
# CPU: progress oracle on all failures, t* on 300
sbatch --export=ALL,TRAJ=$T,OUT=reports/alfworld/progress_sft3b_train_b50.jsonl,N=100000,EXTRA="--progress" slurm/cpu_oracle.sbatch | tee -a $L
sbatch --export=ALL,TRAJ=$T,OUT=reports/alfworld/oracle_sft3b_train_b50.jsonl,N=300 slurm/cpu_oracle.sbatch | tee -a $L
# GPU: Model A no_obs (and full) on the failures
[ "$(squeue -j $J -h -o %t 2>/dev/null)" = "R" ] || { echo "hold $J gone" | tee -a $L; exit 1; }
SR="srun --jobid=$J --overlap -n1 -c4 --mem=0"
$SR $P -u scripts/alfworld_attribute.py --attributor reports/attributor_7b_no_observations/final --variant no_observations --release reports/alfworld/sft3b_train_b50.release.jsonl --out reports/alfworld/attr7bnoobs_sft3b_train_b50.jsonl >> $L 2>&1
$SR $P -u scripts/alfworld_attribute.py --attributor reports/attributor_7b_full/final --release reports/alfworld/sft3b_train_b50.release.jsonl --out reports/alfworld/attr7bfull_sft3b_train_b50.jsonl >> $L 2>&1
echo "=== $(date) PREPARE_DONE" | tee -a $L
