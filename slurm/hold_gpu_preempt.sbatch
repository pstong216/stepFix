#!/bin/bash
#SBATCH --job-name=error_attr_hold_preempt
#SBATCH --account=ee5239
#SBATCH --partition=preempt-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/users/8/pan00389/multi-agent/error-attribution/slurm/hold_gpu_preempt_%j.out
# Preemptible: use ONLY for resumable, per-batch-flushed jobs (self-play collection with --resume).
sleep 86400
