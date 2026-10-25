#!/bin/bash
#SBATCH --job-name=a7b_gpu
#SBATCH --account=qianwen
#SBATCH --partition=preempt-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/users/8/pan00389/multi-agent/error-attribution/slurm/a7b_gpu_%j.out
# Oct 9 19:15 (user: "如果有空的gpu 可以把之前7B的继续跑"): GPUs reserved for the paused 7B ALFWorld runs; 7B drivers use HOLD_NAME_RE with a7b_gpu,
# TextCraft drivers only match tcded_gpu|error_attr_hold_.*, so they never use these.
sleep 28800
