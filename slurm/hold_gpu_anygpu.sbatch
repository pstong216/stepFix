#!/bin/bash
#SBATCH --job-name=error_attr_hold_anygpu
#SBATCH --account=ee5239
#SBATCH --partition=a100-4,msigpu,preempt-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/users/8/pan00389/multi-agent/error-attribution/slurm/hold_gpu_anygpu_%j.out
sleep 86400
