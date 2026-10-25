#!/bin/bash
#SBATCH --job-name=error_attr_hold_l40s
#SBATCH --account=ee5239
#SBATCH --partition=interactive-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/users/8/pan00389/multi-agent/error-attribution/slurm/hold_gpu_l40s_%j.out
sleep 86400
