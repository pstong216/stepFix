#!/bin/bash
#SBATCH --job-name=error_attr_hold_igpu_qw
#SBATCH --account=qianwen
#SBATCH --partition=interactive-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/users/8/pan00389/multi-agent/error-attribution/slurm/hold_gpu_igpu_%j.out
sleep 86400
