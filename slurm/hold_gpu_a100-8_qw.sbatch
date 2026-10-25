#!/bin/bash
#SBATCH --job-name=error_attr_hold_a100-8_qw
#SBATCH --account=qianwen
#SBATCH --partition=a100-8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --exclude=aga36,aga43
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/users/8/pan00389/multi-agent/error-attribution/slurm/hold_gpu_igpu_%j.out
sleep 86400
