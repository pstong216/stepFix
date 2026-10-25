#!/bin/bash
#SBATCH --job-name=error_attr_hold_a40
#SBATCH --account=ee5239
#SBATCH --partition=preempt-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/users/8/pan00389/multi-agent/error-attribution/slurm/hold_gpu_a40_%j.out
sleep 86400
