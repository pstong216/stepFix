#!/bin/bash
#SBATCH --job-name=error_attr_hold_a100x2
#SBATCH --account=ee5239
#SBATCH --partition=a100-4
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/users/8/pan00389/multi-agent/error-attribution/slurm/hold_gpu_a100x2_%j.out
sleep 86400
