#!/bin/bash
#SBATCH --job-name=error_attr_hold_cpu
#SBATCH --account=ee5239
#SBATCH --partition=interactive
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/users/8/pan00389/multi-agent/error-attribution/slurm/hold_cpu_%j.out
sleep 86400
