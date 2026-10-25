#!/bin/bash
#SBATCH --job-name=alf_cpu
#SBATCH --account=ee5239
#SBATCH --partition=msismall
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=reports/alfworld/cpu_%j.log
cd /users/8/pan00389/multi-agent/error-attribution
export ERROR_ATTR_ROOT=$PWD
eval "$CMD"
