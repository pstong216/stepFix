#!/bin/bash
#SBATCH --job-name=tcded_gpu
#SBATCH --account=qianwen
#SBATCH --partition=preempt-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/users/8/pan00389/multi-agent/error-attribution/slurm/tcded_gpu_%j.out
# Oct 9 (user): a GPU reserved for the TextCraft runs only (drivers relaunched with HOLD_NAME_RE='^(tcded_gpu|error_attr_hold_.*)$';
# other drivers only match ^error_attr_hold_, so they never use it).
sleep 28800
