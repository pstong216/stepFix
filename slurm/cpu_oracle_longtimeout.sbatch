#!/bin/bash
#SBATCH --job-name=alf_oracle
#SBATCH --account=ee5239
#SBATCH --partition=msismall
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=reports/alfworld/oracle_%j.log
cd /users/8/pan00389/multi-agent/error-attribution
export ERROR_ATTR_ROOT=$PWD
/users/8/pan00389/miniconda3/envs/error_attr/bin/python -u scripts/alfworld_oracle_label_longtimeout.py --traj ${TRAJ:-reports/alfworld/student3b_train.jsonl} --out ${OUT:-reports/alfworld/oracle_student3b_train.jsonl} --n ${N:-200} --procs 16 ${EXTRA:-}
