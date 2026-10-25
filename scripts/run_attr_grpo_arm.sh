#!/bin/bash
# Run one Attr-GRPO arm on a SLURM hold, then release the hold.
# Usage: run_attr_grpo_arm.sh <jobid> <arm> [extra attr_grpo.py args...]
set -uo pipefail
cd /users/8/pan00389/multi-agent/error-attribution
P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
J=$1; ARM=$2; shift 2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=reports/attr_grpo_3b/$ARM
mkdir -p $OUT
echo "=== $(date) arm=$ARM hold=$J node=$(squeue -j $J -h -o %N) args: $*" | tee -a $OUT/run.log
srun --jobid=$J --overlap -n1 -c4 --mem=0 $P -u scripts/attr_grpo.py --arm $ARM --output_dir $OUT "$@" >> $OUT/run.log 2>&1
echo "=== $(date) arm=$ARM finished (exit $?)" | tee -a $OUT/run.log
scancel $J
