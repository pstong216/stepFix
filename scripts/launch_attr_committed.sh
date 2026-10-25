#!/bin/bash
# Waiter: when hold $1 starts, pick generation budget by GPU memory and run the attr_committed arm.
J=$1; cd /users/8/pan00389/multi-agent/error-attribution
until [ "$(squeue -j $J -h -o %t 2>/dev/null)" = "R" ]; do sleep 30; done
sleep 20
MEM=$(srun --jobid=$J --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | tail -1 | tr -d ' ')
echo "$(date) hold $J running on $(squeue -j $J -h -o %N), GPU memory $MEM MiB" | tee -a reports/attr_grpo_3b/launch_attr_committed.log
if [ -n "$MEM" ] && [ "$MEM" -gt 70000 ]; then GB=128; AB=3e8; else GB=64; AB=0.8e8; fi
COMMON="--group 8 --tasks_per_step 8 --min_informative 4 --max_resample 2 --steps 48 --eval_every 12 --eval_subset 60 --final_samples 4 --lr 2e-5 --beta 0.04"
exec /users/8/pan00389/multi-agent/error-attribution/scripts/run_attr_grpo_arm.sh $J attr_committed $COMMON --gen_batch $GB --attn_budget $AB
