#!/bin/bash
# Oct 11 21:50 (user: baselines to R3 budget): extend the ScienceWorld SERL run from its step-48 checkpoint to step 72 (4,608 episodes,
# ~ our R3 budget 4,449), same run/config (constant LR, resume_mode=auto), then evaluate step 72 with our 848 protocol.
# Needs a WHOLE GPU job named sw_rl_gpu (vLLM + FSDP). Waits for one.
cd /users/8/pan00389/multi-agent/error-attribution; W=/scratch.global/pan00389/error-attribution/serl_work
until JOB=$(squeue -u pan00389 -h -t RUNNING -o "%i %j" | awk '$2=="sw_rl_gpu"{print $1; exit}'); [ -n "$JOB" ]; do sleep 120; done
echo "=== $(date '+%F %T') serl72: training on job $JOB" | tee -a $W/outputs/sciworld_serl/driver.log
JOB=$JOB TOTAL_STEPS=72 SAVE_STEPS='[48,56,64,72]' bash $W/run_serl_sciworld_job_long.sh
bash scripts/sciworld_rl_eval.sh $W/outputs/sciworld_serl 72 serl_s72
