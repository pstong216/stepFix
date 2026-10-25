#!/bin/bash
#SBATCH --job-name=error_attr_drivers
#SBATCH --account=ee5239
#SBATCH --partition=interactive
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/users/8/pan00389/multi-agent/error-attribution/slurm/relaunch_drivers_%j.out
# Oct 11 19:40: the cpu_tmux job (2995731, cn1019) ends ~22:39 and its tmux drivers die with it. This job waits until 2995731 is gone,
# then relaunches every still-needed driver in tmux on THIS node (all drivers resume from their outputs), then holds the node.
cd /users/8/pan00389/multi-agent/error-attribution; S=reports/sciworld/stepfix
echo "$(date) waiting for 2995731 to end on $(hostname)"
while squeue -h -j 2995731 2>/dev/null | grep -q .; do sleep 60; done; sleep 30
echo "$(date) relaunching drivers on $(hostname)"
for a in "abl_rand rand" "abl_nofil nofil" "abl_norep norep" "abl_rftm rftm" "abl_roll roll"; do set -- $a
  grep -q "$1 chain DONE" $S/queue.log || { tmux new-session -d -s sf_ch_$1 "bash scripts/run_sciworld_abl_chain.sh $1 $2 2>&1 | tee -a $S/$1_chain.log"; sleep 20; }; done
grep -q "hsdon_c3: 848 eval DONE\|hsdon_c3 DONE" $S/queue.log || tmux new-session -d -s sw_hsd_online "bash scripts/run_sciworld_hintsd_online.sh >> $S/hsdon_driver.log 2>&1"
grep -q "7B R3 (w7_3" reports/sciworld/stepfix7b/queue.log || tmux new-session -d -s sw7b_r3 "bash scripts/run_sciworld_7b_r3.sh 2>&1 | tee -a reports/sciworld/stepfix7b/r3_driver.log"
tmux ls; echo "- $(date +%H:%M) drivers relaunched on $(hostname) by job $SLURM_JOB_ID (cpu_tmux 2995731 ended)" >> docs/status_2026-10-10_handoff.md
sleep 86000
