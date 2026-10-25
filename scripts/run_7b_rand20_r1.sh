#!/bin/bash
# User-requested (Oct 8 ~22:35): 7B "random 20% + the same 100 single-step replay", R1 only. Same R0 (7B SFT), same R1 collection / Model A
# states / hindsight-teacher candidates / replay file as the 7B final-recipe R1 (a7b_final/r1, 774 = 549); only the selection differs:
# K = 0.2 x failures random teacher-top-1 targets (rule rand20, seed 100). 774 tag a7brand20_r1.
export MODEL=Qwen/Qwen2.5-7B-Instruct GPU_MIN=20000 TRAIN_MIN=36000
source scripts/lib_baselines.sh; C=a7b_rand20; D=$SA/$C/r1; mkdir -p $D
rsync -a --exclude model --exclude targets_env_top --exclude 'eval_*' --exclude train.log $SA/a7b_final/r1/ $D/
bash -x scripts/run_k20_round.sh $C 1 $SA/a7b_final/r0/model/adapter a7brand20_r1 1 rand20 > $SA/$C/r1.driver.log 2>&1 || log "$C r1 FAILED"
