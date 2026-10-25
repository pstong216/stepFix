#!/bin/bash
# User-requested (Oct 9 ~00:15): 7B R3 trained from the ORIGINAL R2 (576) with the SAME 45 corrections as a7b_final/r3, replay = 100 steps
# split over task types in proportion to the agent's OWN training-collection failures of each type (make_replay.py --mode fail_weighted;
# R3 collection failures -> heat 25, two_obj 26, clean 22, cool 17, simple 7, look 2 steps), filled with complete trajectories of that type's
# shortest wins. Uses training data only (no test information). Same trainer, 774 eval, tag a7b_r3fw.
export MODEL=Qwen/Qwen2.5-7B-Instruct GPU_MIN=20000 TRAIN_MIN=36000
source scripts/lib_baselines.sh; S3=$SA/a7b_final/r3; D=$SA/a7b_final/r3_failw; mkdir -p $D
[ -s $D/replay.jsonl ] || $P scripts/make_replay.py --traj $S3/traj.jsonl --n 100 --seed 3 --mode fail_weighted --out $D/replay.jsonl | tee -a $SA/queue.log
train1 $D $SA/a7b_final/r2/model/adapter "$S3/targets_env_top/train.jsonl:1.0 $D/replay.jsonl:1.0"; [ -s $E/eval_heldout_a7b_r3fw.jsonl ] || eval774 $D a7b_r3fw
log "a7b r3_failw DONE (a7b_r3fw; original R3 = 540, R2 = 576, R3 random-full replay = 566)"
