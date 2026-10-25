#!/bin/bash
# User-approved (Oct 8 ~21:00) diagnostic for the 7B R3 drop (576 -> 540, losses concentrated in heat/clean tasks): the old replay used the
# 50 SHORTEST wins (48 of 50 were simple pick-and-place / look tasks, 0 heat/clean steps). Here: same start (7B R2 adapter), the SAME 45
# corrections as a7b_final/r3, only the replay changed to RANDOM wins with their FULL trajectories (make_replay.py --mode random_full, 100
# steps, seed = same as the chain's R3 replay). Same trainer, 774 eval, tag a7b_r3rr.
export MODEL=Qwen/Qwen2.5-7B-Instruct GPU_MIN=20000 TRAIN_MIN=36000
source scripts/lib_baselines.sh; S3=$SA/a7b_final/r3; D=$SA/a7b_final/r3_randreplay; mkdir -p $D
[ -s $D/replay.jsonl ] || $P scripts/make_replay.py --traj $S3/traj.jsonl --n 100 --seed 3 --mode random_full --out $D/replay.jsonl | tee -a $SA/queue.log
train1 $D $SA/a7b_final/r2/model/adapter "$S3/targets_env_top/train.jsonl:1.0 $D/replay.jsonl:1.0"; [ -s $E/eval_heldout_a7b_r3rr.jsonl ] || eval774 $D a7b_r3rr
log "a7b r3_randreplay DONE (a7b_r3rr; original R3 = 540, R2 = 576)"
