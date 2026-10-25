#!/bin/bash
# User decision (Oct 9 ~00:25): 7B R3 from the ORIGINAL R2 (576) with BOTH improvements, everything else = final recipe:
#  (1) corrections: env_top_margin = env_top among candidates whose one-step evidence STRICTLY exceeds the taken action's (K = 45, same
#      a7b_final/r3 collection, Model A states, teacher candidates and one-step evidence);
#  (2) replay: make_replay.py --mode fail_weighted (100 steps split over task types by the agent's OWN training failures; complete
#      trajectories of each type's shortest wins). Same trainer, 774 eval, tag a7b_r3v2.
export MODEL=Qwen/Qwen2.5-7B-Instruct GPU_MIN=20000 TRAIN_MIN=36000 EVIDENCE=generic
source scripts/lib_baselines.sh; S3=$SA/a7b_final/r3; D=$SA/a7b_final/r3_v2; mkdir -p $D
[ -s $D/targets_env_top_margin/train.jsonl ] || SE_DIR=$S3 SE_TRAJ=$S3/traj.jsonl $P scripts/proxy_arms_se3b.py --arm self --rule env_top_margin --k 45 --out $D/targets_env_top_margin | tee -a $SA/queue.log
[ -s $D/replay.jsonl ] || $P scripts/make_replay.py --traj $S3/traj.jsonl --n 100 --seed 3 --mode fail_weighted --out $D/replay.jsonl | tee -a $SA/queue.log
train1 $D $SA/a7b_final/r2/model/adapter "$D/targets_env_top_margin/train.jsonl:1.0 $D/replay.jsonl:1.0"; [ -s $E/eval_heldout_a7b_r3v2.jsonl ] || eval774 $D a7b_r3v2
log "a7b r3_v2 DONE (a7b_r3v2: margin rule + fail-weighted replay; original R3 = 540, R2 = 576)"
