#!/bin/bash
# User-requested (Oct 6 22:30, "能不能看一轮没有100回放，只有40%更新的结果"): ablation D1 at R1 -- the final recipe's R1 with NO replay.
# Identical to selfattr_a3bfinal/r1 except the training set: the SAME 164 corrections (targets_env_top/train.jsonl), without the 100 replay
# steps. Same SFT start, trainer, hyper-parameters and 774 evaluation.
source scripts/lib_baselines.sh; D=$SA/a3bfinal_noreplay/r1; mkdir -p $D
cp $SA/selfattr_a3bfinal/r1/targets_env_top/train.jsonl $D/train.jsonl
log "a3bfinal_noreplay r1: $(grep -c . $D/train.jsonl) corrections, no replay (start $STU)"
train1 $D $STU "$D/train.jsonl:1.0"; [ -s $E/eval_heldout_a3bnorep_r1.jsonl ] || eval774 $D a3bnorep_r1
log "a3bfinal_noreplay r1 DONE"
