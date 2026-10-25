#!/bin/bash
# User-approved (Oct 6 23:40, "20% 纠正 + 100 回放 的 R1 马上排上"): final recipe R1 with K = round(0.2 x 410 failures) = 82 corrections
# (the top 82 of the final R1 ranking; targets_env_top/train.jsonl is stored in rank order) + the SAME 100 replay rows; SFT start, new trainer,
# lr 1e-5, batch 8, 1 epoch; 774 evaluation.  Compare: 40% + 100 replay = 547, 40% no replay = 440, ~20% (top-86) no replay = D-c.
source scripts/lib_baselines.sh; F=$SA/selfattr_a3bfinal/r1; D=$SA/a3bfinal_k20/r1; mkdir -p $D
head -n 82 $F/targets_env_top/train.jsonl > $D/targets.jsonl; cp $F/replay.jsonl $D/replay.jsonl
log "a3bfinal_k20 r1: $(grep -c . $D/targets.jsonl) corrections (20%) + $(grep -c . $D/replay.jsonl) replay"
train1 $D $STU "$D/targets.jsonl:1.0 $D/replay.jsonl:1.0"; [ -s $E/eval_heldout_a3bk20_r1.jsonl ] || eval774 $D a3bk20_r1
log "a3bfinal_k20 r1 DONE"
