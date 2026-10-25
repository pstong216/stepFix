#!/bin/bash
# User-requested (Oct 7 19:20): "20% corrections only" chain (no replay), R1 -> R2.
# R1 = the SAME 82 corrections as a3bfinal_k20/r1 (top 20% of the final R1 ranking) WITHOUT the 100 replay rows; SFT start, new trainer.
# R2 = final-recipe round with K = 20% and no replay (scripts/run_k20_round.sh ... 0).
source scripts/lib_baselines.sh; D=$SA/a3bk20_norep/r1; mkdir -p $D
[ -s $D/train.jsonl ] || cp $SA/a3bfinal_k20/r1/targets.jsonl $D/train.jsonl
log "a3bk20_norep r1: $(grep -c . $D/train.jsonl) corrections (20%), no replay"
train1 $D $STU "$D/train.jsonl:1.0"; [ -s $E/eval_heldout_a3bk20nr_r1.jsonl ] || eval774 $D a3bk20nr_r1
bash scripts/run_k20_round.sh a3bk20_norep 2 $SA/a3bk20_norep/r1/model/adapter a3bk20nr_r2 0
