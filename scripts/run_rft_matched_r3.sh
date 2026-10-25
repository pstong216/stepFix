#!/bin/bash
# RFT matched R3 (Oct 8; fills the results table). Same rule as R1/R2 of this chain: N = the 40% chain's training-set size in the same
# round (R3 = 102 corrections + 100 replay = 202 own-win steps). Collection with the RFT R2 model.
source scripts/lib_baselines.sh; C=$SA/rft_matched; D=$C/r3; mkdir -p $D; R2A=$C/r2/model/adapter
collect4 $D $R2A rft_matched_r3
[ -s $D/train.jsonl ] || $P scripts/make_replay.py --traj $D/traj.jsonl --n 202 --seed 3 --out $D/train.jsonl | tee -a $SA/queue.log
train1 $D $R2A "$D/train.jsonl:1.0"; [ -s $E/eval_heldout_rftm_r3.jsonl ] || eval774 $D rftm_r3
log "rft_matched r3 DONE"
