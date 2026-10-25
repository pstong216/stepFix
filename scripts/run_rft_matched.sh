#!/bin/bash
# User-approved (Oct 6 21:35) baseline B1, RFT matched size = self-imitation only: each round, SFT on N (state, taken action) steps from the
# student's OWN shortest wins of that round (make_replay.py: 2 steps per win, skipping look/inventory/help), with N equal to the final
# method's training-set size in the same round (R1 264 = 164 corrections + 100 replay; R2 210 = 110 + 100). No corrections, no Model A,
# no teacher. Same collections (R1 = the SFT student's 1323-game collection, R2 = 1500 train games T=1.0), trainer and 774 evaluation.
source scripts/lib_baselines.sh; C=$SA/rft_matched
D=$C/r1; mkdir -p $D
[ -s $D/train.jsonl ] || $P scripts/make_replay.py --traj $T1 --n 264 --seed 1 --out $D/train.jsonl | tee -a $SA/queue.log
train1 $D $STU "$D/train.jsonl:1.0"; [ -s $E/eval_heldout_rftm_r1.jsonl ] || eval774 $D rftm_r1
D=$C/r2; mkdir -p $D; R1A=$C/r1/model/adapter
collect4 $D $R1A rft_matched_r2
log "rft_matched r2: $(grep -c . $D/traj.jsonl) games, $(grep -c '"won": true' $D/traj.jsonl) won"
[ -s $D/train.jsonl ] || $P scripts/make_replay.py --traj $D/traj.jsonl --n 210 --seed 2 --out $D/train.jsonl | tee -a $SA/queue.log
train1 $D $R1A "$D/train.jsonl:1.0"; [ -s $E/eval_heldout_rftm_r2.jsonl ] || eval774 $D rftm_r2
log "rft_matched R1->R2 DONE"
