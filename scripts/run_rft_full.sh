#!/bin/bash
# User-approved (Oct 8 ~21:00): STANDARD RFT baseline (STaR / RFT, Yuan et al. 2023): each round, SFT on EVERY step of EVERY successful
# trajectory of that round's own collection (make_replay.py --mode all). No corrections, no Model A, no teacher. Same collections as the
# final method (R1 = the SFT student's 1323-game collection; R2/R3 = 1500 train games, T=1.0, with the previous RFT model), same trainer
# (lr 1e-5, batch 8, 1 epoch, seed 0) and the same 774 evaluation. Equal environment budget; far more training samples than ours.
source scripts/lib_baselines.sh; C=$SA/rft_full
D=$C/r1; mkdir -p $D
[ -s $D/train.jsonl ] || $P scripts/make_replay.py --traj $T1 --mode all --seed 1 --out $D/train.jsonl | tee -a $SA/queue.log
train1 $D $STU "$D/train.jsonl:1.0"; [ -s $E/eval_heldout_rftfull_r1.jsonl ] || eval774 $D rftfull_r1
for R in 2 3; do
  PA=$C/r$((R-1))/model/adapter; D=$C/r$R; mkdir -p $D
  collect4 $D $PA rft_full_r$R
  log "rft_full r$R: $(grep -c . $D/traj.jsonl) games, $(grep -c '"won": true' $D/traj.jsonl) won"
  [ -s $D/train.jsonl ] || $P scripts/make_replay.py --traj $D/traj.jsonl --mode all --seed $R --out $D/train.jsonl | tee -a $SA/queue.log
  train1 $D $PA "$D/train.jsonl:1.0"; [ -s $E/eval_heldout_rftfull_r$R.jsonl ] || eval774 $D rftfull_r$R
  [ -f $D/model/adapter/adapter_config.json ] || { log "rft_full r$R ended WITHOUT a model"; exit 1; }
done
log "rft_full R1->R3 DONE"
