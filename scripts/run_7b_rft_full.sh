#!/bin/bash
# User-requested (Oct 9 ~09:20): STANDARD RFT baseline for the 7B student: each round, SFT on EVERY step of EVERY win of that round's own
# collection (make_replay.py --mode all). R1 = the 7B SFT student's 1500-game collection (a7b_final/r1/traj.jsonl, the same data the 7B
# final-recipe R1 started from); R2/R3 = 1500 train games (T=1.0) with the previous RFT model. Same trainer and 774 eval. Tags a7brft_r1..r3.
export MODEL=Qwen/Qwen2.5-7B-Instruct GPU_MIN=20000 TRAIN_MIN=36000
source scripts/lib_baselines.sh; C=$SA/a7b_rft_full; S0=$SA/a7b_final/r0/model/adapter
D=$C/r1; mkdir -p $D
[ -s $D/train.jsonl ] || $P scripts/make_replay.py --traj $SA/a7b_final/r1/traj.jsonl --mode all --seed 1 --out $D/train.jsonl | tee -a $SA/queue.log
train1 $D $S0 "$D/train.jsonl:1.0"; [ -s $E/eval_heldout_a7brft_r1.jsonl ] || eval774 $D a7brft_r1
for R in 2 3; do
  PA=$C/r$((R-1))/model/adapter; D=$C/r$R; mkdir -p $D
  collect4 $D $PA a7b_rft_full_r$R
  log "a7b_rft_full r$R: $(grep -c . $D/traj.jsonl) games, $(grep -c '"won": true' $D/traj.jsonl) won"
  [ -s $D/train.jsonl ] || $P scripts/make_replay.py --traj $D/traj.jsonl --mode all --seed $R --out $D/train.jsonl | tee -a $SA/queue.log
  train1 $D $PA "$D/train.jsonl:1.0"; [ -s $E/eval_heldout_a7brft_r$R.jsonl ] || eval774 $D a7brft_r$R
  [ -f $D/model/adapter/adapter_config.json ] || { log "a7b_rft_full r$R ended WITHOUT a model"; exit 1; }
done
log "a7b_rft_full R1->R3 DONE"
