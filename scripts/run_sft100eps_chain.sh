#!/bin/bash
# User-requested (Oct 8 ~21:20): "100 条 SFT + 20% correction" — the final recipe with the replay replaced by FULL-TRAJECTORY SFT on 100
# RANDOM own wins per round (make_replay.py --mode random_eps --n 100: every step of each win, ~1.4k steps) + the same top-20% one-step-
# evidence corrections (frozen 3B Model A step, hindsight self-teacher). R1 reuses the final chain's R1 corrections (same student, same
# states and candidates: a3bfinal_k20/r1/targets.jsonl, 82); R2/R3 run scripts/run_k20_round.sh with REPLAY_ARGS. 774 tags sft100e_r1..3.
source scripts/lib_baselines.sh; C=a3b_sft100eps; D=$SA/$C/r1; mkdir -p $D
[ -s $D/targets.jsonl ] || cp $SA/a3bfinal_k20/r1/targets.jsonl $D/targets.jsonl
[ -s $D/replay.jsonl ] || $P scripts/make_replay.py --traj $T1 --mode random_eps --n 100 --seed 1 --out $D/replay.jsonl | tee -a $SA/queue.log
log "$C r1: 82 corrections (= a3bfinal_k20 r1) + $(grep -c . $D/replay.jsonl) full-trajectory SFT steps from 100 random wins"
train1 $D $STU "$D/targets.jsonl:1.0 $D/replay.jsonl:1.0"; [ -s $E/eval_heldout_sft100e_r1.jsonl ] || eval774 $D sft100e_r1
for R in 2 3; do
  REPLAY_ARGS="--mode random_eps --n 100" bash -x scripts/run_k20_round.sh $C $R $SA/$C/r$((R-1))/model/adapter sft100e_r$R 1 > $SA/$C/r$R.driver.log 2>&1 || { log "$C r$R FAILED"; exit 1; }
  [ -f $SA/$C/r$R/model/adapter/adapter_config.json ] || { log "$C r$R ended WITHOUT a model"; exit 1; }
done
log "$C R1->R3 DONE"
