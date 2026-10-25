#!/bin/bash
# User decision (Oct 8 ~21:50): "把随机回放并入最终方法，只需要重新跑 7B 的 R2 和 R3". 7B chain with replay = RANDOM own wins, full trajectories,
# 100 steps (make_replay.py --mode random_full), everything else = final recipe. R1 is the existing a7b_final/r1 (old shortest-wins replay).
# R2 reuses a7b_final/r2's collection / Model A / teacher / one-step evidence / env_top corrections (independent of the replay; copied into
# a7b_rr/r2), so only its replay and training are new. R3 is a full new round from the new R2 model. 774 tags a7b_rr_r2, a7b_rr_r3.
export MODEL=Qwen/Qwen2.5-7B-Instruct GPU_MIN=20000 TRAIN_MIN=36000
source scripts/lib_baselines.sh; C=a7b_rr
for R in 2 3; do
  START=$([ $R = 2 ] && echo $SA/a7b_final/r1/model/adapter || echo $SA/$C/r2/model/adapter)
  [ -f $SA/$C/r$R/model/adapter/adapter_config.json ] && [ -s $E/eval_heldout_a7b_rr_r$R.jsonl ] && continue
  REPLAY_ARGS="--mode random_full --n 100" bash -x scripts/run_k20_round.sh $C $R $START a7b_rr_r$R 1 > $SA/$C/r$R.driver.log 2>&1 || { log "$C r$R FAILED"; exit 1; }
  [ -f $SA/$C/r$R/model/adapter/adapter_config.json ] || { log "$C r$R ended WITHOUT a model"; exit 1; }
done
log "$C R2->R3 DONE (7B, random full-trajectory replay)"
