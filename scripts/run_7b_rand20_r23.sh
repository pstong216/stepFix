#!/bin/bash
# User-requested (Oct 9 ~00:50): continue the 7B "random 20% + replay" ablation to R2 and R3 with the SAME protocol as the 7B final-recipe
# chain it is compared with (a7b_final: R1 549, R2 576, R3 540): scripts/run_k20_round.sh, rule rand20, default (shortest-wins) replay,
# frozen 3B Model A, 7B hindsight self-teacher. Tags a7brand20_r2, a7brand20_r3.
export MODEL=Qwen/Qwen2.5-7B-Instruct GPU_MIN=20000 TRAIN_MIN=36000
source scripts/lib_baselines.sh; C=a7b_rand20
for R in 2 3; do
  [ -f $SA/$C/r$R/model/adapter/adapter_config.json ] && [ -s $E/eval_heldout_a7brand20_r$R.jsonl ] && continue
  bash -x scripts/run_k20_round.sh $C $R $SA/$C/r$((R-1))/model/adapter a7brand20_r$R 1 rand20 > $SA/$C/r$R.driver.log 2>&1 || { log "$C r$R FAILED"; exit 1; }
  [ -f $SA/$C/r$R/model/adapter/adapter_config.json ] || { log "$C r$R ended WITHOUT a model"; exit 1; }
done
log "$C R2->R3 DONE"
