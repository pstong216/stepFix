#!/bin/bash
# User-requested (Oct 9 ~09:20): ONLINE HINT-SD baseline for the 7B student. Same trainer as the 3B run (scripts/hintsd_online.py, official
# algorithm: EMA-teacher hindsight analysis, feedback turn, token-level reverse KL top-100+tail, EMA 0.999, lr 5e-6, 2 tasks x 4 rollouts per
# step) on the 7B chain's budget: 1125 train games x 4 rollouts = 4500 episodes (= the 7B chain's 1500 + 1500 + 1500), start = the 7B SFT
# student (a7b_final/r0). Checkpoints at 1500 / 3000 / 4500 episodes -> 774 eval, tags h7on_c1..c3.
export MODEL=Qwen/Qwen2.5-7B-Instruct GPU_MIN=20000 TRAIN_MIN=36000
source scripts/lib_baselines.sh; CH=a7b_hintsd_online; D=$SA/$CH; mkdir -p $D; HSD_MIN=${HSD_MIN:-36000}
log "$CH: start on $(hostname) (7B online HINT-SD -> $D, evals h7on_c1..3)"
( until grep -q "^DONE hintsd_online" $D/train.log 2>/dev/null; do
    $(S $HSD_MIN | sed 's/-c4/-c8/') $P -u scripts/hintsd_online.py --out $D --model $MODEL --adapter $SA/a7b_final/r0/model/adapter --n_tasks 1125 --ckpt_at 1500,3000 >> $D/train.log 2>&1 || { log "$CH: train exited (resume from state in 3 min)"; sleep 180; }
  done; log "$CH: training DONE $(grep '^SUMMARY' $D/train.log | tail -n 1)" ) &
for c in 1 2 3; do
  ( until [ -f $D/ckpt$c/adapter/adapter_config.json ] && [ -f $D/ckpt$c/adapter/adapter_model.safetensors ]; do sleep 300; done
    R=$D/r$c; mkdir -p $R/model; ln -sfn $PWD/$D/ckpt$c/adapter $R/model/adapter
    [ -s $E/eval_heldout_h7on_c$c.jsonl ] || eval774 $R h7on_c$c ) &
  sleep 5
done
wait
