#!/bin/bash
# HINT-SD ONLINE baseline (Oct 8; user-approved "option B": online loop in our own framework), 3B.  scripts/hintsd_online.py follows the official
# trainer (EMA-teacher hindsight analysis of failed rollouts, feedback turn, token-level reverse KL on top-100+tail, EMA 0.999, lr 5e-6,
# 2 tasks x 4 rollouts per optimizer step) on OUR budget: train games 0..1080 x 4 rollouts (~4324 episodes = our 3 rounds), start = SFT student.
# Adapters at 1323 / 2823 / 4324 episodes -> ckpt1..3; each is evaluated on the same 774 games as every other arm (tags hsdon_c1..c3) as soon
# as it exists (while training continues).  The training process resumes from DIR/state after a kill / hold-job expiry.
#   launch (cpu_tmux node):  tmux new -d -s hsdon "bash scripts/run_hintsd_online.sh"
source scripts/lib_baselines.sh; CH=a3b_hintsd_online; D=$SA/$CH; mkdir -p $D
HSD_MIN=${HSD_MIN:-24000}
log "$CH: start on $(hostname) (train -> $D, evals hsdon_c1..3)"
( until grep -q "^DONE hintsd_online" $D/train.log 2>/dev/null; do
    $(S $HSD_MIN | sed 's/-c4/-c8/') $P -u scripts/hintsd_online.py --out $D "$@" >> $D/train.log 2>&1 || { log "$CH: train exited (resume from state in 3 min)"; sleep 180; }
  done; log "$CH: training DONE $(grep '^SUMMARY' $D/train.log | tail -n 1)" ) &
for c in 1 2 3; do
  ( until [ -f $D/ckpt$c/adapter/adapter_config.json ] && [ -f $D/ckpt$c/adapter/adapter_model.safetensors ]; do sleep 300; done
    R=$D/r$c; mkdir -p $R/model; ln -sfn $PWD/$D/ckpt$c/adapter $R/model/adapter
    [ -s $E/eval_heldout_hsdon_c$c.jsonl ] || eval774 $R hsdon_c$c ) &
  sleep 5
done
wait
log "$CH: DONE (train + 3 evals)"
