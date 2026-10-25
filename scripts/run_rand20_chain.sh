#!/bin/bash
# Ablation (user Oct 8 13:30): "replay + random 20%" — identical to the final 20% chain except the selection: K = round(0.2 x failures)
# RANDOM failure states (teacher top-1 at the Model-A step, no environment check) + the same 100 replay. Shows that ranking by
# one-step goal-progress evidence (top 20%) matters. R1 reuses the final R1's states/teacher (a3b_nofilter/r1 all_top1 = 410 rows).
source scripts/lib_baselines.sh; C=a3b_rand20; D=$SA/$C/r1; mkdir -p $D
if [ ! -s $D/targets.jsonl ]; then $P -c "import random; L=open('$SA/a3b_nofilter/r1/targets_all_top1/train.jsonl').readlines(); random.Random(100).shuffle(L); open('$D/targets.jsonl','w').writelines(L[:82])"; fi
cp $SA/a3bfinal_k20/r1/replay.jsonl $D/replay.jsonl
log "$C r1: $(grep -c . $D/targets.jsonl) random teacher top-1 targets (20% of 410) + 100 replay"
train1 $D $STU "$D/targets.jsonl:1.0 $D/replay.jsonl:1.0"; [ -s $E/eval_heldout_a3brand20_r1.jsonl ] || eval774 $D a3brand20_r1
for R in 2 3; do bash scripts/run_k20_round.sh $C $R $SA/$C/r$((R-1))/model/adapter a3brand20_r$R 1 rand20 || { log "$C r$R FAILED"; exit 1; }
  [ -f $SA/$C/r$R/model/adapter/adapter_config.json ] || { log "$C r$R ended WITHOUT a model"; exit 1; }; done
log "$C R1->R3 DONE"
