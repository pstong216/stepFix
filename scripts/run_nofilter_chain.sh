#!/bin/bash
# User-requested (Oct 7 19:45) baseline: NO FILTER, update on ALL failures. Final recipe except the selection: at every Model-A-attributed
# failure state, the hindsight teacher's top-1 action (a- excluded) becomes a one-hot target (no environment check, no budget K), + 100 replay.
# R1 = the final R1's 410 states/candidates (selfattr_a3bfinal/r1) -> 410 targets + the same 100 replay rows; R2 via run_k20_round.sh ... all_top1.
source scripts/lib_baselines.sh; D=$SA/a3b_nofilter/r1; mkdir -p $D
[ -s $D/replay.jsonl ] || cp $SA/a3bfinal_k20/r1/replay.jsonl $D/replay.jsonl
log "a3b_nofilter r1: $(grep -c . $D/targets_all_top1/train.jsonl) corrections (all failures, no filter) + $(grep -c . $D/replay.jsonl) replay"
train1 $D $STU "$D/targets_all_top1/train.jsonl:1.0 $D/replay.jsonl:1.0"; [ -s $E/eval_heldout_a3bnf_r1.jsonl ] || eval774 $D a3bnf_r1
bash scripts/run_k20_round.sh a3b_nofilter 2 $SA/a3b_nofilter/r1/model/adapter a3bnf_r2 1 all_top1
