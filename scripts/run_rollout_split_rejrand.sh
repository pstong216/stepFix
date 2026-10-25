#!/bin/bash
# Oct 12 00:40 (user): ALFWorld control for the rollout split: 82 RANDOM rollout-rejected candidates (teacher top-2 per R1 state, no
# evidence ranking; states absent from the rollout examples count as all-rejected, as in run_rollout_split.sh) + the SAME 100 replay
# steps, same trainer/seed, from the SFT student, 774 eval. Compare with rej82 (top-82 by one-step evidence among rejected) = 498.
source scripts/lib_baselines.sh; export EVIDENCE=generic
X=$SA/a3b_rolloutsplit; REP=$SA/a3bfinal_k20/r1/replay.jsonl; A=rejrand82
train1 $X/$A $STU "$X/$A/targets.jsonl:1.0 $REP:1.0"; [ -s $E/eval_heldout_rsplit_$A.jsonl ] || eval774 $X/$A rsplit_$A; log "rollout-split $A DONE (rsplit_$A)"
