#!/bin/bash
# User-requested (Oct 8 ~20:55): the SFT-only part of "100 wins full SFT + 20% corrections" — same 1437 steps (every step of the same 100
# random wins of the SFT student's collection, a3b_sft100eps/r1/replay.jsonl), NO corrections, from the SFT student, same trainer, 774 eval.
source scripts/lib_baselines.sh; S1=$SA/a3b_sft100eps/r1; D=$SA/a3b_sft100eps/r1_sftonly; mkdir -p $D
until [ -s $S1/replay.jsonl ]; do sleep 30; done
train1 $D $STU "$S1/replay.jsonl:1.0"; [ -s $E/eval_heldout_sft100only_r1.jsonl ] || eval774 $D sft100only_r1
log "a3b_sft100eps r1_sftonly DONE (sft100only_r1: 100 random wins, full-trajectory SFT, no corrections)"
