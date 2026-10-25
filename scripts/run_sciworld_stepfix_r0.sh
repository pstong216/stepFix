#!/bin/bash
# Oct 9 22:50 ScienceWorld StepFix R0 (user-approved): (a) extend the r0fix train collection (student10, per_task 10, seed 1; tmux sw_r0c)
# to per_task 20 -- same seed => same shuffle; positions 10..19 per task only (--job_slice 10 20), 8 shards; (b) R0 greedy eval on the
# 651 test episodes (dev 197 already done: reports/sciworld/refix/eval_dev_safe.jsonl, 68/197).
source scripts/lib_sciworld_stepfix.sh; D=$SF/r0; mkdir -p $D
slog "r0: start extension collection (job_slice 10 20, 8 shards) + R0 test-651 eval (12 shards) on $(hostname)"
( collect_sw $D/ext $STU0 1 r0fix 8 --per_task 20 --job_slice 10 20; slog "r0: extension collection DONE ($(cat $D/ext/traj_s*.jsonl | grep -c .) episodes)" ) &
sleep 120
( LAUNCH_GAP=40 eval_sw $D $STU0 r0eval 1; slog "r0: test-651 eval DONE: $(grep -c '"won": true' $D/eval_test.jsonl)/$(grep -c . $D/eval_test.jsonl)" ) &
wait
