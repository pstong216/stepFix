#!/bin/bash
# Oct 10 14:40 (user/coordinator): variant B "affordance gain": S_A = S_3 + 1.0 x aff (window creates new goal-related valid actions);
# hybrid env_top by S_A, K = 0.2 x failures, replay_fw, full-view teacher. R2 (r2a) reuses r2's collection / pointer / r2f teacher rows.
source scripts/lib_sciworld_stepfix.sh; RF=scripts/run_sciworld_stepfix_refined.sh; export RHO=0.2 EVID=window_aff GMEM=9000
wait_adapter() { until [ -f $1/adapter_config.json ]; do sleep 60; done; }
bash $RF r2a 2 $SF/r1b/stepfix/adapter envtop 1 "" "r1b=$SF/r1b/eval_stepfix r2h=$SF/r2h/eval r2s3=$SF/r2s3/eval" > $SF/r2a_driver.log 2>&1 &
wait_adapter $SF/r2a/model/adapter
bash $RF r3a 3 $SF/r2a/model/adapter envtop 1 "" "r3h=$SF/r3h/eval r3s3=$SF/r3s3/eval r2a=$SF/r2a/eval r1b=$SF/r1b/eval_stepfix" > $SF/r3a_driver.log 2>&1 &
wait; slog "chains5 A: finished"
