#!/bin/bash
# Oct 10 14:05 (user-approved): 3-step window evidence variant (hybrid env_top by S_3, K = 0.2 x failures, replay_fw, full-view teacher).
# R2 (r2s3) reuses r2's collection / pointer and r2f's full-view teacher rows; R3 (r3s3) collects with r2s3.
source scripts/lib_sciworld_stepfix.sh; RF=scripts/run_sciworld_stepfix_refined.sh; export RHO=0.2 EVID=window GMEM=9000
wait_adapter() { until [ -f $1/adapter_config.json ]; do sleep 60; done; }
bash $RF r2s3 2 $SF/r1b/stepfix/adapter envtop 1 "" "r1b=$SF/r1b/eval_stepfix r2h=$SF/r2h/eval r2=$SF/r2/eval_stepfix r2f=$SF/r2f/eval" > $SF/r2s3_driver.log 2>&1 &
wait_adapter $SF/r2s3/model/adapter
bash $RF r3s3 3 $SF/r2s3/model/adapter envtop 1 "" "r3h=$SF/r3h/eval r2s3=$SF/r2s3/eval r1b=$SF/r1b/eval_stepfix r3f=$SF/r3f/eval" > $SF/r3s3_driver.log 2>&1 &
wait; slog "chains4 S3: finished"
