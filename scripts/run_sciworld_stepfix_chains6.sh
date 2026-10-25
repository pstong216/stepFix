#!/bin/bash
# Oct 10 15:35 (user-approved): next-subgoal relative evidence (H = 9 = p90 of the gaps between score increases in TRAIN gold demos).
# R2 (r2g) reuses r2's collection / pointer and r2f's full-view teacher candidates; R3 (r3g) ONLY if r2g beats r2h (decided by hand).
source scripts/lib_sciworld_stepfix.sh; RF=scripts/run_sciworld_stepfix_refined.sh; export RHO=0.2 GMEM=9000
bash $RF r2g 2 $SF/r1b/stepfix/adapter subgoal 1 "" "r1b=$SF/r1b/eval_stepfix r2h=$SF/r2h/eval r2s3=$SF/r2s3/eval" > $SF/r2g_driver.log 2>&1
slog "chains6 G: r2g finished (R3 only on decision)"
