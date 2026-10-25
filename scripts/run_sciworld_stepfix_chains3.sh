#!/bin/bash
# Oct 10 12:00 (user-approved): (H) hybrid env_top K=0.2 + replay_fw + full-view teacher, R2 -> R3 (r2h, r3h);
# (K40) same with rho = 0.4 (r2k, r3k). R2 reuses r2's collection / Model A pointer and r2f's full-view teacher rows (copied into r2h / r2k).
# (rollD) ablation D extended to R2 / R3 with the refined recipe (abl_roll_r2, abl_roll_r3), lowest priority.
#   bash scripts/run_sciworld_stepfix_chains3.sh H|K40|rollD
source scripts/lib_sciworld_stepfix.sh; W=$1; RF=scripts/run_sciworld_stepfix_refined.sh
wait_adapter() { until [ -f $1/adapter_config.json ]; do sleep 60; done; }
REFS2="r1b=$SF/r1b/eval_stepfix r2=$SF/r2/eval_stepfix r2f=$SF/r2f/eval"
if [ "$W" = H ] || [ "$W" = K40 ]; then
  [ "$W" = H ] && { export RHO=0.2; A=r2h; B=r3h; } || { export RHO=0.4; A=r2k; B=r3k; }
  GMEM=9000 bash $RF $A 2 $SF/r1b/stepfix/adapter envtop 1 "" "$REFS2" > $SF/${A}_driver.log 2>&1 &
  wait_adapter $SF/$A/model/adapter
  GMEM=9000 bash $RF $B 3 $SF/$A/model/adapter envtop 1 "" "r3f=$SF/r3f/eval r3=$SF/r3/eval_stepfix $A=$SF/$A/eval r1b=$SF/r1b/eval_stepfix" > $SF/${B}_driver.log 2>&1 &
  wait
elif [ "$W" = rollD ]; then
  export GMEM=16000 TGMEM=18000 TRMEM=26000
  wait_adapter $SF/abl_roll_r1/model/adapter
  bash $RF abl_roll_r2 2 $SF/abl_roll_r1/model/adapter roll 1 "" "r2f=$SF/r2f/eval abl_roll_r1=$SF/abl_roll_r1/eval" > $SF/abl_roll_r2_driver.log 2>&1 &
  wait_adapter $SF/abl_roll_r2/model/adapter
  bash $RF abl_roll_r3 3 $SF/abl_roll_r2/model/adapter roll 1 "" "r3f=$SF/r3f/eval abl_roll_r2=$SF/abl_roll_r2/eval" > $SF/abl_roll_r3_driver.log 2>&1 &
  wait
fi
slog "chains3 $W: finished"
