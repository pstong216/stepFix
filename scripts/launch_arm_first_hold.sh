#!/bin/bash
# Usage: launch_arm_first_hold.sh <arm> <resume_dir_or_-> <jobid1> <jobid2> [extra attr_grpo args]
ARM=$1; RES=$2; J1=$3; J2=$4; shift 4; cd /users/8/pan00389/multi-agent/error-attribution
J=""; while [ -z "$J" ]; do for c in $J1 $J2; do [ "$(squeue -j $c -h -o %t 2>/dev/null)" = "R" ] && J=$c && break; done; [ -z "$J" ] && sleep 30; done
for c in $J1 $J2; do [ "$c" != "$J" ] && scancel $c; done
EXTRA=""; [ "$RES" != "-" ] && EXTRA="--resume_from $RES"
exec /users/8/pan00389/multi-agent/error-attribution/scripts/launch_arm_when_ready.sh $J $ARM $EXTRA "$@"
