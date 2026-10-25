#!/bin/bash
# Sequential queue on the non-preemptible igpu hold (id read from reports/alfworld/igpu_hold.txt before each task).
cd /users/8/pan00389/multi-agent/error-attribution
runner() { case $1 in const1_r2|attralf_r3) echo scripts/alfworld_task2.sh;; *) echo scripts/alfworld_task.sh;; esac; }
for t in attr_nofb_p1 attr_nofb_p2 attralf_r3 const1 r3_fin seed1 cv; do
  H=$(cat reports/alfworld/igpu_hold.txt); until [ "$(squeue -j $H -h -o %t 2>/dev/null)" = R ]; do sleep 60; H=$(cat reports/alfworld/igpu_hold.txt); done
  $(runner $t) $t $H
done
echo "=== $(date) igpu queue: done" | tee -a reports/alfworld/dispatch.log
