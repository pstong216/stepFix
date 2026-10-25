#!/bin/bash
# Returning-hold dispatcher: one task per preempt/safe hold as they (re)start; locked tasks are skipped.
cd /users/8/pan00389/multi-agent/error-attribution; HOLDS="1436314 1435620 1502947 1521595 1554117"; declare -A USED
runner() { case $1 in const1_r2|attralf_r3) echo scripts/alfworld_task2.sh;; *) echo scripts/alfworld_task.sh;; esac; }
for t in attr_nofb_p1 attr_nofb_p2 attralf_r3 const1 r3_fin seed1 cv; do
  [ -d reports/alfworld/locks/$t ] && continue
  H=""; while [ -z "$H" ]; do
    [ -d reports/alfworld/locks/$t ] && break
    for j in $HOLDS; do if [ "$(squeue -j $j -h -o %t 2>/dev/null)" = R ] && [ "${USED[$j]:-0}" -lt 1 ]; then H=$j; break; fi; done
    [ -z "$H" ] && sleep 60
  done
  [ -z "$H" ] && continue
  USED[$H]=1; tmux new -d -s task_$t "cd $PWD; $(runner $t) $t $H"; sleep 30
done
echo "=== $(date) dispatcher: all tasks placed or taken" | tee -a reports/alfworld/dispatch.log
