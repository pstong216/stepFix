#!/bin/bash
# Oct 10 03:40 priority watcher v2 (coordinator: r2f -> r3f -> long-budget eval -> ablations).
# hold_low ON (ablation chains launch nothing new) while (a) a main round is between pointer and trained adapter, or (b) any main-chain
# process is WAITING for a GPU (its log ends with the "### attempt" line written just before S() for > 60 s).
# If main processes have waited > 150 s, preempt that many low-priority (ablation) GPU processes (srun of reports/sciworld/stepfix/abl_*,
# newest first): they lose only the episode / state in flight and resume later.
SF=reports/sciworld/stepfix; MAIN="r2f r3f longbudget"
while true; do h=0; waiting=0; long=0; now=$(date +%s)
  for d in r2f r3f; do [ -s $SF/$d/pointer.jsonl ] && [ ! -f $SF/$d/model/adapter/adapter_config.json ] && h=1; done
  for d in $MAIN; do [ -d $SF/$d ] || continue
    for lg in $(find $SF/$d -name "*.log" -mmin -600 2>/dev/null); do
      tail -n 1 $lg 2>/dev/null | grep -q "^### attempt" || continue
      age=$((now - $(stat -c %Y $lg))); [ $age -gt 60 ] && { waiting=$((waiting + 1)); h=1; }; [ $age -gt 150 ] && long=$((long + 1)); done; done
  if [ $h = 1 ]; then [ -f $SF/.hold_low ] || { touch $SF/.hold_low; echo "=== $(date '+%F %T') hold_low ON (main waiting: $waiting)"; }
  else [ -f $SF/.hold_low ] && { rm -f $SF/.hold_low; echo "=== $(date '+%F %T') hold_low OFF"; }; fi
  if [ $long -gt 0 ]; then
    pids=$(ps -eo pid,lstart,args --sort=-start_time | grep "^ *[0-9].*srun --jobid" | grep "stepfix/abl_" | grep -v grep | awk '{print $1}' | head -n $long)
    [ -n "$pids" ] && { echo "=== $(date '+%F %T') preempting $(echo $pids | wc -w) ablation process(es) for $long waiting main process(es): $pids"; kill $pids; sleep 90; }
  fi
  [ -f $SF/r3f/model/adapter/adapter_config.json ] && [ -s $SF/longbudget/r2f/eval_test.jsonl ] && [ -s $SF/r3f/eval/eval_test.jsonl ] && { rm -f $SF/.hold_low; echo "main chain done; watcher exits"; break; }
  sleep 30; done
