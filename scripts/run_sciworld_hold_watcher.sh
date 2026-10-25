#!/bin/bash
# Oct 10: hold low-priority (ablation) launches while a main refined round (r2f, r3f) is between pointer and trained adapter.
SF=reports/sciworld/stepfix
while true; do h=0
  for d in r2f r3f; do [ -s $SF/$d/pointer.jsonl ] && [ ! -f $SF/$d/model/adapter/adapter_config.json ] && h=1; done
  if [ $h = 1 ]; then [ -f $SF/.hold_low ] || { touch $SF/.hold_low; echo "=== $(date '+%F %T') hold_low ON"; }
  else [ -f $SF/.hold_low ] && { rm -f $SF/.hold_low; echo "=== $(date '+%F %T') hold_low OFF"; }; fi
  [ -f $SF/r3f/model/adapter/adapter_config.json ] && { rm -f $SF/.hold_low; echo "main adapters done; watcher exits"; break; }
  sleep 30; done
