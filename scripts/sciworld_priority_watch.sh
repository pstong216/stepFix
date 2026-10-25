#!/bin/bash
# Oct 11 03:00: priority = 3B w chain GPU-heavy stages (w3 teacher, w-chain 848 evals) > 7B w chain > the rest (x2rft, RL evals).
# While a 3B heavy stage runs, 7B uses its reserved GPU only ($SF7/.res_only_until = now + 180 s).
cd /users/8/pan00389/multi-agent/error-attribution; SF=reports/sciworld/stepfix; SF7=reports/sciworld/stepfix7b
while true; do heavy=0
  for yy in w3 wk3; do ls $SF/$yy/teacher_s*.log >/dev/null 2>&1 && ! grep -q "DONE teacher (merged)" $SF/$yy/teacher.log 2>/dev/null && heavy=1; done
  pgrep -f "[-]-tag (w2|w2k|w3|w3k|wk3)_(test|dev)" >/dev/null && heavy=1
  if [ $heavy = 1 ]; then echo $(( $(date +%s) + 180 )) > $SF7/.res_only_until; fi
  sleep 60; done
