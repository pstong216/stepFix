#!/bin/bash
tmux kill-session -t hind_r2spec 2>/dev/null; sleep 1
for p in $(pgrep -f "r2_spec_vs_generic[.]sh|CHAIN_SUFFIX=_hindpoolspec"); do kill $p 2>/dev/null; done
for p in $(pgrep -f "selfattr_hindpoolspec/r2/model"); do kill $p 2>/dev/null; done
echo "=== $(date "+%F %T") specific-vs-generic R2 control cancelled: 97/100 identical corrections, evidence ruled out" | tee -a reports/alfworld/self_attr_rounds/queue.log
