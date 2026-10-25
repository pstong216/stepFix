#!/bin/bash
tmux kill-session -t pool_chain 2>/dev/null; sleep 1
for p in $(pgrep -f "run_pool_chain[.]sh"); do kill $p 2>/dev/null; done
for p in $(pgrep -f "self_attr_round.sh selfattr 3 reports/alfworld/self_attr_rounds/selfattr_pool/r2"); do kill $p 2>/dev/null; done; sleep 2
for p in $(pgrep -f "out reports/alfworld/self_attr_rounds/selfattr_pool/r3/traj.jsonl"); do kill $p 2>/dev/null; done
echo "=== $(date "+%F %T") all-adjudication chain paused before R3 (R2 dropped to 160; waiting for the R2 budget rerun)" | tee -a reports/alfworld/self_attr_rounds/queue.log
