#!/bin/bash
tmux kill-session -t heldout500 2>/dev/null; sleep 1
for p in $(pgrep -f "run_heldout500[.]sh"); do kill $p 2>/dev/null; done
for p in $(pgrep -f "tag h500_"); do kill $p 2>/dev/null; done
echo "=== $(date "+%F %T") user: pool validation reported on 274 only -> held-out daemon paused (resumable)" | tee -a reports/alfworld/self_attr_rounds/queue.log
