#!/bin/bash
tmux kill-session -t hind_main 2>/dev/null; sleep 1
for p in $(pgrep -f "run_final_chains_hind.sh main"); do kill $p 2>/dev/null; done
for p in $(pgrep -f "self_attr_round.sh selfattr 3 reports/alfworld/self_attr_rounds/selfattr_hind/r2"); do kill $p 2>/dev/null; done; sleep 2
for p in $(pgrep -f "out reports/alfworld/self_attr_rounds/selfattr_hind/r3/traj.jsonl"); do kill $p 2>/dev/null; done
echo "=== $(date "+%F %T") late-rule main chain R3 paused (not needed for adjudication validation)" | tee -a reports/alfworld/self_attr_rounds/queue.log
