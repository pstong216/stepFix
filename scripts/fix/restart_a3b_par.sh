#!/bin/bash
cd /users/8/pan00389/multi-agent/error-attribution
tmux kill-session -t a3b_final 2>/dev/null; sleep 1
for p in $(pgrep -f "run_a3b_final_chain[.]sh"); do kill $p 2>/dev/null; done
for p in $(pgrep -f "self_attr_round.sh selfattr 1 reports/alfworld/student3b_low100/final"); do kill $p 2>/dev/null; done
for p in $(pgrep -f "SE_DIR=reports/alfworld/self_attr_rounds/selfattr_a3bfinal|proxy_eval_se3b.py onestep --arms self --limit 300$"); do :; done
sleep 2
for p in $(pgrep -f "proxy_eval_se3b[.]py onestep"); do if tr "\\0" " " < /proc/$p/environ 2>/dev/null | grep -q "selfattr_a3bfinal"; then kill $p; fi; done
sleep 2
tmux new-session -d -s a3b_final "bash scripts/run_a3b_final_chain.sh"
echo restarted
