#!/bin/bash
# switch the final chains' round-2+ step selector to the LATE log rule: write the control file, restart the rounds already in R2
# (main, placebo: collection resumes from files); chains still in R1 pick the new script up when their R2 starts.
cd /users/8/pan00389/multi-agent/error-attribution; SA=reports/alfworld/self_attr_rounds
echo "${1:-3}" > $SA/STEP_LATE_FROM_R2
for s in hind_main hind_placebo; do tmux kill-session -t $s 2>/dev/null; done; sleep 2
for p in $(pgrep -f 'run_final_chains_hind.sh (main|placebo)'); do kill $p 2>/dev/null; done
for p in $(pgrep -f 'self_attr_round.sh selfattr(_placebo)? 2 reports/alfworld/self_attr_rounds/selfattr(_placebo)?_hind/r1'); do kill $p 2>/dev/null; done; sleep 2
for p in $(pgrep -f 'tag selfattr(_placebo)?_hind_r2'); do kill $p 2>/dev/null; done; sleep 5
echo "left: $(pgrep -f 'selfattr(_placebo)?_hind_r2' | wc -l)"
echo "=== $(date '+%F %T') user: R2+ step selector = LATE log rule (window ${1:-3}); main/placebo R2 restarted (collection resumes)" | tee -a $SA/queue.log
tmux new-session -d -s hind_main "bash scripts/run_final_chains_hind.sh main"; sleep 30
tmux new-session -d -s hind_placebo "bash scripts/run_final_chains_hind.sh placebo"
