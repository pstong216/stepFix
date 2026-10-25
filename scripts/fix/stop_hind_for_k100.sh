#!/bin/bash
# stop the final-recipe chain / builders (teacher passes are resumable) so the budget can be switched 86 -> 100
for s in hind_main hind_share hind_arms hind_sel_arms; do tmux kill-session -t $s 2>/dev/null; done; sleep 2
for p in $(pgrep -f 'run_final_chains_hind.sh|build_hind_arms.sh|build_hind_selector_arms.sh'); do kill $p 2>/dev/null; done
for p in $(pgrep -f 'self_attr_round.sh selfattr 1 reports/alfworld/student3b_low100/final'); do kill $p 2>/dev/null; done; sleep 2
for p in $(pgrep -f 'selfattr_hind/r1/(self|sel_random|sel_last)'); do kill $p 2>/dev/null; done; sleep 3
echo "left: $(pgrep -f 'selfattr_hind' | wc -l)"
