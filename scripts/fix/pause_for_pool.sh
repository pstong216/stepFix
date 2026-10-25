#!/bin/bash
# user Oct 6 01:55: all compute to validating environment adjudication -> pause teacher-pointed control, WebShop, duplicate R1 pool arm
cd /users/8/pan00389/multi-agent/error-attribution
for s in hind_r2tp webshop_gen webshop_eqv rig_pool rig_hind_sel; do tmux kill-session -t $s 2>/dev/null; done; sleep 2
for p in $(pgrep -f 'r2_teacher_point.sh|webshop_round_generic.sh|webshop_eqv.sh'); do kill $p 2>/dev/null; done
for p in $(pgrep -f 'self_attr_round.sh selfattr 2 .*_hindtp|CHAIN_SUFFIX=_hindtp'); do kill $p 2>/dev/null; done; sleep 2
for p in $(pgrep -f 'tag selfattr_hindtp|tag wsg_|tag rg_hind_(pool|first)_r1|webshop_evidence.py train --data reports/webshop'); do kill $p 2>/dev/null; done; sleep 3
R=reports/alfworld/rigorous; for a in hind_pool_r1 hind_first_r1_rep; do [ -f "$R/$a/DONE" ] || rm -f "${R:?}/${a:?}/CLAIM"; done
echo "=== $(date '+%F %T') user: all compute to ENVIRONMENT ADJUDICATION chain (R1-R3); paused teacher-point control, WebShop, duplicate pool arm" | tee -a reports/alfworld/self_attr_rounds/queue.log
tmux new-session -d -s pool_chain "bash scripts/run_pool_chain.sh"
