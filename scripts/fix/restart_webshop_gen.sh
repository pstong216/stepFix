#!/bin/bash
cd /users/8/pan00389/multi-agent/error-attribution; G=reports/webshop/round1/generic
tmux kill-session -t webshop_gen 2>/dev/null; sleep 2
for p in $(pgrep -f 'webshop_round_generic.sh|webshop_generic.py (scan|rollout)'); do kill $p 2>/dev/null; done; sleep 3
mkdir -p $G/archive_no_teacher; for f in scan.jsonl scan.log; do [ -e $G/$f ] && mv $G/$f $G/archive_no_teacher/; done
echo "=== $(date '+%F %T') WebShop generic: candidates now from the SAME teacher as ALFWorld (hindsight + own analysis); scan restarted" | tee -a reports/webshop/queue.log
tmux new-session -d -s webshop_gen "bash scripts/webshop_round_generic.sh"
