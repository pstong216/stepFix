#!/bin/bash
# WebShop round 1 restart #2 (Oct 5 21:15): stricter verifiable product-page progress + replay of own wins; rescore the stored scan.
cd /users/8/pan00389/multi-agent/error-attribution; R=reports/webshop/round1
tmux kill-session -t webshop_r1 2>/dev/null; sleep 2
for p in $(pgrep -f 'webshop_round.sh'); do kill $p 2>/dev/null; done
for p in $(pgrep -f 'webshop_evidence.py train|webshop_agent.py collect --adapter reports/webshop/round1/arm'); do kill $p 2>/dev/null; done; sleep 3
A=$R/archive_evidence_v1_arms; mkdir -p $A; for f in labels.jsonl steps.jsonl arm_env_top arm_conf_top arm_placebo arm_all; do [ -e $R/$f ] && mv $R/$f $A/; done
echo "=== $(date '+%F %T') WEBSHOP r1 restarted #2: verifiable product-page progress (options offered, price cap, attribute phrase) + replay; scan rescored" | tee -a reports/webshop/queue.log
tmux new-session -d -s webshop_r1 "bash scripts/webshop_round.sh 100"
