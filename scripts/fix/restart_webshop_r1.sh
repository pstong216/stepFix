#!/bin/bash
# WebShop round 1 restart after the evidence fix (Oct 5 19:55): the instruction banner on every page made every goal word 'known',
# so the relevance/progress terms never fired and the filter degenerated to novelty (env_top dev 75/200 < student 84/195).
# Stop the driver + its current GPU step, archive the buggy scan/labels/arms, relaunch (student SFT, dev baseline, collect are reused).
cd /users/8/pan00389/multi-agent/error-attribution; R=reports/webshop/round1
tmux kill-session -t webshop_r1 2>/dev/null; sleep 2
for p in $(pgrep -f 'webshop_round.sh'); do kill $p 2>/dev/null; done
for p in $(pgrep -f 'webshop_evidence.py (train|scan)|webshop_agent.py collect --adapter reports/webshop/round1/arm'); do kill $p 2>/dev/null; done; sleep 3
A=$R/archive_evidence_bug; mkdir -p $A
for f in scan_s0.jsonl scan_s1.jsonl scan_s0.log scan_s1.log scan.jsonl scan.log labels.jsonl steps.jsonl arm_env_top arm_conf_top arm_placebo arm_all; do [ -e $R/$f ] && mv $R/$f $A/; done
echo "archived: $(ls $A | tr '\n' ' ')"
echo "=== $(date '+%F %T') WEBSHOP r1 restarted after evidence fix (instruction banner excluded from 'known'; product-page attribute matches = progress); buggy arms archived" | tee -a reports/webshop/queue.log
tmux new-session -d -s webshop_r1 "bash scripts/webshop_round.sh 100"
