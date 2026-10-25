#!/bin/bash
# Round 1 of step-localised verified Attr-OPD on the low-data student. Waits for the low100 chain and its progress-oracle
# labels, then runs arms on three holds: $1: attr -> random ; $2: oracle -> last ; $3: uniform (equal teacher calls).
H1=$1; H2=$2; H3=$3; cd /users/8/pan00389/multi-agent/error-attribution; L=reports/alfworld/opd_round1.log
export TAG=student3b_low100 TEACHER=Qwen/Qwen2.5-3B-Instruct TEACHER_ADAPTER=reports/alfworld/student3b_sft/final ROUND=round1
until grep -q "CHAIN_DONE" reports/alfworld/chain_student3b_low100.log 2>/dev/null; do sleep 60; done
NF=$(grep -c '"won": false' reports/alfworld/${TAG}_train_b50.jsonl)
echo "=== $(date) chain done: $NF failures; waiting for progress-oracle labels" | tee -a $L
for i in $(seq 1 90); do NP=$(wc -l < reports/alfworld/progress_${TAG}_train_b50.jsonl 2>/dev/null || echo 0); [ "$NP" -ge $((NF * 95 / 100)) ] && break; sleep 60; done
echo "=== $(date) progress labels: $NP / $NF" | tee -a $L
run() { H=$1; shift; for ARM in "$@"; do
  [ "$(squeue -j $H -h -o %t 2>/dev/null)" = "R" ] || { echo "hold $H not running; skipping $ARM" | tee -a $L; continue; }
  echo "=== $(date) start $ARM on hold $H" | tee -a $L
  scripts/run_alfworld_opd_arm.sh $H $ARM 1000 2 >> $L 2>&1
  echo "=== $(date) end $ARM: $(tail -1 reports/alfworld/opd/$TAG/$ARM/run.log | cut -c1-160)" | tee -a $L
done; }
run $H1 attr random &
run $H2 oracle last &
( until grep -q "TEACHER7B_DONE" reports/alfworld/chain_teacher7b_e2.log 2>/dev/null; do sleep 60; done; run $H3 uniform ) &
wait
echo "=== $(date) ROUND1_DONE" | tee -a $L
