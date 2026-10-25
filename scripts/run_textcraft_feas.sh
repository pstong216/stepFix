#!/bin/bash
# TextCraft feasibility (Oct 8, approved by the user as a feasibility check only -- no self-evolution rounds):
# 100 oracle demos -> 3B LoRA SFT student -> greedy eval on test (324 = 108 held-out goals x 3 distractor seeds) and on 300 train-pool tasks.
source scripts/lib_baselines.sh
T=reports/textcraft; STU=$T/student100; mkdir -p $T
tlog() { echo "=== $(date '+%F %T') $*" | tee -a $T/queue.log; }
until [ -f $STU/final/adapter_config.json ]; do $(S 24000) $P -u scripts/textcraft_agent.py sft --n_demos 100 --out $STU > $T/sft_student100.log 2>&1 || { tlog "sft FAILED (retry)"; sleep 120; }; done
tlog "SFT done"
ev() { local SP=$1 OFF=$2 N=$3 O=$4 TG=$5; until grep -q "DONE collect" ${O%.jsonl}.log 2>/dev/null; do $(S 10000) $P -u scripts/textcraft_agent.py collect --adapter $STU/final --split $SP --offset $OFF --n $N --temperature 0 --out $O --tag $TG >> ${O%.jsonl}.log 2>&1 || sleep 60; done; }
ev test 0 162 $T/eval_test_s0.jsonl stu100 & sleep 20
ev test 162 162 $T/eval_test_s1.jsonl stu100 & sleep 20
ev train 0 150 $T/eval_train_s0.jsonl stu100 & sleep 20
ev train 150 150 $T/eval_train_s1.jsonl stu100 & wait
cat $T/eval_test_s0.jsonl $T/eval_test_s1.jsonl > $T/eval_test.jsonl; cat $T/eval_train_s0.jsonl $T/eval_train_s1.jsonl > $T/eval_train300.jsonl
cat $T/eval_test_s0.release.jsonl $T/eval_test_s1.release.jsonl > $T/eval_test.release.jsonl; cat $T/eval_train_s0.release.jsonl $T/eval_train_s1.release.jsonl > $T/eval_train300.release.jsonl
$P -u scripts/textcraft_agent.py report $T/eval_test.jsonl $T/eval_train300.jsonl | tee -a $T/queue.log
tlog "TEXTCRAFT FEAS EVAL DONE"
