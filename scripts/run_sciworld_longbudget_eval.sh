#!/bin/bash
# Supplementary LONG-BUDGET eval (user request Oct 10 02:55): the 7 task types whose median TRAIN gold length > 50 get budget
# min(150, ceil(1.5 x median)); same eval episodes as the 848 set (test per_task 30 + dev per_task 7, seed 0), greedy, --snap safe.
#   bash scripts/run_sciworld_longbudget_eval.sh NAME ADAPTER
source scripts/lib_sciworld_stepfix.sh; NAME=$1; AD=$2; D=$SF/longbudget/$NAME; mkdir -p $D
TASKS="inclined-plane-determine-angle inclined-plane-friction-named-surfaces inclined-plane-friction-unnamed-surfaces grow-fruit measure-melting-point-unknown-substance mendelian-genetics-known-plant mendelian-genetics-unknown-plant"
BM='{"inclined-plane-determine-angle": 144, "inclined-plane-friction-named-surfaces": 102, "inclined-plane-friction-unnamed-surfaces": 150, "grow-fruit": 132, "measure-melting-point-unknown-substance": 140, "mendelian-genetics-known-plant": 150, "mendelian-genetics-unknown-plant": 150}'
slog "longbudget $NAME: start ($AD)"
for i in $(seq 0 7); do shard_loop $D/eval_test_s$i.log scripts/sciworld_agent.py collect --adapter $AD --split test --per_task 30 --budget 50 --budget_map "$BM" --tasks $TASKS --temperature 0 --seed 0 --snap safe --shard $i 8 --out $D/eval_test_s$i.jsonl --tag lb_${NAME}_test & sleep ${LAUNCH_GAP:-30}; done
for i in 0 1; do shard_loop $D/eval_dev_s$i.log scripts/sciworld_agent.py collect --adapter $AD --split dev --per_task 7 --budget 50 --budget_map "$BM" --tasks $TASKS --temperature 0 --seed 0 --snap safe --shard $i 2 --out $D/eval_dev_s$i.jsonl --tag lb_${NAME}_dev & sleep ${LAUNCH_GAP:-30}; done
wait; cat $D/eval_test_s*.jsonl > $D/eval_test.jsonl; cat $D/eval_dev_s*.jsonl > $D/eval_dev.jsonl
slog "longbudget $NAME: DONE $(cat $D/eval_test.jsonl $D/eval_dev.jsonl | grep -c '\"won\": true')/$(cat $D/eval_test.jsonl $D/eval_dev.jsonl | grep -c .)"
