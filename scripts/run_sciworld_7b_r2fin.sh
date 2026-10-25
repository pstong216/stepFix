#!/bin/bash
# Oct 11 14:46: finish the w7_2 848 eval after the 7B chain was stopped (user: no 7B R3 for now; tune R2 first).
source scripts/lib_sciworld_7b.sh; D=$SF7/w7_2
eval7 $D/eval $D/model/adapter w7_2; s7log "w7_2 848: $(sum848 $D/eval/eval_test.jsonl $D/eval/eval_dev.jsonl)"
for rf in "R0_7B=$SF7/r0" "w7_1=$SF7/w7_1/eval" "w2k_3B=$SF3/w2k/eval"; do rn=${rf%%=*}; rd=${rf#*=}; $P scripts/sciworld_stepfix_compare.py --a $rn=$rd/eval_test.jsonl,$rd/eval_dev.jsonl --b w7_2=$D/eval/eval_test.jsonl,$D/eval/eval_dev.jsonl | head -1 | tee -a $SF7/queue.log; done
