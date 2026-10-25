#!/bin/bash
# Oct 11 10:55 (user: 7B ablations today): ONE-round 7B ablations of the final recipe from the 7B R0 student, w7_1's collection / 7B
# Model A / top-4 teacher rows / retention replay (389).  NAME DATA...  (abl7_rand: random teacher top-1 x162 + replay; abl7_norep:
# w7_1 corrections only; abl7_nofil: teacher top-1 at every failure + replay).  848 eval (eval7).
source scripts/lib_sciworld_7b.sh; pick7res() { :; }   # Oct 11 11:10: 7B ablations never use the GPUs reserved (sw7b_gpu) for the 7B main chain
NM=$1; shift; D=$SF7/$NM; mkdir -p $D
s7log "$NM: start (rows $(cat "$@" | grep -c .))"
NEED=$TRAIN7 loop7 $D/train.log "DONE train" scripts/sciworld_stepfix.py train --model $MODEL --data "$@" --adapter $STU7 --out $D/model --seed 1
s7log "$NM: trained ($(tail -1 $D/model/train_log.jsonl))"
eval7 $D/eval $D/model/adapter $NM; s7log "$NM 848: $(sum848 $D/eval/eval_test.jsonl $D/eval/eval_dev.jsonl)"
for rf in "R0_7B=$SF7/r0" "w7_1=$SF7/w7_1/eval"; do rn=${rf%%=*}; rd=${rf#*=}; $P scripts/sciworld_stepfix_compare.py --a $rn=$rd/eval_test.jsonl,$rd/eval_dev.jsonl --b $NM=$D/eval/eval_test.jsonl,$D/eval/eval_dev.jsonl | head -1 | tee -a $SF7/queue.log; done
s7log "$NM DONE"
