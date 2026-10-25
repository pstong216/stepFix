#!/bin/bash
# Oct 11 14:46 (user: "先把7B R2跑好"): 7B R2 re-trained on w7_2's corrections + retention replay with a different KL anchor strength / lr.
# usage: NAME BETA LR   (start = w7_1 adapter, seed 2, --kl_replay --kl_beta BETA --lr LR); 848 eval.
source scripts/lib_sciworld_7b.sh; NM=$1; BETA=$2; LR=$3; S=$SF7/w7_2; D=$SF7/$NM; mkdir -p $D
s7log "$NM: start (KL beta $BETA, lr $LR, data w7_2)"
NEED=$TRAIN7 loop7 $D/train.log "DONE train" scripts/sciworld_stepfix.py train --model $MODEL --data $S/targets_cap.jsonl $S/replay_ret.jsonl --adapter $SF7/w7_1/model/adapter --out $D/model --seed 2 --kl_replay --kl_beta $BETA --lr $LR
s7log "$NM: trained ($(tail -1 $D/model/train_log.jsonl))"
eval7 $D/eval $D/model/adapter $NM; s7log "$NM 848: $(sum848 $D/eval/eval_test.jsonl $D/eval/eval_dev.jsonl)"
for rf in "R0_7B=$SF7/r0" "w7_1=$SF7/w7_1/eval" "w7_2=$SF7/w7_2/eval" "w2k_3B=$SF3/w2k/eval"; do rn=${rf%%=*}; rd=${rf#*=}; [ -s $rd/eval_test.jsonl ] && $P scripts/sciworld_stepfix_compare.py --a $rn=$rd/eval_test.jsonl,$rd/eval_dev.jsonl --b $NM=$D/eval/eval_test.jsonl,$D/eval/eval_dev.jsonl | head -1 | tee -a $SF7/queue.log; done
s7log "$NM DONE"
