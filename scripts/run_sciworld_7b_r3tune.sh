#!/bin/bash
# Oct 11 21:20 (user: try to make 7B R3 improve further): 7B R3 re-trained from w7_2b3lr on w7_3's teacher rows with a different setting.
# usage: NAME DATA_DIR BETA LR   (DATA_DIR holds targets_cap.jsonl + replay_ret.jsonl; seed 3; --kl_replay); 848 eval.
source scripts/lib_sciworld_7b.sh; NM=$1; DD=$2; BETA=$3; LR=$4; D=$SF7/$NM; mkdir -p $D
s7log "$NM: start (data $DD, KL beta $BETA, lr $LR, from w7_2b3lr)"
NEED=$TRAIN7 loop7 $D/train.log "DONE train" scripts/sciworld_stepfix.py train --model $MODEL --data $DD/targets_cap.jsonl $DD/replay_ret.jsonl --adapter $SF7/w7_2b3lr/model/adapter --out $D/model --seed 3 --kl_replay --kl_beta $BETA --lr $LR
s7log "$NM: trained ($(tail -1 $D/model/train_log.jsonl))"
eval7 $D/eval $D/model/adapter $NM; s7log "$NM 848: $(sum848 $D/eval/eval_test.jsonl $D/eval/eval_dev.jsonl)"
for rf in "R0_7B=$SF7/r0" "w7_1=$SF7/w7_1/eval" "w7_2b3lr=$SF7/w7_2b3lr/eval" "w7_3=$SF7/w7_3/eval"; do rn=${rf%%=*}; rd=${rf#*=}; [ -s $rd/eval_test.jsonl ] || { cat $rd/eval_test_s*.jsonl > /tmp/claude-$$-t.jsonl 2>/dev/null; }; [ -s $rd/eval_test.jsonl ] && $P scripts/sciworld_stepfix_compare.py --a $rn=$rd/eval_test.jsonl,$rd/eval_dev.jsonl --b $NM=$D/eval/eval_test.jsonl,$D/eval/eval_dev.jsonl | head -1 | tee -a $SF7/queue.log; done
s7log "$NM DONE"
