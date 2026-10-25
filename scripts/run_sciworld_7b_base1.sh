#!/bin/bash
# Oct 12 01:10 (user): ScienceWorld 7B one-round baselines from the 7B R0 student on w7_1's collection (1,483 variations):
#   rft  : standard RFT = every step of every win (exact duplicates dropped), train from 7B R0
#   roll : rollout verification = every w7_1 teacher candidate (top-4, 7B R0 teacher) rolled out to the episode end by 7B R0; K = round(0.2 x
#          failures) random states with a winning candidate; + w7_1's retention replay (389)
# usage: NAME MODE   (848 eval with eval7)
source scripts/lib_sciworld_7b.sh; NM=$1; MODE=$2; S=$SF7/w7_1; D=$SF7/$NM; mkdir -p $D
s7log "$NM: start (mode $MODE)"
if [ $MODE = rft ]; then
  [ -s $D/traj.jsonl ] || cp $S/traj.jsonl $D/traj.jsonl
  [ -s $D/train_rows.jsonl ] || $P scripts/sciworld_baselines.py build --dir $D --mode rftfull --n 0 --seed 1 | sed "s/^/$NM /" | tee -a $SF7/queue.log
  DATA="$D/train_rows.jsonl"
else
  [ -s $D/pointer.jsonl ] || { cp $S/{traj.jsonl,ids_fail.jsonl,traj.release.jsonl,pointer.jsonl,pointer_stats.json} $D/; mkdir -p $D/teacher; cp $S/teacher/rows_s*.jsonl $D/teacher/; }
  if ! grep -q "DONE rollout (merged)" $D/rollout.log 2>/dev/null; then pids=""
    for i in $(seq 0 47); do NEED=$GMEM7 loop7 $D/rollout_s$i.log "DONE rollout" scripts/sciworld_stepfix.py rollout --model $MODEL --dir $D --adapter $STU7 --shard $i 48 & pids="$pids $!"; sleep 10; done; wait $pids
    echo "DONE rollout (merged)" >> $D/rollout.log; fi
  $P scripts/sciworld_stepfix.py select_rollout --dir $D --round 1 | sed "s/^/$NM /" | tee -a $SF7/queue.log
  DATA="$D/targets_roll.jsonl $S/replay_ret.jsonl"
fi
NEED=$TRAIN7 loop7 $D/train.log "DONE train" scripts/sciworld_stepfix.py train --model $MODEL --data $DATA --adapter $STU7 --out $D/model --seed 1
s7log "$NM: trained ($(tail -1 $D/model/train_log.jsonl))"
eval7 $D/eval $D/model/adapter $NM; s7log "$NM 848: $(sum848 $D/eval/eval_test.jsonl $D/eval/eval_dev.jsonl)"
for rf in "R0_7B=$SF7/r0" "w7_1=$SF7/w7_1/eval"; do rn=${rf%%=*}; rd=${rf#*=}; $P scripts/sciworld_stepfix_compare.py --a $rn=$rd/eval_test.jsonl,$rd/eval_dev.jsonl --b $NM=$D/eval/eval_test.jsonl,$D/eval/eval_dev.jsonl | head -1 | tee -a $SF7/queue.log; done
s7log "$NM DONE"
