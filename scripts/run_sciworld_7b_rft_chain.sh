#!/bin/bash
# Oct 12 21:10 (user: 7B ScienceWorld baselines to R3): rounds 2 and 3 of 7B standard RFT, continuing from rft7_1 (R1 = 333). Same protocol as
# the 3B RFT chain (x2rft/x3rft): collect the 1,483 variations (per_task 75, seed 1, T=1.0) with the previous RFT adapter, train on every
# step of every win (exact duplicates dropped; sciworld_baselines.py build --mode rftfull), seed R, plain CE as rft7_1; 848 eval.
source scripts/lib_sciworld_7b.sh; umask 0027; export LAUNCH_GAP=45 LOOP7_MAX=400   # Oct 12 23:30: gap 8 s caused OOM thrash (pick() races) and 25-attempt give-ups -> partial collections
ev7() { local D=$1 NM=$(basename $1) E=$1/eval pids=""; mkdir -p $E
  for i in $(seq 0 23); do loop7 $E/eval_test_s$i.log "DONE collect" scripts/sciworld_agent.py collect --model $MODEL --adapter $D/model/adapter --split test --per_task 30 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 24 --out $E/eval_test_s$i.jsonl --tag ${NM}_test & pids="$pids $!"; sleep 30; done
  for i in $(seq 0 5); do loop7 $E/eval_dev_s$i.log "DONE collect" scripts/sciworld_agent.py collect --model $MODEL --adapter $D/model/adapter --split dev --per_task 7 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 6 --out $E/eval_dev_s$i.jsonl --tag ${NM}_dev & pids="$pids $!"; sleep 30; done
  wait $pids; cat $E/eval_test_s*.jsonl > $E/eval_test.jsonl; cat $E/eval_dev_s*.jsonl > $E/eval_dev.jsonl
  s7log "$NM 848: $(sum848 $E/eval_test.jsonl $E/eval_dev.jsonl)"
  for rf in "R0_7B=$SF7/r0" "w7_2b3lr=$SF7/w7_2b3lr/eval" "w7_3=$SF7/w7_3/eval" "rft7_1=$SF7/rft7_1/eval"; do rn=${rf%%=*}; rd=${rf#*=}; [ "$rn" = "$NM" ] && continue
    $P scripts/sciworld_stepfix_compare.py --a $rn=$rd/eval_test.jsonl,$rd/eval_dev.jsonl --b $NM=$E/eval_test.jsonl,$E/eval_dev.jsonl | head -1 | tee -a $SF7/queue.log; done; }
round() { local NM=$1 R=$2 AD=$3 D=$SF7/$1; mkdir -p $D; s7log "$NM: start (standard RFT R$R, from $AD)"
  [ -s $D/traj.jsonl ] || { until [ "$(cat $D/col/traj_s*.jsonl 2>/dev/null | grep -c .)" -ge 1483 ]; do collect7 $D/col $AD 1 $NM 60 --per_task 75; done; $P scripts/sciworld_baselines.py merge --dir $D --inputs "$D/col/traj_s*.jsonl" --expect 1483 | sed "s/^/$NM /" | tee -a $SF7/queue.log; }
  [ -s $D/train_rows.jsonl ] || $P scripts/sciworld_baselines.py build --dir $D --mode rftfull --n 0 --seed $R | sed "s/^/$NM /" | tee -a $SF7/queue.log
  NEED=$TRAIN7 loop7 $D/train.log "DONE train" scripts/sciworld_stepfix.py train --model $MODEL --data $D/train_rows.jsonl --adapter $AD --out $D/model --seed $R
  s7log "$NM: trained on $(grep -c . $D/train_rows.jsonl) rows"; }
round rft7_2 2 $SF7/rft7_1/model/adapter; ( ev7 $SF7/rft7_2 ) & E2=$!
round rft7_3 3 $SF7/rft7_2/model/adapter; ev7 $SF7/rft7_3
wait $E2; s7log "7B RFT chain (rft7_2, rft7_3) DONE"
