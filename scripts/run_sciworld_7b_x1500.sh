#!/bin/bash
# Oct 10 21:45 (user: "7b 得按照更新的配方，之前的不算"): 7B ScienceWorld R1 with the CURRENT recipe = the 3B x1500 recipe:
#   1,483 distinct train variations (per_task 75, seed-1 shuffle; the 518 = positions 0..19 already collected by the 7B R0 student in
#   stepfix7b/r1/col, new = positions 20..74 via --job_slice 20 75), 7B Model A (no_observations), FULL-VIEW teacher (the 7B student) with
#   relaxed validity + refill, one-step generic evidence, env_top K = round(0.2 x failures), replay_fw n = round(1.2 x K), train from the
#   7B student (1 ep lr 1e-5 bs 8 seed 1), 848 eval.  No random control (GPU budget).  Small shards (60 collect / 48 teacher) so 7B
#   processes turn over fast and the 3B x1500 chain (top priority) can take GPUs back via $SF7/.res_only_until (set by the watcher).
source scripts/lib_sciworld_7b.sh; R=1; D=$SF7/x1; mkdir -p $D
s7log "x1500-7B: driver start on $(hostname) (pid $$)"
if [ ! -s $D/traj.jsonl ]; then
  collect7 $D/col $STU7 1 r0x7 60 --job_slice 20 75
  $P scripts/sciworld_stepfix.py prep --dir $D --inputs "$SF7/r1/col/traj_s*.jsonl" "$D/col/traj_s*.jsonl" --expect 1483 | sed "s/^/x7 /" | tee -a $SF7/queue.log
  $P -c "
L=open('$D/traj.release.jsonl').readlines()
for i in range(6): open('$D/rel_s%d.jsonl'%i,'w').writelines(L[i::6])"
fi
if [ ! -s $D/pointer.jsonl ]; then
  for i in $(seq 0 5); do NEED=$GMEM7 loop7 $D/modelA_s$i.log "^DONE" scripts/alfworld_attribute.py --attributor $MA7 --variant no_observations --release $D/rel_s$i.jsonl --out $D/modelA_pred_s$i.jsonl & sleep 20; done; wait
  $P scripts/sciworld_stepfix.py pointer --dir $D | sed "s/^/x7 /" | tee -a $SF7/queue.log
fi
if ! grep -q "DONE teacher (merged)" $D/teacher.log 2>/dev/null; then
  for i in $(seq 0 47); do NEED=$TEACH7 loop7 $D/teacher_s$i.log "DONE teacher" scripts/sciworld_stepfix.py teacher --model $MODEL --dir $D --adapter $STU7 --shard $i 48 --seed $R --accept_env --max_extra 16 --full_view & sleep 15; done; wait
  echo "DONE teacher (merged)" >> $D/teacher.log
fi
$P scripts/sciworld_stepfix.py select --dir $D --round $R --rho 0.2 > $D/select.log; grep -v "^random\|^fp" $D/select.log | sed "s/^/x7 /" | tee -a $SF7/queue.log
[ -s $D/replay_fw.jsonl ] || $P scripts/sciworld_stepfix.py replay_fw --dir $D --n $(python3 -c "print(round(1.2 * $(grep -c . $D/targets_env_top.jsonl)))") | sed "s/^/x7 /" | tee -a $SF7/queue.log
NEED=$TRAIN7 loop7 $D/train.log "DONE train" scripts/sciworld_stepfix.py train --model $MODEL --data $D/targets_env_top.jsonl $D/replay_fw.jsonl --adapter $STU7 --out $D/stepfix --seed $R
s7log "x7: trained ($(tail -1 $D/stepfix/train_log.jsonl))"
eval7 $D/eval $D/stepfix/adapter x7
s7log "x7 848: $(sum848 $D/eval/eval_test.jsonl $D/eval/eval_dev.jsonl)"
for rf in "R0_7B=$SF7/r0" "x1_3B=$SF3/x1/eval" "r1b_3B=$SF3/r1b/eval_stepfix"; do rn=${rf%%=*}; rd=${rf#*=}
  $P scripts/sciworld_stepfix_compare.py --a $rn=$rd/eval_test.jsonl,$rd/eval_dev.jsonl --b x7=$D/eval/eval_test.jsonl,$D/eval/eval_dev.jsonl | head -1 | tee -a $SF7/queue.log; done
s7log "x1500-7B DONE"
