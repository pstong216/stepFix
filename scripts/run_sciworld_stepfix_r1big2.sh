#!/bin/bash
# Oct 10 16:50 v2 (restart between stages, after the collection): teacher 32 shards, 848 eval 36 shards (28 test + 8 dev). Oct 10 16:30 (user-approved, top priority): (I1) BIG R1, budget-matched to the 3-round chain: student10 collects the same 518 train
# variations (per_task 20, seed 1) THREE times at T=1.0 (sample 1 = the existing r0 collection; samples 2 and 3 new, tags r0fix_s2 / r0fix_s3),
# then the R1 pipeline on ALL failures: 3B Model A pointer (6 shards), FULL-VIEW teacher + relaxed validity + refill (24 shards), one-step
# generic evidence, env_top K = round(0.2 x failures), replay_fw 300 (100 x 3), train from student10 (1 ep, lr 1e-5, bs 8), 848 eval.
source scripts/lib_sciworld_stepfix.sh; D=$SF/r1big; mkdir -p $D; export GMEM=9000
slog "r1big: start on $(hostname)"
( collect_sw $D/col2 $STU0 1 r0fix_s2 15 --per_task 20 ) &
sleep 30; ( collect_sw $D/col3 $STU0 1 r0fix_s3 15 --per_task 20 ) &
wait
eval_wide() { local E=$1 AD=$2 TG=$3; mkdir -p $E   # 28 test + 8 dev shards (same episodes as eval_sw; files eval_test_w*/eval_dev_w*)
  for i in $(seq 0 27); do shard_loop $E/eval_test_w$i.log scripts/sciworld_agent.py collect --adapter $AD --split test --per_task 30 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 28 --out $E/eval_test_w$i.jsonl --tag ${TG}_test & sleep 10; done
  for i in $(seq 0 7); do shard_loop $E/eval_dev_w$i.log scripts/sciworld_agent.py collect --adapter $AD --split dev --per_task 7 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 8 --out $E/eval_dev_w$i.jsonl --tag ${TG}_dev & sleep 10; done
  wait; cat $E/eval_test_w*.jsonl > $E/eval_test.jsonl; cat $E/eval_dev_w*.jsonl > $E/eval_dev.jsonl; }
[ -s $D/pointer.jsonl ] || {
  $P scripts/sciworld_stepfix.py prep --dir $D --inputs "reports/sciworld/r0fix/traj_s?.jsonl" "$SF/r0/ext/traj_s*.jsonl" "$D/col2/traj_s*.jsonl" "$D/col3/traj_s*.jsonl" --expect 1554 | sed "s/^/r1big /" | tee -a $SF/queue.log
  $P -c "
L=open('$D/traj.release.jsonl').readlines()
for i in range(6): open('$D/rel_s%d.jsonl'%i,'w').writelines(L[i::6])"
  for i in $(seq 0 5); do ( until grep -q "^DONE" $D/modelA_s$i.log 2>/dev/null; do wait_free "--out $D/modelA_pred_s$i.jsonl"; $(SG $GMEM) $P -u scripts/alfworld_attribute.py --attributor reports/attributor_no_observations/final --variant no_observations --release $D/rel_s$i.jsonl --out $D/modelA_pred_s$i.jsonl >> $D/modelA_s$i.log 2>&1 || sleep 30; done ) & sleep 20; done; wait
  $P scripts/sciworld_stepfix.py pointer --dir $D | sed "s/^/r1big /" | tee -a $SF/queue.log; }
if ! grep -q "DONE teacher (merged)" $D/teacher.log 2>/dev/null; then
  for i in $(seq 0 31); do ( until grep -q "DONE teacher" $D/teacher_s$i.log 2>/dev/null; do echo "### attempt $(date '+%F %T')" >> $D/teacher_s$i.log
      wait_free "teacher --dir $D --adapter .* --shard $i 32"; $(SG 12000) $P -u scripts/sciworld_stepfix.py teacher --dir $D --adapter $STU0 --shard $i 32 --seed 1 --accept_env --max_extra 16 --full_view >> $D/teacher_s$i.log 2>&1 || sleep 30; done ) & sleep 8; done; wait
  echo "DONE teacher (merged)" >> $D/teacher.log
fi
$P scripts/sciworld_stepfix.py select --dir $D --round 1 --rho 0.2 > $D/select.log; grep -v "^random\|^fp" $D/select.log | sed "s/^/r1big /" | tee -a $SF/queue.log
[ -s $D/replay_fw.jsonl ] || $P scripts/sciworld_stepfix.py replay_fw --dir $D --n 300 | sed "s/^/r1big /" | tee -a $SF/queue.log
slog "r1big: kept corrections with S >= 2.0: $($P -c "import json; print(sum(json.loads(l)['S'] >= 2.0 for l in open('$D/targets_env_top.jsonl')))") of $(grep -c . $D/targets_env_top.jsonl)"
until [ -f $D/model/adapter/adapter_config.json ]; do echo "### attempt $(date '+%F %T')" >> $D/train.log
  wait_free "train --data .* --out $D/model "; $(SG 20000) $P -u scripts/sciworld_stepfix.py train --data $D/targets_env_top.jsonl $D/replay_fw.jsonl --adapter $STU0 --out $D/model --seed 1 >> $D/train.log 2>&1 || { slog "r1big: train attempt failed (retry)"; sleep 60; }
  for k in $(seq 30); do [ -f $D/model/adapter/adapter_config.json ] && break; sleep 2; done; done
slog "r1big: trained ($(tail -1 $D/model/train_log.jsonl))"
eval_wide $D/eval $D/model/adapter r1big; slog "r1big: 848 eval DONE"
for rf in "R0=$SF/r0/eval_test.jsonl,reports/sciworld/refix/eval_dev_safe.jsonl" "r1b=$SF/r1b/eval_stepfix/eval_test.jsonl,$SF/r1b/eval_stepfix/eval_dev.jsonl" "r2h=$SF/r2h/eval/eval_test.jsonl,$SF/r2h/eval/eval_dev.jsonl" "r3h=$SF/r3h/eval/eval_test.jsonl,$SF/r3h/eval/eval_dev.jsonl"; do
  $P scripts/sciworld_stepfix_compare.py --a $rf --b r1big=$D/eval/eval_test.jsonl,$D/eval/eval_dev.jsonl | head -1 | tee -a $SF/queue.log; done
slog "r1big DONE"
