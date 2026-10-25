#!/bin/bash
# Oct 11 11:05 (user-approved, "scheme A"): RL + correction. One StepFix round (final recipe, KL-anchored replay) on top of the GiGPO
# checkpoint at step 24 (1,536 RL episodes; 328 wins). Base = merged student10 (the RL start point), adapter = extracted GiGPO LoRA.
# +1,483 collection episodes -> ~3,019 total, compared with GiGPO step 48 (3,072 episodes; 392 wins).
source scripts/lib_sciworld_stepfix.sh; export GMEM=9000; NS=24
BASE=/scratch.global/pan00389/error-attribution/serl_work/models/qwen25_3b_sciworld_student10_merged
GA=/scratch.global/pan00389/error-attribution/serl_work/outputs/sciworld_gigpo/lora_extracted/global_step_24/actor/lora_adapter
D=$SF/ga1; E=$D/eval; mkdir -p $E; SEL="--replay_mult 2.4 --min_s 2 --commit_guard --cap_mult 1000"
slog "ga1: start (StepFix round on GiGPO step 24)"
if [ ! -s $D/traj.jsonl ]; then collect_sw $D/col $GA 1 ga1 $NS --per_task 75 --model $BASE
  $P scripts/sciworld_stepfix.py prep --dir $D --inputs "$D/col/traj_s*.jsonl" --expect 1483 | sed "s/^/ga1 /" | tee -a $SF/queue.log
  $P -c "
L=open('$D/traj.release.jsonl').readlines()
for i in range(6): open('$D/rel_s%d.jsonl'%i,'w').writelines(L[i::6])"; fi
if [ ! -s $D/pointer.jsonl ]; then pids=""; for i in $(seq 0 5); do ( until grep -q "^DONE" $D/modelA_s$i.log 2>/dev/null; do $(SG $GMEM) $P -u scripts/alfworld_attribute.py --attributor reports/attributor_no_observations/final --variant no_observations --release $D/rel_s$i.jsonl --out $D/modelA_pred_s$i.jsonl >> $D/modelA_s$i.log 2>&1 || sleep 30; done ) & pids="$pids $!"; sleep 20; done; wait $pids
  $P scripts/sciworld_stepfix.py pointer --dir $D | sed "s/^/ga1 /" | tee -a $SF/queue.log; fi
if ! grep -q "DONE teacher (merged)" $D/teacher.log 2>/dev/null; then pids=""
  for i in $(seq 0 31); do ( until grep -q "DONE teacher" $D/teacher_s$i.log 2>/dev/null; do $(SG 12000) $P -u scripts/sciworld_stepfix.py teacher --model $BASE --dir $D --adapter $GA --shard $i 32 --seed 1 --accept_env --max_extra 16 --full_view --top 4 >> $D/teacher_s$i.log 2>&1 || sleep 30; done ) & pids="$pids $!"; sleep 8; done; wait $pids
  echo "DONE teacher (merged)" >> $D/teacher.log; fi
[ -s $D/targets_cap.jsonl ] || $P scripts/sciworld_select_capped.py --dir $D --out $D $SEL | tee $D/select.log | sed "s/^/ga1 select /" | cut -c1-400 | tee -a $SF/queue.log
until [ -f $D/model/adapter/adapter_config.json ]; do $(SG 24000) $P -u scripts/sciworld_stepfix.py train --model $BASE --data $D/targets_cap.jsonl $D/replay_ret.jsonl --adapter $GA --out $D/model --seed 1 --kl_replay >> $D/train.log 2>&1 || sleep 30; done
slog "ga1: trained ($(tail -1 $D/model/train_log.jsonl))"
pids=""; for i in $(seq 0 31); do shard_loop $E/eval_test_w$i.log scripts/sciworld_agent.py collect --model $BASE --adapter $D/model/adapter --split test --per_task 30 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 32 --out $E/eval_test_w$i.jsonl --tag ga1_test & pids="$pids $!"; sleep 5; done
for i in $(seq 0 7); do shard_loop $E/eval_dev_w$i.log scripts/sciworld_agent.py collect --model $BASE --adapter $D/model/adapter --split dev --per_task 7 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 8 --out $E/eval_dev_w$i.jsonl --tag ga1_dev & pids="$pids $!"; sleep 5; done
wait $pids; cat $E/eval_test_w*.jsonl > $E/eval_test.jsonl; cat $E/eval_dev_w*.jsonl > $E/eval_dev.jsonl; slog "ga1: 848 eval DONE"
RB=reports/sciworld/rl_baselines; for t in gigpo_s24 gigpo_s48; do cat $RB/$t/eval_test_s*.jsonl > $D/_ref_${t}_test.jsonl; cat $RB/$t/eval_dev_s*.jsonl > $D/_ref_${t}_dev.jsonl; $P scripts/sciworld_stepfix_compare.py --a $t=$D/_ref_${t}_test.jsonl,$D/_ref_${t}_dev.jsonl --b ga1=$E/eval_test.jsonl,$E/eval_dev.jsonl | head -1 | tee -a $SF/queue.log; done
slog "ga1 DONE"
