#!/bin/bash
# User-requested (Oct 6 23:30, "我想看看R2用A0（就是不更新A1）会怎么样"): the original Oct-4 main chain's R2 (selfattr/r2, 531/774) with the
# step selector replaced by the NON-updated A0 (reports/alfworld/self_attr/r0/both/adapter) instead of the R2-retrained attributor.
# Everything else identical to selfattr/r2: same R2 trajectories (collected by selfattr/r1), same failures (352), same 7B Model A diagnosis
# for the teacher, hindsight + diagnosis + own analysis teacher (R1 adapter), ALFWorld-specific one-step evidence, env top-86, the same 86
# replay rows, OLD trainer (the one selfattr/r2 used), 774 evaluation.  Starts only after the trainer diagnostics are done (user priority).
source scripts/lib_baselines.sh; O=$SA/selfattr/r2; D=$SA/selfattr_r2A0/r2; T=$D/traj.jsonl; R1A=$SA/selfattr/r1/model/adapter; A0=reports/alfworld/self_attr/r0/both/adapter
mkdir -p $D/self $D/proxy
until grep -q "trainer diagnostics DONE" $SA/queue.log && [ -s $E/eval_heldout_diag_d_seed1.jsonl ]; do sleep 60; done
for f in traj.jsonl traj.release.jsonl ids_fail.jsonl diag_modelA.raw.jsonl replay.jsonl; do [ -s $D/$f ] || cp $O/$f $D/$f; done
log "selfattr_r2A0: start (R2 of the original chain with A0 steps instead of the retrained attributor)"
until [ -s $D/pointer.raw.jsonl ] && grep -q "DONE pointer" $D/pointer.log 2>/dev/null; do $(S 12000) $P -u scripts/alfworld_self_attributor.py pointer --traj $T --ids $D/ids_fail.jsonl --adapter $A0 --out $D/pointer.jsonl > $D/pointer.log 2>&1 || sleep 60; done
cp $D/pointer.jsonl $D/self/steps.jsonl
$P -c "import json,statistics as s; ts=[json.loads(l)['t_hat'] for l in open('$D/pointer.raw.jsonl')]; print('=== selfattr_r2A0: A0 pointer %d states, mean t %.1f, t0 %.2f (retrained attributor in selfattr/r2: mean t 2.7, t0 0.15)' % (len(ts), s.mean(ts), sum(t==0 for t in ts)/len(ts)))" | tee -a $SA/queue.log
if ! grep -q "DONE informed_teacher" $D/teacher.log 2>/dev/null; then
  for i in 0 1 2; do ( until grep -q "DONE informed_teacher" $D/teacher_s$i.log 2>/dev/null; do $(S 12000) $P -u scripts/self_teacher_informed.py --traj $T --steps $D/self/steps.jsonl --diag $D/diag_modelA.raw.jsonl --adapter $R1A --out $D/self/targets_s$i.jsonl --shard $i 3 >> $D/teacher_s$i.log 2>&1 || sleep 30; done ) & sleep 60; done; wait
  cat $D/self/targets_s*.jsonl > $D/self/targets.jsonl; echo "DONE informed_teacher (merged 3 shards)" >> $D/teacher.log
fi
if ! grep -q "DONE onestep" $D/onestep.log 2>/dev/null; then
  for i in $(seq 0 11); do ( until grep -q "DONE onestep" $D/onestep_s$i.log 2>/dev/null; do SE_DIR=$D SE_TRAJ=$T $P -u scripts/proxy_eval_se3b.py onestep --arms self --limit 300 --shard $i 12 >> $D/onestep_s$i.log 2>&1; sleep 3; done ) & done; wait
  cat $D/proxy/onestep_s*.jsonl >> $D/proxy/onestep.jsonl; rm -f $D/proxy/onestep_s*.jsonl; echo "DONE onestep (merged 12 shards)" >> $D/onestep.log
fi
[ -s $D/targets_env_top/train.jsonl ] || SE_DIR=$D SE_TRAJ=$T $P scripts/proxy_arms_se3b.py --arm self --rule env_top --k 86 --out $D/targets_env_top | tee -a $SA/queue.log
until [ -f $D/model/adapter/adapter_config.json ]; do $(S 24000) $P -u scripts/alfworld_opd_plus_v1_backup_oct6.py train --data $D/targets_env_top/train.jsonl:1.0 $D/replay.jsonl:1.0 --student Qwen/Qwen2.5-3B-Instruct --adapter $PWD/$R1A --lr 1e-5 --epochs 1 --batch 8 --seed 0 --out $D/model > $D/train.log 2>&1 || { log "selfattr_r2A0: TRAIN FAILED (retry)"; sleep 180; }; done
[ -s $E/eval_heldout_r2A0.jsonl ] || eval774 $D r2A0
log "selfattr_r2A0 DONE (compare: selfattr/r2 retrained attributor 531, s1/s2 R2 log rule 464/500, R1 475)"
