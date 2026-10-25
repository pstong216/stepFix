#!/bin/bash
# 7B self-evolve v4: all 178 failures; env evidence as VETO + hindsight-teacher ranking + verb cap; DPO vs the base model
# (no anchor needed); placebo arm; eval on unseen + seen (274 games). Two A40s.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; O=reports/alfworld/self_evolve7b; M=$O/queue.log
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SE_DIR=$O SE_TRAJ=$O/r1_train.jsonl SE_MODEL=Qwen/Qwen2.5-7B-Instruct SE_ADAPTER= SE_ATTR=$O/attr7b.jsonl
J1=2467981; J2=2467982; S1="srun --jobid=$J1 --overlap -n1 -c4 --mem=0"; S2="srun --jobid=$J2 --overlap -n1 -c4 --mem=0"; SC="srun --jobid=2239109 --overlap -n1 -c2 --mem=24G"
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
log "v4b: re-scoring self-trained attributor on all 178 (previous run hit OOM)"
until [ "$(wc -l < $O/selfprog_scores.jsonl)" -ge 178 ]; do
  $S1 $P -u scripts/alfworld_progress_model.py score --traj $O/r1_train.jsonl --student Qwen/Qwen2.5-7B-Instruct --adapter $O/selfprog_model/adapter --ids_from $O/ids_all.jsonl --out $O/selfprog_scores.jsonl >> $O/selfprog.log 2>&1 || sleep 60
done
$P scripts/alfworld_progress_model.py pointer --scores $O/selfprog_scores.jsonl --out $O/selfprog_pointer.jsonl --delta 1.0 >> $O/selfprog.log 2>&1
$P scripts/self_evolve.py choose >> $M 2>&1 && $P scripts/self_evolve.py targets >> $M 2>&1 || { log "choose/targets FAILED"; exit 1; }
for arm in rand_early entropy attr hkl selfattr selfprog; do
  rm -f $O/$arm/onestep.log.v4done
  until grep -q "DONE onestep" <(tail -2 $O/$arm/onestep.log 2>/dev/null) && [ -f $O/$arm/onestep.log.v4done ]; do
    $SC $P -u scripts/env_evidence_filter.py --teacher_dir $O/$arm --traj $O/r1_train.jsonl onestep >> $O/$arm/onestep.log 2>&1 && touch $O/$arm/onestep.log.v4done; sleep 3; done
done
$P scripts/self_evolve.py select_v2 >> $M 2>&1 || { log "select_v2 FAILED"; exit 1; }
$S1 $P -u scripts/self_evolve.py refs --arms placebo,rand_early,entropy,attr > $O/refs_a.log 2>&1 &
$S2 $P -u scripts/self_evolve.py refs --arms hkl,selfattr,selfprog > $O/refs_b.log 2>&1 &
wait; log "v4 DPO training + eval (unseen + seen)"
run() {
  local a=$1 S=$2; local D=$O/$a
  rm -rf $D/model_v4; rm -f $D/eval_unseen_v4.jsonl $D/eval_seen_v4.jsonl $D/eval_*_v4.release.jsonl
  $S $P -u scripts/alfworld_opd_plus.py train --data $D/dpo_pairs.jsonl --student Qwen/Qwen2.5-7B-Instruct --adapter "" --lr 2e-5 --epochs 2 --beta 0.1 --out $D/model_v4 > $D/train_v4.log 2>&1 || { echo "=== v4 $a train FAILED" >> $M; return; }
  for sp in valid_unseen valid_seen; do
    $S $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-7B-Instruct --adapter $D/model_v4/adapter --split $sp --budget 50 --temperature 0 --out $D/eval_${sp#valid_}_v4.jsonl --tag se7b4_$a >> $D/train_v4.log 2>&1
  done
  echo "=== $(date '+%F %T') v4 $a: $(wc -l < $D/dpo_pairs.jsonl) pairs | unseen $(grep -c '"won": true' $D/eval_unseen_v4.jsonl)/$(wc -l < $D/eval_unseen_v4.jsonl) seen $(grep -c '"won": true' $D/eval_seen_v4.jsonl)/$(wc -l < $D/eval_seen_v4.jsonl)" >> $M
}
(run placebo "$S1"; run entropy "$S1"; run hkl "$S1"; run selfprog "$S1") &
(run attr "$S2"; run rand_early "$S2"; run selfattr "$S2") &
wait; log "v4 ALL DONE"
