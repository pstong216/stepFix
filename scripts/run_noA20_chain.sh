#!/bin/bash
# Ablation A3 (autonomous, Oct 7 23:00; user authorized paper-needed runs): the FINAL recipe WITHOUT Model A.
# Step selection = environment adjudication (first-step + late proposals, per failure the state with the largest verified improvement,
# STEP_POOL=1), teacher = hindsight + own analysis (no diagnosis), generic one-step evidence, K = 20% of failures + 100 replay, new
# trainer, R1 -> R3, 774 eval.  Identical to the final chain (a3bfinal_k20) except the step selector and the teacher's diagnosis.
# R1 reuses the adjudication chain's R1 pointer/teacher/one-step outputs (selfattr_pool/r1: same inputs); only filter/train/eval rerun.
source scripts/lib_baselines.sh
export STEP=logfix STEP_POOL=1 TEACHER_NO_DIAG=1 K_FRAC=0.2 REPLAY_N=100 EVIDENCE=generic PAR=3
CH=selfattr; SUF=_noA20; A0=reports/alfworld/self_attr/r0/both/adapter; C=$SA/$CH$SUF
D=$C/r1; mkdir -p $D
if [ ! -s $D/self/targets.jsonl ]; then F=$SA/selfattr_pool/r1
  cp -r $F/self $F/proxy $F/envlab $F/attr_labels $D/ 2>/dev/null; for f in pointer.jsonl pointer.raw.jsonl diag.raw.jsonl ids_fail.jsonl teacher.log onestep.log; do cp $F/$f $D/$f; done
  rm -rf $D/proxy/targets* 2>/dev/null; log "selfattr_noA20 r1: reused adjudication R1 pointer/teacher/one-step from $F"; fi
S=$STU
for R in 1 2 3; do D=$C/r$R; mkdir -p $D
  if [ ! -f $D/ADAPTER ]; then
    [ "$R" != "1" ] && collect4 $D $S selfattr_noA20_r$R
    CHAIN_SUFFIX=$SUF SEED=0 bash scripts/self_attr_round.sh $CH $R $S $A0 any || { log "selfattr_noA20 r$R FAILED"; exit 1; }
  fi
  [ -s $E/eval_heldout_noA20_r$R.jsonl ] || eval774 $D noA20_r$R
  S=$(cat $D/ADAPTER)
done
log "selfattr_noA20 R1->R3 DONE"
