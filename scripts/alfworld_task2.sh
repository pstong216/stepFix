#!/bin/bash
# Second task file (the first is executing and must not be edited): round-2 controls and the round-3 attributor refresh.
# Usage: alfworld_task2.sh <const1_r2|attralf_r3> <hold>
T=$1; H=$2; cd /users/8/pan00389/multi-agent/error-attribution; A=reports/alfworld/opd/student3b_low100; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; SR="srun --jobid=$H --overlap -n1 -c4 --mem=0"
mkdir reports/alfworld/locks/$T 2>/dev/null || { echo "task $T already taken"; exit 0; }
echo "=== $(date) task $T START on hold $H ($(squeue -j $H -h -o %N))" | tee -a reports/alfworld/dispatch.log
run_distill_loop() { local T2=$1 TG=$2 FROM=$3 OUT=$4 L=$5; shift 5; local n0; n0=$(grep -c "DONE distill" $L 2>/dev/null || echo 0)
  for chunk in 1 2 3 4 5; do
    $SR $P -u scripts/alfworld_attr_opd.py distill --traj $T2 --targets $TG --student Qwen/Qwen2.5-3B-Instruct --adapter $FROM --out $OUT --verify_k 2 --exclude_a_minus --max_examples_per_run 250 "$@" >> $L 2>&1 || return 1
    [ "$(grep -c "DONE distill" $L)" -gt "$n0" ] && return 0
  done; return 1; }
eval_unseen() { $SR $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $1 --split valid_unseen --budget 50 --temperature 0 --out $2 --tag $3 >> $4 2>&1; echo "=== $(date) $3: $(grep -c '"won": true' $2)/$(wc -l < $2) unseen" | tee -a $4 reports/alfworld/dispatch.log; }
case $T in
  const1_r2)   # attribution-free control at round 2: same start adapter (attralf round 1), same 575 failures, same teacher; pointer = always step 1
    O=$A/const1_r2; mkdir -p $O; T2=reports/alfworld/student3b_low100_round2_attralf_nofb_train_b50.jsonl; L=$O/run.log
    echo "=== $(date) const1_r2 on hold $H: const step 1 on the attralf round-2 failure set, distill from attralf_nofb/round1" >> $L
    $P -u scripts/alfworld_attr_opd.py choose --traj $T2 --arm const --const_step 1 --ids_from $A/attralf_nofb/steps_round2.jsonl --max_failures 1000 --out $O/steps.jsonl >> $L 2>&1 || exit 1
    $SR $P -u scripts/alfworld_attr_opd.py teacher --traj $T2 --steps $O/steps.jsonl --teacher Qwen/Qwen2.5-3B-Instruct --teacher_adapter reports/alfworld/student3b_sft/final --out $O/targets.jsonl >> $L 2>&1 || exit 1
    run_distill_loop $T2 $O/targets.jsonl $A/attralf_nofb/round1/adapter $O/round2 $L || exit 1
    eval_unseen $O/round2/adapter $O/eval_unseen_round2.jsonl opd_const1_r2 $L
    scripts/alfworld_eval_extended.sh $H opd_const1_r2 $O/round2/adapter;;
  attralf_r3)   # per-round attributor refresh (v2 data adds the round-2 failure sets), then round 3 of attralf_nofb with the new attributor
    L=reports/alfworld/attributor_alf_v2.log; echo "=== $(date) attributor v2 on hold $H" >> $L
    $P -u scripts/build_attributor_alf_data.py --out data/attributor_sft_alf_v2 >> $L 2>&1 || exit 1
    $SR $P -u scripts/finetune_attributor_alf.py --init reports/attributor_7b_no_observations/final --data data/attributor_sft_alf_v2/train.jsonl --out reports/attributor_7b_alf_v2 >> $L 2>&1 || exit 1
    $SR $P -u scripts/alfworld_attribute.py --attributor reports/attributor_7b_alf_v2/final --variant no_observations --release reports/alfworld/student3b_low100_round3_oracle_train_b50.release.jsonl --out reports/alfworld/attr7balfv2_round3_oracle.jsonl >> $L 2>&1
    $P - <<'PY' >> $L 2>&1
import json
lab={r["id"]:r for r in map(json.loads,open("reports/alfworld/progress_student3b_low100_round3_oracle.jsonl"))}; lab={k:v for k,v in lab.items() if v.get("t_first_regress") is not None}
for pre in ["attr7bnoobs","attr7balf","attr7balfv2"]:
    pred={r["id"]:r for r in map(json.loads,open(f"reports/alfworld/{pre}_round3_oracle.jsonl"))}; ids=[i for i in lab if i in pred and pred[i].get("t_hat") is not None]
    d=[pred[i]["t_hat"]-lab[i]["t_first_regress"] for i in ids]; n=len(d); print(f"HELDOUT oracle_r3 {pre}: n={n} exact {100*sum(x==0 for x in d)/n:.1f}% within2 {100*sum(abs(x)<=2 for x in d)/n:.1f}% mean|diff| {sum(abs(x) for x in d)/n:.1f}")
PY
    echo "ATTRIBUTOR_V2_DONE" >> $L
    ATTRIBUTOR=reports/attributor_7b_alf_v2/final scripts/launch_alfworld_roundN_v2.sh $H attralf_nofb round2 round3
    scripts/alfworld_eval_extended.sh $H opd_attralf_nofb_r3 $A/attralf_nofb/round3/adapter;;
  *) echo "unknown task $T"; exit 1;;
esac
echo "=== $(date) task $T FINISHED on hold $H" | tee -a reports/alfworld/dispatch.log
