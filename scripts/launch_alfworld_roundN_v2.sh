#!/bin/bash
# Round 2 (iterated Attr-OPD) for one arm on hold $1: recollect T=1 failures with the round-1 adapter -> attribute/oracle -> distill from round-1 adapter -> eval.
J=$1; ARM=$2; FROM=${3:-round1}; TO=${4:-round2}; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
NOFB=0; [[ "$ARM" == *_nofb* ]] && NOFB=1; ARM0=${ARM/_nofb/}; FB="--fallback_unverified"; [ "$NOFB" = 1 ] && FB=""   # *_nofb arms: verified targets only
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=student3b_low100; O=reports/alfworld/opd/$TAG/$ARM; R1=$O/$FROM/adapter; T2=reports/alfworld/${TAG}_${TO}_${ARM}_train_b50.jsonl; L=$O/$TO.log; SR="srun --jobid=$J --overlap -n1 -c4 --mem=0"
echo "=== $(date) $TO $ARM on hold $J ($(squeue -j $J -h -o %N)): collect with $R1" | tee -a $L
for c in 1 2 3 4; do   # chunks of 450 episodes per process (host-RAM leak per env creation)
  $SR $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $R1 --split train --n_games 1500 --budget 50 --temperature 1.0 --out $T2 --tag ${TO}_$ARM --max_episodes_per_run 450 >> $L 2>&1
  grep -q "^DONE" <(tail -3 $L) && break
done
NF=$(grep -c '"won": false' $T2); echo "=== $(date) $NF failures" | tee -a $L
CHOOSE_ARM=$ARM0
if [ "$ARM0" = "attr" ]; then
  $SR $P -u scripts/alfworld_attribute.py --attributor reports/attributor_7b_no_observations/final --variant no_observations --release ${T2%.jsonl}.release.jsonl --out reports/alfworld/attr7bnoobs_${TAG}_${TO}_${ARM}.jsonl >> $L 2>&1
  CH="--attr reports/alfworld/attr7bnoobs_${TAG}_${TO}_${ARM}.jsonl"
elif [ "$ARM0" = "attralf" ]; then   # refreshed attributor (ALFWorld self-play labels); ATTRIBUTOR env overrides the adapter dir
  AD=${ATTRIBUTOR:-reports/attributor_7b_alf/final}
  $SR $P -u scripts/alfworld_attribute.py --attributor $AD --variant no_observations --release ${T2%.jsonl}.release.jsonl --out reports/alfworld/attr7balf_${TAG}_${TO}_${ARM}.jsonl >> $L 2>&1
  CH="--attr reports/alfworld/attr7balf_${TAG}_${TO}_${ARM}.jsonl"; CHOOSE_ARM=attr
else
  JID=$(sbatch --parsable --export=ALL,TRAJ=$T2,OUT=reports/alfworld/progress_${TAG}_${TO}_${ARM}.jsonl,N=100000,EXTRA="--progress" slurm/cpu_oracle.sbatch); echo "progress oracle job $JID" | tee -a $L
  for i in $(seq 1 150); do [ -f reports/alfworld/progress_${TAG}_${TO}_${ARM}.jsonl ] && [ $(wc -l < reports/alfworld/progress_${TAG}_${TO}_${ARM}.jsonl) -ge $((NF * 90 / 100)) ] && break; sleep 60; done
  CH="--progress reports/alfworld/progress_${TAG}_${TO}_${ARM}.jsonl"
fi
$P -u scripts/alfworld_attr_opd.py choose --traj $T2 --arm $CHOOSE_ARM $CH --max_failures 1000 --out $O/steps_$TO.jsonl >> $L 2>&1 || exit 1
$SR $P -u scripts/alfworld_attr_opd.py teacher --traj $T2 --steps $O/steps_$TO.jsonl --teacher Qwen/Qwen2.5-3B-Instruct --teacher_adapter reports/alfworld/student3b_sft/final --out $O/targets_$TO.jsonl >> $L 2>&1 || exit 1
for chunk in 1 2 3 4 5 6; do
  $SR $P -u scripts/alfworld_attr_opd.py distill --traj $T2 --targets $O/targets_$TO.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter $R1 --out $O/$TO --verify_k 2 $FB --exclude_a_minus --max_examples_per_run 250 >> $L 2>&1 || exit 1
  grep -q "DONE distill" $L && break
done
$SR $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $O/$TO/adapter --split valid_unseen --budget 50 --temperature 0 --out $O/eval_unseen_$TO.jsonl --tag ${TO}_$ARM >> $L 2>&1
echo "=== $(date) ROUND_DONE $TO $ARM: $(grep -E '^DONE [0-9]+ episodes' $L | tail -1)" | tee -a $L
