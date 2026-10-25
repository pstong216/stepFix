#!/bin/bash
# One Attr-OPD round for one arm on GPU hold $1: choose -> teacher targets (14B) -> verified distill (3B SFT student) -> greedy eval on unseen.
# Usage: run_alfworld_opd_arm.sh <hold> <arm> [max_failures=400] [verify_k=2] [extra distill args]
# Env: TAG (student tag, default student3b_low100), TEACHER (HF id, default Qwen/Qwen2.5-14B-Instruct), TEACHER_ADAPTER (optional LoRA dir), ROUND (default round1)
J=$1; ARM=$2; MAXF=${3:-400}; VK=${4:-2}; shift 4 2>/dev/null; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=${TAG:-student3b_low100}; TEACHER=${TEACHER:-Qwen/Qwen2.5-14B-Instruct}; ROUND=${ROUND:-round1}
T=reports/alfworld/${TAG}_train_b50.jsonl; O=reports/alfworld/opd/$TAG/$ARM; mkdir -p $O; L=$O/run.log; SR="srun --jobid=$J --overlap -n1 -c4 --mem=0"
TA=""; [ -n "${TEACHER_ADAPTER:-}" ] && TA="--teacher_adapter $TEACHER_ADAPTER"
IDS=""; [ -f reports/alfworld/progress_${TAG}_train_b50.jsonl ] && IDS="--ids_from reports/alfworld/progress_${TAG}_train_b50.jsonl"   # same (oracle-labelled) failure set for every arm
[ -n "${IDS_FROM:-}" ] && IDS="--ids_from $IDS_FROM"   # override: e.g. the attr arm's steps.jsonl so every arm trains on exactly the same failures
echo "=== $(date) arm $ARM hold $J node $(squeue -j $J -h -o %N) max_failures $MAXF verify_k $VK extra: $*" | tee -a $L
NOFB=0; [[ "$ARM" == *_nofb* ]] && NOFB=1; ARM0=${ARM/_nofb/}
CV=""; [[ "$ARM0" == *_cv* ]] && CV="--contrastive"; ARM0=${ARM0/_cv/}   # *_cv = contrastive verification: skip states whose original action also wins
OFF=""; [[ "$ARM0" =~ _p([0-9]+)$ ]] && OFF="--offset ${BASH_REMATCH[1]}" && ARM0=${ARM0%_p*}; [[ "$ARM0" =~ _m([0-9]+)$ ]] && OFF="--offset -${BASH_REMATCH[1]}" && ARM0=${ARM0%_m*}   # oracle_nofb_p2 = oracle step +2, _m2 = -2 (imprecision controls)
BASE_ARM=${ARM0%%_w*}; WIN=""; [[ "$ARM0" == *_w* ]] && WIN="--window ${ARM0##*_w}"   # e.g. attr_w1 = attr step +-1; attr_nofb = attr without unverified fallback
FB="--fallback_unverified"; [ "$NOFB" = 1 ] && FB=""
CHOOSE_ARM=$BASE_ARM
case $BASE_ARM in
  attr)    CH="--attr reports/alfworld/attr7bnoobs_${TAG}_train_b50.jsonl";;
  tstar)   CH="--tstar reports/alfworld/oracle_${TAG}_train_b50.jsonl";;
  attralf) CH="--attr reports/alfworld/attr7balf_${TAG}_train_b50.jsonl"; CHOOSE_ARM=attr;;   # Model A fine-tuned on ALFWorld self-play labels
  oracle)  CH="--progress reports/alfworld/progress_${TAG}_train_b50.jsonl";;
  uniform) CH="--max_teacher_calls $((MAXF))";;   # equal teacher budget: as many supervised states as the attr arm has failures
  random_matched) CH="--progress reports/alfworld/progress_${TAG}_train_b50.jsonl";;   # depth-matched random: resample from the real pointer's own empirical position distribution
  const*)  CH="--const_step ${BASE_ARM#const}"; CHOOSE_ARM=const;;   # const1 = always step 1 (no attribution; the early-error prior)
  *)       CH="";;
esac
$P -u scripts/alfworld_attr_opd.py choose --traj $T --arm $CHOOSE_ARM $CH $IDS $WIN $OFF --max_failures $MAXF --out $O/steps.jsonl >> $L 2>&1 || exit 1
$SR $P -u scripts/alfworld_attr_opd.py teacher --traj $T --steps $O/steps.jsonl --teacher $TEACHER $TA --out $O/targets.jsonl >> $L 2>&1 || exit 1
N0=$(grep -c "DONE distill" $L)   # run.log is shared across rounds/variants: only a NEW marker ends the chunk loop
for chunk in 1 2 3 4 5 6 7 8; do   # chunks of 250 examples per process (host RAM grows ~45 MB per env creation)
  $SR $P -u scripts/alfworld_attr_opd.py distill --traj $T --targets $O/targets.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter reports/alfworld/$TAG/final --out $O/$ROUND --verify_k $VK $FB $CV --exclude_a_minus --max_examples_per_run 250 "$@" >> $L 2>&1 || exit 1
  [ "$(grep -c "DONE distill" $L)" -gt "$N0" ] && break
done
$SR $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $O/$ROUND/adapter --split valid_unseen --budget 50 --temperature 0 --out $O/eval_unseen_$ROUND.jsonl --tag opd_$ARM >> $L 2>&1
echo "=== $(date) arm $ARM $ROUND finished: $(grep -E '^DONE' $L | tail -1)" | tee -a $L
