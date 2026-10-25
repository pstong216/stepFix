#!/bin/bash
# Oct 10 ~11:40 (main approved): three cheap single-round 7B fillers on the main chain's R2 data (a7b_v3b/r2: the same 214 7B Model A
# states on a7b_final/r2's 1500 games, the same full-view teacher candidates, the same fail_weighted replay file), all starting from the
# same R1 adapter (a7b_final/r1 = a7b_r1, 549), same trainer (CKPT_EVERY=50) and 774 eval; compared with a7bv3b_r2 = 576 and a7b_r1 = 549.
#   rand   (A) K = round(0.2 x 214 failures) = 43 RANDOM teacher top-1 targets (rule all_top1, random.Random(200)) + the same replay. K = 43
#              matches the main chain's K (its env_top_margin set has 43 targets).                                        tag a7bv3b_r2_rand
#   norep  (C) the main chain's env_top_margin targets only (43), no replay.                                               tag a7bv3b_r2_norep
#   nofilt (B) every teacher top-1 target (214) + the same replay.                                                         tag a7bv3b_r2_nofilt
# Nothing in a7b_v3b/r2 is written; outputs go to self_attr_rounds/a7b_v3b_fill/<mode>.
MODE=$1; case $MODE in rand|norep|nofilt) ;; *) echo "usage: $0 rand|norep|nofilt"; exit 2;; esac
export MODEL=Qwen/Qwen2.5-7B-Instruct GPU_MIN=20000 TRAIN_MIN=36000 EVIDENCE=generic HOLD_NAME_RE='^(a7b_gpu|error_attr_hold_.*)$' CKPT_EVERY=50
source scripts/lib_baselines.sh
# Oct 10 03:35: a running 7B training (e.g. the RFT R3 training) holds only ~20 GB between its peaks, so GPU_MIN=20000 alone did NOT keep
# this driver off its card (a teacher shard landed next to it). Local override of pick(): never use a (job, node) on which ANY
# alfworld_opd_plus.py train process (launched from this node) is running. lib_baselines.sh itself is unchanged.
# Oct 10 03:50: the 7B --full_view teacher grows to ~30 GB, so two processes on one 40 GB card OOMed each other: the teacher now needs
# 32 GB free, and no other process is placed on a card that runs a full-view teacher.
_pick_raw() { local best="" bm=0 busy; busy=$(pgrep -af "srun --jobid=[0-9]+ .*(alfworld_opd_plus.py train|self_teacher_informed.py.*--full_view)" | sed -nE 's/.*--jobid=([0-9]+)( .* -w ([^ ]+))? .*/\1:\3/p')
  for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk -v re="${HOLD_NAME_RE:-^error_attr_hold_}" '$2 ~ re && $3=="RUNNING"{print $1}'); do for N in $(scontrol show hostnames $(squeue -h -j $H -o %N) 2>/dev/null); do
    echo "$busy" | grep -qxE "$H:($N)?" && continue
    [ -e $CL/${H}_$N ] && [ $(( $(date +%s) - $(stat -c %Y $CL/${H}_$N) )) -lt 90 ] && continue
    local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 -w $N nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best="$H $N"; }; done; done; [ "$bm" -ge ${1:-12000} ] 2>/dev/null && echo $best; }
# Oct 10 04:05: two drivers polling at the same minute both picked the same empty card (OOM). Picks are now serialized with flock, and a
# card just handed out is skipped for 90 s (until its process has loaded and shows up in nvidia-smi).
CL=$SA/.ablv3_claims; mkdir -p $CL
pick() { ( flock 9; local r; r=$(_pick_raw "$@"); [ -n "$r" ] && touch $CL/${r// /_}; echo $r ) 9>/dev/shm/ablv3_pick.lock; }
V=$SA/a7b_v3b/r2; T=$SA/a7b_final/r2/traj.jsonl; START=$SA/a7b_final/r1/model/adapter; D=$SA/a7b_v3b_fill/$MODE; TAG=a7bv3b_r2_$MODE; mkdir -p $D
[ -f $D/model/adapter/adapter_config.json ] && [ -s $E/eval_heldout_$TAG.jsonl ] && exit 0
log "a7b_v3b_fill $MODE: start (data = a7b_v3b/r2, from $START)"
if [ $MODE != norep ]; then
  [ -s $D/targets_all_top1/train.jsonl ] || SE_DIR=$V SE_TRAJ=$T $P scripts/proxy_arms_se3b.py --arm self --rule all_top1 --out $D/targets_all_top1 | tee -a $SA/queue.log
fi
NF=$(grep -c . $V/ids_fail.jsonl); K=$($P -c "print(max(1, round(0.2 * $NF)))")
case $MODE in
  rand) mkdir -p $D/targets_rand; [ -s $D/targets_rand/train.jsonl ] || $P -c "import random; L=open('$D/targets_all_top1/train.jsonl').readlines(); random.Random(200).shuffle(L); open('$D/targets_rand/train.jsonl','w').writelines(L[:$K])"
        DATA="$D/targets_rand/train.jsonl:1.0 $V/replay.jsonl:1.0";;
  norep) DATA="$V/targets_env_top_margin/train.jsonl:1.0";;
  nofilt) DATA="$D/targets_all_top1/train.jsonl:1.0 $V/replay.jsonl:1.0";;
esac
log "a7b_v3b_fill $MODE: K = 0.2 x $NF = $K; training data: $(for x in $DATA; do f=${x%:*}; echo -n "$(basename $(dirname $f))/$(basename $f)=$(grep -c . $f) "; done)"
train1 $D $START "$DATA"; [ -s $E/eval_heldout_$TAG.jsonl ] || eval774 $D $TAG
[ -f $D/model/adapter/adapter_config.json ] || { log "a7b_v3b_fill $MODE ended WITHOUT a model"; exit 1; }
log "a7b_v3b_fill $MODE DONE ($TAG)"
