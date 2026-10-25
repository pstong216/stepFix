#!/usr/bin/env bash
# Evaluation of the ScienceWorld online-RL baselines (GiGPO / SERL, Oct 10 2026) with OUR harness, exactly the StepFix 848 protocol
# (scripts/lib_sciworld_stepfix.sh eval_sw): scripts/sciworld_agent.py collect, greedy (T=0), budget 50, --snap safe,
#   test 651 = --split test --per_task 30 --seed 0 (12 round-robin shards) + dev 197 = --split dev --per_task 7 --seed 0 (4 shards),
# with --model = the merged student10 base (the RL start point) and --adapter = the RL LoRA extracted from the verl checkpoint.
#   usage (tmux on cn1019):  bash scripts/sciworld_rl_eval.sh RUN_DIR STEP TAG
#     RUN_DIR = serl_work/outputs/sciworld_gigpo (checkpoints/global_step_<STEP>/actor), TAG = e.g. gigpo_s8
#   results: reports/sciworld/rl_baselines/<TAG>/eval_test.jsonl + eval_dev.jsonl ; summary via scripts/sciworld_rl_summary.py
# GPUs: the reserved eval job (sw_rl_gpu 3125925) first, then error_attr_hold_* / tcded_gpu / a7b_gpu GPUs only if >= 16 GB free (coordinator, Oct 10 14:40) (other agents' ScienceWorld
# runs have priority there); jobs listed in reports/sciworld/rl_baselines/.skip_jobs (e.g. the GPU that trains) are never used.
set -u
cd /users/8/pan00389/multi-agent/error-attribution
R=$1; N=$2; TG=$3
P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; SPY=/scratch.global/pan00389/error-attribution/envs/serl/bin/python
W=/scratch.global/pan00389/error-attribution/serl_work; BASE=$W/models/qwen25_3b_sciworld_student10_merged
RB=reports/sciworld/rl_baselines; D=$RB/$TG; mkdir -p $D
export ERROR_ATTR_ROOT=$PWD HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CK=$R/checkpoints; A=$CK/global_step_$N/actor; X=$R/lora_extracted/global_step_$N/actor/lora_adapter
log() { echo "=== $(date '+%F %T') $*" | tee -a $RB/queue.log; }
if [ "$N" = 0 ]; then ADARGS=()   # step 0 = the RL start point itself (merged student10, no adapter): the merge/numerics reference
else
  until [ "$(cat $CK/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)" -ge "$N" ] 2>/dev/null && [ -s $A/model_world_size_1_rank_0.pt ]; do sleep 60; done
  sleep 60
  until [ -s $X/adapter_model.safetensors ]; do $SPY -I $W/port/extract_lora_adapter.py $A $X || sleep 120; done
  ADARGS=(--adapter $X)
fi
log "$TG: evaluating ${ADARGS[*]:-no adapter} (848 = test 651 + dev 197, greedy, budget 50, snap safe, base $BASE)"

EVAL_RE=${EVAL_RE:-^sw_rl_gpu$}; HOLD_RE=${HOLD_RE:-^(error_attr_hold_.*|tcded_gpu|a7b_gpu)$}; GMEM=${GMEM:-9000}; HMEM=${HMEM:-16000}
freemem() { srun --jobid=$1 --overlap -n1 -c1 --mem=0 -w $2 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null </dev/null | head -n1; }
pick() { local skip=" $(cat $RB/.skip_jobs 2>/dev/null | tr '\n' ' ') " H J N m best="" bm=0
  for pass in eval hold; do
    local re=$EVAL_RE need=$GMEM; [ $pass = hold ] && { re=$HOLD_RE; need=$HMEM; }
    for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk -v re="$re" '$2 ~ re && $3=="RUNNING"{print $1}'); do
      case "$skip" in *" $H "*) continue;; esac
      for N in $(scontrol show hostnames $(squeue -h -j $H -o %N) 2>/dev/null); do m=$(freemem $H $N); [[ "$m" =~ ^[0-9]+$ ]] || continue
        [ $m -ge $need ] && [ $m -gt $bm ] && { bm=$m; best="$H $N"; }; done; done
    [ -n "$best" ] && { echo $best; return; }
  done; }
# one launch at a time across all eval drivers (a new process needs ~40 s to allocate its memory)
launch_lock() { exec 8>/dev/shm/pan00389_sw_rl_eval_launch.lock; flock 8; }   # Oct 10 16:05: LOCAL tmpfs on cn1019 (flock on NFS left a stale lock whose holder had exited)
shard() { local LG=$1; shift; local n=0
  until grep -q "DONE collect" $LG 2>/dev/null; do n=$((n + 1)); local H=""
    ( launch_lock; until H=$(pick); [ -n "$H" ]; do sleep 60; done; set -- $H; echo "### attempt $n $(date '+%F %T') job $1 node $2" >> $LG; echo "$1 $2" > $LG.slot; sleep 45 )
    set -- $(cat $LG.slot) "$@"; local J=$1 NN=$2; shift 2
    srun --jobid=$J --overlap -n1 -c2 --mem=0 -w $NN $P -u scripts/sciworld_agent.py collect --model $BASE "${ADARGS[@]}" "$@" >> $LG 2>&1 || { echo "### attempt $n exited non-zero $(date '+%F %T')" >> $LG; sleep 30; }
    [ $n -ge 25 ] && { log "$TG: GIVING UP on $LG"; return 1; }; done; }
for i in $(seq 0 11); do shard $D/eval_test_s$i.log --split test --per_task 30 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 12 --out $D/eval_test_s$i.jsonl --tag ${TG}_test & sleep 5; done
for i in 0 1 2 3; do shard $D/eval_dev_s$i.log --split dev --per_task 7 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 4 --out $D/eval_dev_s$i.jsonl --tag ${TG}_dev & sleep 5; done
wait
cat $D/eval_test_s{0..11}.jsonl > $D/eval_test.jsonl; cat $D/eval_dev_s{0..3}.jsonl > $D/eval_dev.jsonl
log "$TG: EVAL_DONE $(cat $D/eval_test.jsonl $D/eval_dev.jsonl | grep -c '"won": true') wins / $(cat $D/eval_test.jsonl $D/eval_dev.jsonl | grep -c .) episodes"
$P scripts/sciworld_rl_summary.py $TG | tee -a $RB/queue.log
