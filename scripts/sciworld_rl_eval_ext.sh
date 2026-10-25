#!/usr/bin/env bash
# Supplementary EXTENDED ScienceWorld eval of an online-RL baseline checkpoint (Oct 10 2026, coordinator request): the same settings as
# scripts/run_sciworld_eval_ext.sh (test variations beyond the first 30 per task of the seed-0 shuffle: --per_task 9999 --job_slice 30 9999,
# greedy, budget 50, --snap safe; 16 round-robin shards) with --model = merged student10 and --adapter = the RL LoRA extracted by
# scripts/sciworld_rl_eval.sh. Starts only after that checkpoint's 848 eval is done (the 848 comes first).
#   usage (tmux on cn1019): bash scripts/sciworld_rl_eval_ext.sh RUN_DIR STEP TAG   -> reports/sciworld/eval_ext/<TAG>/eval_test_ext_s<i>.jsonl
set -u
cd /users/8/pan00389/multi-agent/error-attribution
R=$1; N=$2; TG=$3
P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
W=/scratch.global/pan00389/error-attribution/serl_work; BASE=$W/models/qwen25_3b_sciworld_student10_merged
RB=reports/sciworld/rl_baselines; EXT=reports/sciworld/eval_ext; D=$EXT/$TG; mkdir -p $D; NS=16
X=$R/lora_extracted/global_step_$N/actor/lora_adapter
export ERROR_ATTR_ROOT=$PWD HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "=== $(date '+%F %T') $*" | tee -a $RB/queue.log; }
until grep -q "$TG: EVAL_DONE" $RB/queue.log 2>/dev/null && [ -s $X/adapter_model.safetensors ]; do sleep 60; done
log "$TG: extended eval (test beyond 30/task, job_slice 30 9999) -> $D"
HOLD_RE=${HOLD_RE:-^(error_attr_hold_.*|tcded_gpu|a7b_gpu)$}; HMEM=${HMEM:-16000}
freemem() { srun --jobid=$1 --overlap -n1 -c1 --mem=0 -w $2 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null </dev/null | head -n1; }
pick() { local skip=" $(cat $RB/.skip_jobs 2>/dev/null | tr '\n' ' ') " H N m best="" bm=0
  for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk -v re="$HOLD_RE" '$2 ~ re && $3=="RUNNING"{print $1}'); do
    case "$skip" in *" $H "*) continue;; esac
    for N in $(scontrol show hostnames $(squeue -h -j $H -o %N) 2>/dev/null); do m=$(freemem $H $N); [[ "$m" =~ ^[0-9]+$ ]] || continue
      [ $m -ge $HMEM ] && [ $m -gt $bm ] && { bm=$m; best="$H $N"; }; done; done
  [ -n "$best" ] && echo $best; }
shard() { local i=$1 LG=$D/eval_test_ext_s$1.log n=0
  until grep -q "DONE collect" $LG 2>/dev/null; do n=$((n + 1))
    ( exec 8>/dev/shm/pan00389_sw_rl_ext_launch.lock; flock 8;   # own local lock: ext waits for >=20 GB while holding it and must never block the 848 evals (Oct 10 19:00)
      until H=$(pick); [ -n "$H" ]; do sleep 60; done; echo "### attempt $n $(date '+%F %T') $H" >> $LG; echo "$H" > $LG.slot; sleep 45 )
    set -- $(cat $LG.slot)
    srun --jobid=$1 --overlap -n1 -c2 --mem=0 -w $2 $P -u scripts/sciworld_agent.py collect --model $BASE --adapter $X --split test --per_task 9999 --job_slice 30 9999 \
      --budget 50 --temperature 0 --seed 0 --snap safe --shard $i $NS --out $D/eval_test_ext_s$i.jsonl --tag ext_${TG}_test >> $LG 2>&1 || { echo "### attempt $n exited non-zero $(date '+%F %T')" >> $LG; sleep 30; }
    [ $n -ge 25 ] && { log "$TG ext: GIVING UP on $LG"; return 1; }; done; }
for i in $(seq 0 $((NS - 1))); do shard $i & sleep 5; done; wait
log "$TG: EXT_DONE $(cat $D/eval_test_ext_s*.jsonl | grep -c '"won": true') wins / $(cat $D/eval_test_ext_s*.jsonl | grep -c .) episodes"
