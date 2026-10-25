# ScienceWorld StepFix helpers (Oct 9 2026, user-approved).  Only the TextCraft dedicated jobs (tcded_gpu) are used; never a7b_gpu.
#   source scripts/lib_sciworld_stepfix.sh
export HOLD_NAME_RE='^(tcded_gpu|a7b_gpu|error_attr_hold_.*)$'   # Oct 10 12:00 (coordinator): idle A100 hold jobs too; tcded_rft excluded
source scripts/lib_baselines.sh > /dev/null
SF=reports/sciworld/stepfix; STU0=reports/sciworld/student10/final; mkdir -p $SF
slog() { echo "=== $(date '+%F %T') $*" | tee -a $SF/queue.log; }
# Oct 10 04:25: local pick() override -- skip the tcded jobs listed in $SF/.skip_jobs (re-read at EVERY call; e.g. 3185163 reserved for the RFT
# baselines). Same measurement as lib_baselines.sh pick() otherwise.
pick() { local best="" bm=0 skip=" $(cat $SF/.skip_jobs 2>/dev/null | tr '\n' ' ') "; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk -v re="${HOLD_NAME_RE:-^error_attr_hold_}" '$2 ~ re && $3=="RUNNING"{print $1}'); do case "$skip" in *" $H "*) continue;; esac; for N in $(scontrol show hostnames $(squeue -h -j $H -o %N) 2>/dev/null); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 -w $N nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best="$H $N"; }; done; done; [ "$bm" -ge ${1:-12000} ] 2>/dev/null && echo $best; }
# wait_free PATTERN : orphan safety after a driver restart -- do not launch while a process matching PATTERN (same output) still runs
wait_free() { while pgrep -f -- "$1" > /dev/null; do sleep 30; done; }
GMEM=${GMEM:-9000}   # MiB free needed per student process (one process ~6.7 GB; ~5-6 per 46 GB GPU)

# shard_loop LOG CMD... : rerun CMD (on any tcded GPU with GMEM free) until LOG has "DONE collect"; every attempt appends to LOG
# Oct 10 03:20: low-priority chains (LOWPRIO=1, the ablations) launch nothing new while $SF/.hold_low exists (set by the main chain's watcher
# while a main round is in its teacher/train stage), so the main chain's GPU-critical stages are never starved.
gate() { [ -n "$LOWPRIO" ] && while [ -f $SF/.hold_low ] || { [ "$LOWPRIO" = 2 ] && [ -f $SF/.hold_cd ]; }; do sleep 30; done; return 0; }   # LOWPRIO=2 (ablations C/D, Oct 10 04:10): also wait while .hold_cd (RFT baselines first)
SG() { gate; S "$@"; }
shard_loop() { local LG=$1; shift; local n=0
  local OUT=""; local prev=""; for x in "$@"; do [ "$prev" = "--out" ] && OUT=$x; prev=$x; done
  until grep -q "DONE collect" $LG 2>/dev/null; do n=$((n + 1)); gate; [ -n "$OUT" ] && wait_free "--out $OUT "; echo "### attempt $n $(date '+%F %T')" >> $LG
    $(S $GMEM) $P -u "$@" >> $LG 2>&1 || { echo "### attempt $n exited non-zero $(date '+%F %T')" >> $LG; sleep 30; }
    [ $n -ge ${MAX_ATTEMPTS:-200} ] && { slog "GIVING UP on $LG after $n attempts"; return 1; }; done; }   # Oct 10 15:30: cap 25 -> 200 (load-time OOM bursts)

# collect_sw OUTDIR ADAPTER SEED TAG NSHARDS [extra collect args] : train collection, T=1.0, budget 50, --snap safe, round-robin shards
collect_sw() { local D=$1 AD=$2 SEED=$3 TG=$4 NS=$5; shift 5; mkdir -p $D
  for i in $(seq 0 $((NS - 1))); do shard_loop $D/collect_s$i.log scripts/sciworld_agent.py collect --adapter $AD --split train --budget 50 --temperature 1.0 --seed $SEED --snap safe --shard $i $NS --out $D/traj_s$i.jsonl --tag $TG "$@" & sleep ${LAUNCH_GAP:-30}; done; wait; }

# eval_sw OUTDIR ADAPTER TAG [skip_dev] : greedy eval on the 848 = test 651 (per_task 30, seed 0; 12 shards) + dev 197 (per_task 7, seed 0; 4 shards)
eval_sw() { local D=$1 AD=$2 TG=$3 SKIPDEV=${4:-0}; mkdir -p $D
  for i in $(seq 0 11); do shard_loop $D/eval_test_s$i.log scripts/sciworld_agent.py collect --adapter $AD --split test --per_task 30 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 12 --out $D/eval_test_s$i.jsonl --tag ${TG}_test & sleep ${LAUNCH_GAP:-30}; done
  [ "$SKIPDEV" = 1 ] || for i in 0 1 2 3; do shard_loop $D/eval_dev_s$i.log scripts/sciworld_agent.py collect --adapter $AD --split dev --per_task 7 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 4 --out $D/eval_dev_s$i.jsonl --tag ${TG}_dev & sleep ${LAUNCH_GAP:-30}; done
  wait
  cat $D/eval_test_s*.jsonl > $D/eval_test.jsonl; [ "$SKIPDEV" = 1 ] || cat $D/eval_dev_s*.jsonl > $D/eval_dev.jsonl; }
