# ScienceWorld 7B student helpers (Oct 10 2026, user request "先把7B的R1 science world跑了，用之前的配方").
#   source scripts/lib_sciworld_7b.sh
# Same protocol as the 3B StepFix (lib_sciworld_stepfix.sh) with --model Qwen/Qwen2.5-7B-Instruct passed to every student process.
# Every GPU stage is resumable; every attempt writes its own log LOG.aN (done = "DONE ..." in any attempt log).
source scripts/lib_sciworld_stepfix.sh > /dev/null   # P, S/pick (skips reports/sciworld/stepfix/.skip_jobs), HOLD_NAME_RE incl. tcded/a7b/error_attr_hold
export MODEL=Qwen/Qwen2.5-7B-Instruct
export HOLD_NAME_RE='^(tcded_gpu|a7b_gpu|error_attr_hold_.*|sw7b_gpu)$'   # Oct 10 15:15: + sw7b_gpu (job 3162349, reserved for this 7B run by the coordinator)
SF3=reports/sciworld/stepfix; SF7=reports/sciworld/stepfix7b; mkdir -p $SF7
MA7=reports/attributor_7b_no_observations/final   # Oct 10 15:20 user change: 7B Model A (ALFWorld 7B v3b attributor), variant no_observations
STU7DIR=reports/sciworld/student10_7b; STU7=$STU7DIR/final
GMEM7=${GMEM7:-20000}; TEACH7=${TEACH7:-24000}; TRAIN7=${TRAIN7:-32000}; SFT7=${SFT7:-30000}   # MiB free needed: generation / teacher / train / SFT
s7log() { echo "=== $(date '+%F %T') $*" | tee -a $SF7/queue.log; }
# pick() of lib_sciworld_stepfix.sh reads $SF/.skip_jobs -- keep SF pointing at the 3B dir so the shared skip list is honoured.
SF=$SF3

# loop7 LOG DONE_RE CMD... : rerun CMD on a GPU (S $NEED, default GMEM7) until some attempt log LOG.aN contains DONE_RE
loop7() { local LG=$1 RE=$2; shift 2; local n=0
  while ! cat $LG.a* 2>/dev/null | grep -q "$RE"; do n=$((n + 1)); [ $n -gt ${LOOP7_MAX:-25} ] && { s7log "GIVING UP on $LG after ${LOOP7_MAX:-25} attempts"; return 1; }
    local OUT="" prev=""; for x in "$@"; do [ "$prev" = "--out" ] && OUT=$x; prev=$x; done   # orphan guard after a driver restart: never two writers on one output
    [ -n "$OUT" ] && while pgrep -f -- "--out $OUT( |$)" > /dev/null; do sleep 30; done
    cat $LG.a* 2>/dev/null | grep -q "$RE" && break
    local A=$LG.a$n; while [ -e $A ]; do n=$((n + 1)); A=$LG.a$n; done
    echo "### attempt $n $(date '+%F %T')" > $A; local G=$(S7 ${NEED:-$GMEM7}); echo "### $G" >> $A
    $G $P -u "$@" >> $A 2>&1 || { local rc=$?; echo "### attempt $n exit $rc $(date '+%F %T')" >> $A
      grep -q "OutOfMemory\|CUDA out of memory" $A && s7log "OOM in $A (retrying)"; s7log "attempt failed rc=$rc: $A"; sleep 45; }
  done; }

# collect7 OUTDIR ADAPTER SEED TAG NSHARDS [extra] : train collection T=1.0, budget 50, --snap safe, round-robin shards
collect7() { local D=$1 AD=$2 SEED=$3 TG=$4 NS=$5; shift 5; mkdir -p $D
  for i in $(seq 0 $((NS - 1))); do loop7 $D/collect_s$i.log "DONE collect" scripts/sciworld_agent.py collect --model $MODEL --adapter $AD --split train --budget 50 --temperature 1.0 --seed $SEED --snap safe --shard $i $NS --out $D/traj_s$i.jsonl --tag $TG "$@" & cat $D/collect_s$i.log.a* 2>/dev/null | grep -q "DONE collect" || sleep ${LAUNCH_GAP:-40}; done; wait; }   # Oct 12 23:30: no launch gap for finished shards

# eval7 OUTDIR ADAPTER TAG : greedy 848 = test 651 (per_task 30, seed 0; 12 shards) + dev 197 (per_task 7, seed 0; 4 shards), budget 50, --snap safe
eval7() { local D=$1 AD=$2 TG=$3; mkdir -p $D
  for i in $(seq 0 11); do loop7 $D/eval_test_s$i.log "DONE collect" scripts/sciworld_agent.py collect --model $MODEL --adapter $AD --split test --per_task 30 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 12 --out $D/eval_test_s$i.jsonl --tag ${TG}_test & sleep ${LAUNCH_GAP:-40}; done
  for i in 0 1 2 3; do loop7 $D/eval_dev_s$i.log "DONE collect" scripts/sciworld_agent.py collect --model $MODEL --adapter $AD --split dev --per_task 7 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 4 --out $D/eval_dev_s$i.jsonl --tag ${TG}_dev & sleep ${LAUNCH_GAP:-40}; done
  wait; cat $D/eval_test_s*.jsonl > $D/eval_test.jsonl; cat $D/eval_dev_s*.jsonl > $D/eval_dev.jsonl
  s7log "$TG: eval rows test $(grep -c . $D/eval_test.jsonl) dev $(grep -c . $D/eval_dev.jsonl)"; }

# score summary of an 848 eval: wins, mean score (negatives clipped to 0), score<0 count
sum848() { $P - "$@" <<'PY'
import json, sys
rows = {}
for f in sys.argv[1:]:
    for l in open(f):
        try: e = json.loads(l)
        except Exception: continue
        rows.setdefault((e["split"], e["task"], e["variation"]), e)
R = list(rows.values())
print(f"{len(R)} episodes | wins {sum(e['won'] for e in R)} | mean score (neg->0) {sum(max(e['final_score'], 0) for e in R) / max(1, len(R)):.1f} | score<0 {sum(e['final_score'] < 0 for e in R)}"
      f" | mapped actions {sum(s['mapped'] for e in R for s in e['steps'])} / {sum(len(e['steps']) for e in R)} steps")
PY
}

# Oct 10 15:20: the coordinator reserved job sw7b_gpu (3162349, aga40) for this run and put it on .skip_jobs (other agents' pickers skip it).
# S7 NEED : the reserved GPU first (if NEED MiB free), else the shared pool via pick() (which honours .skip_jobs).
pick7res() { local H N m; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2=="sw7b_gpu" && $3=="RUNNING"{print $1}'); do
    for N in $(scontrol show hostnames $(squeue -h -j $H -o %N) 2>/dev/null); do
      m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 -w $N nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits < /dev/null 2>/dev/null | head -n1)
      [ "${m:-0}" -ge ${1:-20000} ] 2>/dev/null && { echo "$H $N"; return; }; done; done; }
# Oct 10 17:50 (coordinator): while now < the epoch in $SF7/.res_only_until, use ONLY the reserved GPU (no new shards on the shared pool).
res_only() { local u=$(cat $SF7/.res_only_until 2>/dev/null); [ -n "$u" ] && [ "$(date +%s)" -lt "$u" ]; }
S7() { local H=""; until H=$(pick7res $1); [ -n "$H" ] || { res_only || H=$(pick $1); }; [ -n "$H" ]; do sleep 60; done; set -- $H; echo "srun --jobid=$1 --overlap -n1 -c4 --mem=0 -w $2"; }
