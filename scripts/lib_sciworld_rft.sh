# ScienceWorld RFT baselines (Oct 10 2026): helpers shared by scripts/run_sciworld_rft.sh.  Same collection (T=1.0, budget 50, --snap safe,
# per_task 20 seed 1 = the 518 train variations), trainer (scripts/sciworld_stepfix.py train, read-only) and 848 eval (test per_task 30 seed 0 +
# dev per_task 7 seed 0, greedy) as StepFix.  GPUs: the tcded_gpu jobs (spare windows only; StepFix has priority) and, from Oct 10 04:15,
# the reserved job tcded_rft (coordinator decision; StepFix no longer launches there).  Never a7b_gpu.
#   source scripts/lib_sciworld_rft.sh
export HOLD_NAME_RE='^(tcded_gpu|tcded_rft|error_attr_hold_.*)$'; RFT_NAME=tcded_rft
RFT_TMEM=${RFT_TMEM:-14000}; RFT_SMEM=${RFT_SMEM:-21000}   # on tcded_rft: training needs 14 GB; a shard needs 9 GB + 12 GB kept free for a training
source scripts/lib_baselines.sh > /dev/null   # pick(), P, cd to the repo root
SF=reports/sciworld/stepfix; STU0=reports/sciworld/student10/final; BL=reports/sciworld/baselines; mkdir -p $BL
GMEM=${GMEM:-9000}; TMEM=${TMEM:-20000}; LOCK=$BL/.gpu.lock
blog() { echo "=== $(date '+%F %T') $*" | tee -a $BL/queue.log; }
# our GPU processes = distinct --out paths of srun clients on this node that write under reports/sciworld/baselines/
myprocs() { ps -eo args | grep '^srun' | grep 'reports/sciworld/baselines/' | grep -o -- '--out [^ ]*' | sort -u | wc -l; }
maxp() { local m; m=$(cat $BL/maxp 2>/dev/null); echo ${m:-20}; }   # cap on our concurrent GPU processes (edit $BL/maxp to change it live)
tmem() { local m; m=$(cat $BL/tmem 2>/dev/null); echo ${m:-$TMEM}; }   # MiB free needed to launch training (edit $BL/tmem to change it live)
gmem() { local m; m=$(cat $BL/gmem 2>/dev/null); echo ${m:-$GMEM}; }   # MiB free needed per student process (edit $BL/gmem to change it live)

# gpufree : "JOB NODE FREE_MiB NAME" per running tcded GPU, cached for 20 s in $BL/.gpufree (call under the lock)
gpufree() { local H J N m
  if [ -z "$(find $BL/.gpufree -newermt '-20 seconds' 2>/dev/null)" ]; then
    squeue -u pan00389 -h -o "%i %j %T" | awk -v re="$HOLD_NAME_RE" '$2 ~ re && $3=="RUNNING"{print $1, $2}' | while read H J; do for N in $(scontrol show hostnames $(squeue -h -j $H -o %N) 2>/dev/null); do
      m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 -w $N nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null </dev/null | head -n1); [[ "$m" =~ ^[0-9]+$ ]] && echo "$H $N $m $J"; done; done > $BL/.gpufree.tmp
    mv $BL/.gpufree.tmp $BL/.gpufree; fi
  cat $BL/.gpufree; }
# effree : "JOB NODE EFF_FREE NAME" -- free memory minus what our own launches of the last 120 s requested on that GPU (not yet allocated)
effree() { local now=$(date +%s) H N m J pend
  while read H N m J; do pend=$(awk -v j=$H -v t=$((now - 120)) '$1 == j && $2 > t {s += $3} END {print s + 0}' $BL/.launches 2>/dev/null); echo "$H $N $((m - ${pend:-0})) $J"; done < <(gpufree); }
# pickr MEM : choose a GPU and record the launch (a training reserves 30 GB of a shared GPU, 14 GB of tcded_rft, for 120 s).
#   training (PRIO=1): tcded_rft if it has RFT_TMEM free, else the shared GPU with the most room if >= MEM (tmem)
#   shard           : the shared GPU with the most room if >= MEM (gmem, spare windows), else tcded_rft if it has RFT_SMEM free
pickr() { local now=$(date +%s) H N m J best="" bm=-999999 rft="" rm=-999999 rec
  while read H N m J; do if [ "$J" = "$RFT_NAME" ]; then [ $m -gt $rm ] && { rm=$m; rft="$H $N"; }; else [ $m -gt $bm ] && { bm=$m; best="$H $N"; }; fi; done < <(effree)
  local pick=""
  if [ "${PRIO:-0}" = 1 ]; then
    if [ -n "$rft" ] && [ $rm -ge $RFT_TMEM ]; then pick=$rft; rec=$RFT_TMEM; elif [ -n "$best" ] && [ $bm -ge $1 ]; then pick=$best; rec=30000; fi
  else
    if [ -n "$best" ] && [ $bm -ge $1 ]; then pick=$best; rec=$1; elif [ -n "$rft" ] && [ $rm -ge $RFT_SMEM ]; then pick=$rft; rec=$1; fi
  fi
  [ -n "$pick" ] && { echo "${pick% *} $now $rec ${pick#* }" >> $BL/.launches; echo $pick; }; }
# train_fits : true if a training could launch now (used to hold shards off while a training waits)
train_fits() { local t=$(tmem); effree | awk -v r=$RFT_NAME -v rt=$RFT_TMEM -v t=$t '($4 == r && $3 >= rt) || ($4 != r && $3 >= t) {f = 1} END {exit !f}'; }
# prio_waiting : true if a live training launch is waiting (its $BL/.prio.PID file exists and PID is alive)
prio_waiting() { local f; for f in $BL/.prio.*; do [ -e "$f" ] || continue; kill -0 ${f##*.} 2>/dev/null && return 0; rm -f "$f"; done; return 1; }

# gpu_run MEM LOG ARGS... : wait for our cap and a tcded GPU with MEM MiB (effective) free, then launch "$P -u ARGS" there (output appended
# to LOG); the lock is held only for one pick + launch.  PRIO=1 (training) polls every 15 s and, while it waits,
# non-PRIO launches hold off whenever some GPU has room for the training (they still use windows too small for it).
gpu_run() { local MEM=$1 LG=$2; shift 2; local H="" pid me=$BASHPID
  [ "${PRIO:-0}" = 1 ] && touch $BL/.prio.$me
  while :; do
    { flock 9
      H=""; if [ "$(myprocs)" -lt "$(maxp)" ] && { [ "${PRIO:-0}" = 1 ] || ! prio_waiting || ! train_fits; }; then H=$(pickr $MEM); fi
      if [ -n "$H" ]; then echo "### on job ${H% *} node ${H#* } $(date '+%F %T')" >> $LG
        srun --jobid=${H% *} --overlap -n1 -c4 --mem=0 -w ${H#* } $P -u "$@" >> $LG 2>&1 9>&- &
        pid=$!; sleep 3; fi
    } 9>>$LOCK
    [ -n "$H" ] && break; sleep $([ "${PRIO:-0}" = 1 ] && echo 15 || echo 45); done
  rm -f $BL/.prio.$me; wait $pid; }

# shard_loop LOG ARGS... : rerun until LOG has "DONE collect" (episodes append and are skipped on restart); every attempt logged in LOG
shard_loop() { local LG=$1; shift; local n=0 nf=0 rc
  until grep -q "DONE collect" $LG 2>/dev/null; do n=$((n + 1)); echo "### attempt $n $(date '+%F %T')" >> $LG
    gpu_run $(gmem) $LG "$@"; rc=$?
    if ! grep -q "DONE collect" $LG; then echo "### attempt $n exited rc=$rc $(date '+%F %T')" >> $LG
      if awk '/^### attempt [0-9]+ 20/{o=0} /OutOfMemoryError|CUDA error: out of memory/{o=1} END{exit !o}' $LG; then blog "OOM-RETRY $LG attempt $n (GPU filled up by concurrent launches; retrying)"
      else nf=$((nf + 1)); blog "FAIL shard $LG attempt $n rc=$rc: $(grep -E 'Error|error|Killed|CANCELLED|PREEMPT|srun:' $LG | tail -1 | cut -c1-200)"; fi; sleep 30; fi
    [ $nf -ge 25 ] || [ $n -ge 100 ] && { blog "GIVING UP on $LG after $n attempts ($nf non-OOM failures)"; return 1; }; done; return 0; }

# collect_rft OUTDIR ADAPTER TAG NSHARDS : the 518 train variations (per_task 20, seed 1), T=1.0, budget 50, --snap safe, round-robin shards
collect_rft() { local D=$1 AD=$2 TG=$3 NS=$4; mkdir -p $D
  for i in $(seq 0 $((NS - 1))); do shard_loop $D/collect_s$i.log scripts/sciworld_agent.py collect --adapter $AD --split train --per_task 20 --budget 50 --temperature 1.0 --seed 1 --snap safe --shard $i $NS --out $D/traj_s$i.jsonl --tag $TG & sleep 2; done; wait; }

# eval_rft OUTDIR ADAPTER TAG : greedy 848 = test 651 (per_task 30, seed 0; 16 shards) + dev 197 (per_task 7, seed 0; 6 shards)
eval_rft() { local D=$1 AD=$2 TG=$3; mkdir -p $D
  for i in $(seq 0 15); do shard_loop $D/eval_test_s$i.log scripts/sciworld_agent.py collect --adapter $AD --split test --per_task 30 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 16 --out $D/eval_test_s$i.jsonl --tag ${TG}_test & sleep 2; done
  for i in $(seq 0 5); do shard_loop $D/eval_dev_s$i.log scripts/sciworld_agent.py collect --adapter $AD --split dev --per_task 7 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 6 --out $D/eval_dev_s$i.jsonl --tag ${TG}_dev & sleep 2; done
  wait; cat $D/eval_test_s*.jsonl > $D/eval_test.jsonl; cat $D/eval_dev_s*.jsonl > $D/eval_dev.jsonl
  local nt=$(grep -c . $D/eval_test.jsonl) nd=$(grep -c . $D/eval_dev.jsonl); [ "$nt" = 651 ] && [ "$nd" = 197 ] || blog "WARNING $D: eval has test $nt / dev $nd episodes (expected 651 / 197)"; }

# train_rft DIR START_ADAPTER SEED : StepFix trainer (1 epoch, lr 1e-5, batch 8; resumes from its checkpoint after preemption)
train_rft() { local D=$1 AD=$2 SD=$3 n=0 rc
  until [ -f $D/model/adapter/adapter_config.json ]; do n=$((n + 1)); echo "### attempt $n $(date '+%F %T')" >> $D/train.log
    PRIO=1 gpu_run $(tmem) $D/train.log scripts/sciworld_stepfix.py train --data $D/train_rows.jsonl --adapter $AD --out $D/model --seed $SD; rc=$?
    for k in $(seq 30); do [ -f $D/model/adapter/adapter_config.json ] && break; sleep 2; done
    [ -f $D/model/adapter/adapter_config.json ] || { blog "FAIL train $D attempt $n rc=$rc: $(grep -E 'Error|error|Killed|srun:' $D/train.log | tail -1 | cut -c1-200)"; sleep 60; }
    [ $n -ge 10 ] && { blog "GIVING UP train $D"; return 1; }; done; return 0; }

# cmp848 NAME DIR : vs R0 and vs our same-round StepFix model
cmp848() { local NM=$1 D=$2 R=$3 O=""
  $P scripts/sciworld_stepfix_compare.py --a R0=$SF/r0/eval_test.jsonl,reports/sciworld/refix/eval_dev_safe.jsonl --b $NM=$D/eval_test.jsonl,$D/eval_dev.jsonl | tee -a $BL/queue.log
  if [ $R = 1 ]; then O=$SF/r1b/eval_stepfix; else for c in $SF/r${R}f/eval_stepfix $SF/r$R/eval_stepfix; do [ "$(cat $c/eval_test.jsonl $c/eval_dev.jsonl 2>/dev/null | grep -c .)" = 848 ] && { O=$c; break; }; done; fi
  [ -n "$O" ] && { echo "(ours = $O)" | tee -a $BL/queue.log; $P scripts/sciworld_stepfix_compare.py --a ours_r$R=$O/eval_test.jsonl,$O/eval_dev.jsonl --b $NM=$D/eval_test.jsonl,$D/eval_dev.jsonl | tee -a $BL/queue.log; } || blog "$NM: our R$R eval not complete yet (compare later)"; }
