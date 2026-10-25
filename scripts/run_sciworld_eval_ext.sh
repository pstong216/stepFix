#!/bin/bash
# Oct 10 2026 15:30 (user-requested supplementary EXTENDED eval, evaluation only): ScienceWorld test variations BEYOND the first 30 per task of the
# same seed-0 shuffle (--job_slice 30 9999 -> 1,168 episodes), greedy, budget 50, --snap safe (same settings as the 848 eval).
# NS round-robin shards per model -> reports/sciworld/eval_ext/<model>/eval_test_ext_s<i>.jsonl (append + resumable).
# GPUs: pick()/S() (lib_sciworld_stepfix.sh override = lib_baselines.sh pick + .skip_jobs), HOLD_NAME_RE tcded/a7b/error_attr_hold, 16 GB free
# per launch; low priority (LOWPRIO=1 gate on .hold_low); at most $EXT/.maxp concurrent eval_ext processes (re-read live); launches serialised (mkdir lock).
#   bash scripts/run_sciworld_eval_ext.sh            (run in tmux on cn1019)
source scripts/lib_sciworld_stepfix.sh
export HOLD_NAME_RE='^(tcded_gpu|a7b_gpu|error_attr_hold_.*)$' LOWPRIO=1
EXT=reports/sciworld/eval_ext; NS=${NS:-16}; GFREE=${GFREE:-16000}; mkdir -p $EXT; [ -f $EXT/.maxp ] || echo 24 > $EXT/.maxp
elog() { echo "=== $(date '+%F %T') $*" | tee -a $EXT/driver.log; }
S_=reports/sciworld/stepfix; B_=reports/sciworld/baselines
MODELS="R0:reports/sciworld/student10/final r1b:$S_/r1b/stepfix/adapter r2h:$S_/r2h/model/adapter r3h:$S_/r3h/model/adapter
 rftm_r1:$B_/rftm/r1/model/adapter rftm_r2:$B_/rftm/r2/model/adapter rftm_r3:$B_/rftm/r3/model/adapter
 rftfull_r1:$B_/rftfull/r1/model/adapter rftfull_r2:$B_/rftfull/r2/model/adapter rftfull_r3:$B_/rftfull/r3/model/adapter
 r1b_random:$S_/r1b/random/adapter r2:$S_/r2/stepfix/adapter r3:$S_/r3/stepfix/adapter r2f:$S_/r2f/model/adapter r3f:$S_/r3f/model/adapter"
nrun() { pgrep -af "sciworld_agent.py collect --adapter reports/sciworld/.*--out $EXT/" | grep -o -- "--out [^ ]*" | sort -u | wc -l; }
# run_shard M ADAPTER i : until the shard log has "DONE collect": (under the launch lock) wait for gate + a free slot + a GPU, launch, wait.
run_shard() { local M=$1 AD=$2 i=$3 MI=$4; local D=$EXT/$M; local key=$(printf "%02d_%02d" $MI $i); local LG=$D/eval_test_ext_s$i.log OUT=$D/eval_test_ext_s$i.jsonl n=0 pid C
  until grep -q "DONE collect" $LG 2>/dev/null; do n=$((n + 1)); [ $n -gt 60 ] && { elog "GIVING UP $LG after 60 attempts"; return 1; }
    wait_free "--out $OUT "
    touch $WD/$key   # Oct 10 18:00: priority queue -- only the waiting shard with the smallest (model rank, shard) key may take the lock
    until [ "$(ls $WD | head -n 1)" = "$key" ] && mkdir $LK 2>/dev/null; do sleep 5; done   # mkdir lock (flock on this NFS leaked a lock held by a dead process, Oct 10 15:45)
    gate; while [ "$(nrun)" -ge "$(cat $EXT/.maxp 2>/dev/null || echo 24)" ]; do sleep 30; done
    C=$(S $GFREE); echo "### attempt $n $(date '+%F %T') $C" >> $LG
    $C $P -u scripts/sciworld_agent.py collect --adapter $AD --split test --per_task 9999 --job_slice 30 9999 --budget 50 --temperature 0 --seed 0 \
      --snap safe --shard $i $NS --out $OUT --tag ext_${M}_test >> $LG 2>&1 & pid=$!; rm -f $WD/$key
    sleep 45; rmdir $LK
    wait $pid || { echo "### attempt $n exited non-zero $(date '+%F %T')" >> $LG; sleep 30; }; done; }
run_model() { local M=$1 AD=$2 MI=$3; mkdir -p $EXT/$M
  for i in $(seq 0 $((NS - 1))); do run_shard $M $AD $i $MI & sleep 2; done; wait
  local n=$(cat $EXT/$M/eval_test_ext_s*.jsonl | grep -c .); elog "MODEL DONE $M: $n episodes"; echo "$M" >> $EXT/.done_models; }
LK=$EXT/.launch.lockd; rmdir $LK 2>/dev/null; WD=$EXT/.wait; mkdir -p $WD; rm -f $WD/*; rm -f $EXT/.launch.lock   # a restarted driver clears a stale lock
elog "START NS=$NS GFREE=$GFREE maxp=$(cat $EXT/.maxp)"
MI=0; for x in $MODELS; do M=${x%%:*}; AD=${x#*:}; MI=$((MI + 1))
  [ -f $AD/adapter_config.json ] || { elog "SKIP $M: adapter missing ($AD)"; continue; }
  grep -qx "$M" $EXT/.done_models 2>/dev/null && continue
  elog "queue $M ($AD)"; run_model $M $AD $MI &
  # priority order: the next model's shards queue only after every shard of this model has had its first launch
  sleep 40; until [ "$(grep -l '^### attempt' $EXT/$M/eval_test_ext_s*.log 2>/dev/null | wc -l)" -ge $NS ] && ! ls $WD | grep -q "^$(printf %02d $MI)_"; do sleep 20; done
done; wait; elog "ALL DONE"
