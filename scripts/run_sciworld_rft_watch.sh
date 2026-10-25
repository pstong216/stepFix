#!/bin/bash
# Watchdog for the ScienceWorld RFT baselines (Oct 10 2026), every 5 min:
#  STALL    : both drivers alive and no baseline output grew for 20 min
#  IDLE-GPU : some of our shards wait for a GPU while a tcded GPU has >= gmem + 8 GB free, seen on 2 consecutive checks (10 min)
# Each writes one line to queue.log per episode.
source scripts/lib_sciworld_rft.sh; last=-1; still=0; warned=0; idle=0; iwarned=0
while pgrep -f "run_sciworld_rft.sh rft" > /dev/null; do
  sz=$(find $BL/rftm $BL/rftfull -name "*.jsonl" -printf "%s\n" 2>/dev/null | awk '{s += $1} END {print s + 0}')
  if [ "$sz" = "$last" ]; then still=$((still + 5)); else still=0; warned=0; fi; last=$sz
  wait_n=$(for f in $(find $BL/rftm $BL/rftfull -name "*.log" 2>/dev/null); do tail -n 1 $f | grep -q "^### attempt [0-9]* 20" && echo $f; done | wc -l)
  free=""; maxfree=0
  for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk -v re="$HOLD_NAME_RE" '$2 ~ re && $3=="RUNNING"{print $1}'); do N=$(squeue -h -j $H -o %N)
    m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 -w $N nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1); free="$free $H/$N:${m}MiB"
    [[ "$m" =~ ^[0-9]+$ ]] && [ $m -gt $maxfree ] && maxfree=$m; done
  if [ $wait_n -gt 0 ] && [ "$(myprocs)" -lt "$(maxp)" ] && [ $maxfree -ge $(( $(gmem) + 8000 )) ]; then idle=$((idle + 1)); else idle=0; iwarned=0; fi
  [ $idle -ge 2 ] && [ $iwarned = 0 ] && { iwarned=1; blog "IDLE-GPU: $wait_n of our shards waiting while a GPU has ${maxfree} MiB free (gmem $(gmem), our procs $(myprocs), cap $(maxp)); tcded free:$free"; }
  [ $still -ge 20 ] && [ $warned = 0 ] && { warned=1; blog "STALL: no baseline output grew for $still min; our GPU procs $(myprocs), shards waiting for a GPU $wait_n; tcded free:$free"; }
  sleep 300; done
blog "watchdog: drivers gone, exiting"
