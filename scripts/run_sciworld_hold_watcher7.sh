#!/bin/bash
# Oct 10 04:15 priority watcher v7 (main = r2g/r3g, r2a eval, r3s3; C/D held while r2g runs; Oct 10 15:35) (main = 3-step chain r2s3/r3s3, Oct 10 14:15) (coordinator 04:05): 1. main r2f/r3f + long-budget  2. ablations A/B and the RFT baselines (other agent)  3. C/D.
# .hold_low (blocks NEW launches of every ablation chain: A/B LOWPRIO=1, C/D LOWPRIO=2) is ON while
#   (a) a main round is between pointer and trained adapter, or (b) a main process waits for a GPU (log ends with "### attempt" > 60 s), or
#   (c) A+B already run >= AB_CAP GPU processes (cap ~60% of the ablation slots, so the RFT baselines get windows; file $SF/.ab_cap, default 8).
# .hold_cd (blocks C/D only) is ON until rftm and rftfull have each finished their R1 eval (848 rows) and trained R2.
# Main processes waiting > 150 s preempt that many ablation processes, C/D first, then A/B (newest first): only the episode in flight is lost.
SF=reports/sciworld/stepfix; BL=reports/sciworld/baselines; MAIN="r2g r3g r2a r3s3"
cnt() { ps -eo pid,ppid,args | awk -v pat="$1" '$3=="srun" && $4 ~ /^--jobid/ {p[$1]=$2; a[$1]=$0} END {n=0; for (k in p) if (!(p[k] in p) && a[k] ~ pat) n++; print n}'; }   # one srun step = 2 srun processes; count the parents only
while true; do h=0; why=""; waiting=0; long=0; now=$(date +%s)
  for d in r2g r3g r2a r3s3; do [ -s $SF/$d/pointer.jsonl ] && [ ! -f $SF/$d/model/adapter/adapter_config.json ] && { h=1; why="$why $d-teacher/train"; }; done
  for d in $MAIN; do [ -d $SF/$d ] || continue
    for lg in $(find $SF/$d -name "*.log" -mmin -600 2>/dev/null); do
      tail -n 1 $lg 2>/dev/null | grep -q "^### attempt" || continue
      age=$((now - $(stat -c %Y $lg))); [ $age -gt 60 ] && { waiting=$((waiting + 1)); h=1; }; [ $age -gt 150 ] && long=$((long + 1)); done; done
  [ $waiting -gt 0 ] && why="$why main-waiting:$waiting"
  cap=$(cat $SF/.ab_cap 2>/dev/null || echo 8); ab=$(cnt "stepfix/abl_(rand|norep)"); [ $ab -ge $cap ] && { h=1; why="$why A+B=$ab>=cap$cap"; }
  if [ $h = 1 ]; then [ -f $SF/.hold_low ] || echo "=== $(date '+%F %T') hold_low ON ($why)"; touch $SF/.hold_low
  else [ -f $SF/.hold_low ] && { rm -f $SF/.hold_low; echo "=== $(date '+%F %T') hold_low OFF"; }; fi
  rft_ok=1; for c in rftm rftfull; do n=$(cat $BL/$c/r1/eval/eval_*_s*.jsonl 2>/dev/null | grep -c .); { [ "$n" -ge 848 ] && [ -f $BL/$c/r2/model/adapter/adapter_config.json ]; } || rft_ok=0; done
  for g in r2g r3g; do [ -s $SF/$g/pointer.jsonl ] && ! grep -q "$g DONE" $SF/queue.log && rft_ok=0; done   # v7: C/D paused while r2g / r3g run
  if [ $rft_ok = 0 ]; then [ -f $SF/.hold_cd ] || echo "=== $(date '+%F %T') hold_cd ON (RFT R1 eval / R2 train pending)"; touch $SF/.hold_cd
  else [ -f $SF/.hold_cd ] && { rm -f $SF/.hold_cd; echo "=== $(date '+%F %T') hold_cd OFF (rftm + rftfull R1 eval done and R2 trained)"; }; fi
  if [ $long -gt 0 ]; then
    pids=$( { ps -eo pid,args --sort=-start_time | grep "^ *[0-9]* srun --jobid" | grep "stepfix/abl_unf\|stepfix/abl_roll"; ps -eo pid,args --sort=-start_time | grep "^ *[0-9]* srun --jobid" | grep "stepfix/abl_rand\|stepfix/abl_norep"; } | awk '{print $1}' | head -n $long)
    [ -n "$pids" ] && { echo "=== $(date '+%F %T') preempting $(echo $pids | wc -w) ablation process(es) for $long waiting main process(es): $pids"; kill $pids; sleep 90; }
  fi
  sleep 30; done
