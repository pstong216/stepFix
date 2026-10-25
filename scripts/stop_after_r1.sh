#!/bin/bash
# Oct 8 20:50 (user: "rft先做1轮看看。20%回放+100条sft也是做一轮"): stop a chain right after its R1 774 line is logged.
#   bash scripts/stop_after_r1.sh <R1 tag> <driver script name>
TAG=$1; DRV=$2; Q=reports/alfworld/self_attr_rounds/queue.log
desc() { local p; for p in $(ps -o pid= --ppid $1); do echo $p; desc $p; done; }
until grep -q "BASELINE $TAG:" $Q; do sleep 15; done
for R in $(pgrep -f "^bash scripts/$DRV"); do ALL="$R $(desc $R)"
  B=$(for p in $ALL; do ps -o pid=,comm= -p $p; done | awk '$2=="bash"{print $1}'); S=$(for p in $ALL; do ps -o pid=,comm= -p $p; done | awk '$2=="srun"{print $1}')
  kill $B 2>/dev/null; sleep 1; kill $S 2>/dev/null; done
echo "=== $(date '+%F %T') $DRV stopped after R1 ($TAG) by user decision (one round only)" | tee -a $Q
