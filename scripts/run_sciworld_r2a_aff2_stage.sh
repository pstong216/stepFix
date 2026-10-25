#!/bin/bash
# Oct 10 15:00: variant B v2 (targeted unlock) -- window stage + selection for r2a ONLY (no training until the firing rate is checked)
source scripts/lib_sciworld_stepfix.sh; D=$SF/r2a; START=$SF/r1b/stepfix/adapter
for i in $(seq 0 11); do ( until grep -q "DONE window" $D/window_aff2_s$i.log 2>/dev/null; do echo "### attempt $(date '+%F %T')" >> $D/window_aff2_s$i.log
    wait_free "window --dir $D .*--shard $i 12"; $(S 12000) $P -u scripts/sciworld_stepfix.py window --dir $D --adapter $START --shard $i 12 --k 3 --aff2 >> $D/window_aff2_s$i.log 2>&1 || sleep 30; done ) & sleep 15; done; wait
$P scripts/sciworld_stepfix.py select --dir $D --round 2 --rho 0.2 --evidence window_aff2 > $D/select.log; slog "r2a: aff2 window stage + selection done (no training yet)"
