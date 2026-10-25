#!/bin/bash
# Orchestrates the self-attribution chains once the round-0 attributors are evaluated:
#   qw hold   : selfattr R1 -> R2 -> R3 (main chain; attributor and student refreshed every round)
#   igpu hold : selfattr_plain R1, then m5_informed R1 (single-round ablations, same protocol)
# Attributor pick: among {synth, envlab, both} with synth held-out +-1 step accuracy >= 0.5, the one whose real-failure pointer
# is least collapsed onto step 0 (lowest t0); fallback 'both'.
cd /users/8/pan00389/multi-agent/error-attribution; B=reports/alfworld/self_attr_rounds; mkdir -p $B; M=$B/queue.log
A=reports/alfworld/self_attr/r0; STU=reports/alfworld/student3b_low100/final; M5=reports/alfworld/self_evolve3b_rounds/env/r3/m5/adapter
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
until grep -q "ATTR R0 DONE" reports/alfworld/self_evolve3b/queue.log 2>/dev/null; do sleep 120; done
PICK=$(python3 - <<'PY'
import re
best=None
for l in open("reports/alfworld/self_evolve3b/queue.log"):
    m=re.search(r"ATTR R0 (\w+): .*\(\+-1 ([0-9.]+)\).*t0 ([0-9.]+)", l)
    if m and float(m.group(2))>=0.5 and (best is None or float(m.group(3))<best[1]): best=(m.group(1), float(m.group(3)))
print(best[0] if best else "both")
PY
)
[ -f $A/$PICK/adapter/adapter_config.json ] || PICK=both
log "self-attr chains: attributor R0 = $PICK"
( S=$STU; AT=$A/$PICK/adapter
  for R in 1 2 3; do bash scripts/self_attr_round.sh selfattr $R $S $AT error_attr_hold_igpu_qw || break
    S=$(cat $B/selfattr/r$R/ADAPTER); AT=$(cat $B/selfattr/r$R/ATTR); done ) &
( bash scripts/self_attr_round.sh selfattr_plain 1 $STU $A/$PICK/adapter error_attr_hold_igpu
  bash scripts/self_attr_round.sh m5_informed 1 $STU $M5 error_attr_hold_igpu ) &
wait; log "SELF-ATTR CHAINS DONE"
