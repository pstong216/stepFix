#!/bin/bash
# Same-trajectory control #2 (Oct 5 22:25): the student with hindsight names the first wrong step from a compact view; same R2
# trajectories and R1 start as the final chain; everything downstream identical.  Dir selfattr_hindtp/r2.
cd /users/8/pan00389/multi-agent/error-attribution; SA=reports/alfworld/self_attr_rounds; S=$SA/selfattr_hind/r2; N=$SA/selfattr_hindtp/r2; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
pick() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge 12000 ] 2>/dev/null && echo $best; }
until tail -n 3 $S/collect.log 2>/dev/null | grep -q '^DONE' && [ -s $S/replay.jsonl ]; do sleep 120; done
mkdir -p $N/self $N/proxy $N/envlab $N/attr_labels; for f in traj.jsonl traj.release.jsonl collect.log replay.jsonl; do [ -e $N/$f ] || cp $S/$f $N/; done
$P - <<PY
import json
eps=[json.loads(l) for l in open("$N/traj.jsonl")][:1000]
open("$N/ids_fail.jsonl","w").writelines(json.dumps({"id":e["id"]})+"\n" for e in eps if not e["won"])
PY
until grep -q "DONE teacher-pointed" $N/point.log 2>/dev/null; do H=""; until H=$(pick); [ -n "$H" ]; do sleep 60; done
  srun --jobid=$H --overlap -n1 -c4 --mem=0 $P -u scripts/self_point_step.py --traj $N/traj.jsonl --ids $N/ids_fail.jsonl --adapter $(cat $SA/selfattr_hind/r1/ADAPTER) --out $N/pointer.jsonl >> $N/point.log 2>&1 || sleep 60; done
echo "=== $(date '+%F %T') R2 same-trajectory control: TEACHER-POINTED step (selfattr_hindtp) | $(grep 'DONE teacher-pointed' $N/point.log | tail -n1)" | tee -a $SA/queue.log
EVIDENCE=generic STEP=logfix TEACHER_NO_DIAG=1 K=100 CHAIN_SUFFIX=_hindtp SEED=0 bash scripts/self_attr_round.sh selfattr 2 $(cat $SA/selfattr_hind/r1/ADAPTER) reports/alfworld/self_attr/r0/both/adapter any
echo "=== $(date '+%F %T') R2 same-trajectory control DONE (selfattr_hindtp)" | tee -a $SA/queue.log
