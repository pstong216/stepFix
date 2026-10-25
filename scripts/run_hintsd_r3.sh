#!/bin/bash
# User-approved (Oct 7 20:05): HINT-SD baseline re-implemented in our framework, 3B, R1 -> R2 (scripts/hintsd.py).
# Per round: failures among the first 1000 games (same as ours) -> the student analyzes each failed trajectory itself (<= 3 steps + feedback,
# 4 GPU shards) -> the student conditioned on the feedback at the target turn = teacher (3 shards) -> reverse-KL rows at those steps ->
# train (same trainer / lr / batch / epoch as ours; NO replay, NO environment check, NO budget) -> 774 eval.  R1 uses the SFT student's
# 1323-game collection (as ours); R2 collects 1500 train games with the R1 model.
source scripts/lib_baselines.sh; CH=a3b_hintsd
hround() { local R=$1 START=$2 T=$3 D=$SA/$CH/r$1; mkdir -p $D
  if [ "$R" != "1" ]; then collect4 $D $START ${CH}_r$R; T=$D/traj.jsonl; fi
  $P -c "import json; eps=[json.loads(l) for l in open('$T')][:1000]; open('$D/ids_fail.jsonl','w').writelines(json.dumps({'id':e['id']})+'\n' for e in eps if not e['won'])"
  log "$CH r$R: start on $(hostname), $(grep -c . $D/ids_fail.jsonl) failures (from $START)"
  if ! grep -q "DONE" $D/analyze.log 2>/dev/null; then
    for i in 0 1 2 3; do ( until grep -q "DONE analyze" $D/analyze_s$i.log 2>/dev/null; do $(S 12000) $P -u scripts/hintsd.py analyze --traj $T --ids $D/ids_fail.jsonl --adapter $START --out $D/analysis_s$i.jsonl --shard $i 4 --seed $i >> $D/analyze_s$i.log 2>&1 || sleep 30; done ) & sleep 60; done; wait
    cat $D/analysis_s*.jsonl > $D/analysis.jsonl; echo DONE >> $D/analyze.log
  fi
  $P -c "
import json, collections; R=[json.loads(l) for l in open('$D/analysis.jsonl')]; ts=[x['t'] for r in R for x in r['targets']]
print('=== $CH r$R: analyzer parsed %d/%d failures, %d target steps (%.2f per failure), t0 share %.0f%%, mean t %.1f' % (sum(r['parsed'] for r in R), len(R), len(ts), len(ts)/max(1,sum(r['parsed'] for r in R)), 100*sum(t==0 for t in ts)/max(1,len(ts)), sum(ts)/max(1,len(ts))))" | tee -a $SA/queue.log
  if ! grep -q "DONE" $D/teacher.log 2>/dev/null; then
    for i in 0 1 2; do ( until grep -q "DONE hintsd_teacher" $D/teacher_s$i.log 2>/dev/null; do $(S 12000) $P -u scripts/hintsd.py teacher --traj $T --analysis $D/analysis.jsonl --adapter $START --out $D/targets_s$i.jsonl --shard $i 3 >> $D/teacher_s$i.log 2>&1 || sleep 30; done ) & sleep 60; done; wait
    cat $D/targets_s*.jsonl > $D/targets.jsonl; echo DONE >> $D/teacher.log
  fi
  log "$CH r$R: $(grep -c . $D/targets.jsonl) reverse-KL targets (no replay, no filter)"
  train1 $D $START "$D/targets.jsonl:1.0"; [ -s $E/eval_heldout_a3bhsd_r$R.jsonl ] || eval774 $D a3bhsd_r$R; }
until grep -q "a3b_hintsd R1->R2 DONE" $SA/queue.log; do sleep 120; done
hround 3 $SA/$CH/r2/model/adapter ""
log "$CH R2->R3 DONE"
