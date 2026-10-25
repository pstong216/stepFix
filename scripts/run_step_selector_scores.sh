#!/bin/bash
# GPU side of the step-selector comparison (user request Oct 5): M5 scores needed to compare
#   log-evidence rule | M5 fixed delta | M5 calibrated change-point | union    against the replay-based env-evidence labels.
#   S0 = original student (failures 319 with env labels; 200 wins for calibration), scored with the original M5
#   S1 = selfattr/r1 student (failures 352 with env labels; 200 wins), scored with an M5 refreshed on S1's own wins (as a round would)
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; O=reports/alfworld/step_selectors; mkdir -p $O
M5_0=reports/alfworld/progress_model/model/adapter; T0=reports/alfworld/student3b_low100_train_b50.jsonl; T1=reports/alfworld/self_attr_rounds/selfattr/r2/traj.jsonl
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pick() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge "${1:-10000}" ] 2>/dev/null && echo $best; }
S() { local H=""; until H=$(pick $1); [ -n "$H" ]; do sleep 60; done; echo "srun --jobid=$H --overlap -n1 -c4 --mem=0"; }
log() { echo "=== $(date '+%F %T') $*" | tee -a reports/alfworld/self_evolve3b/queue.log; }
# id lists: failures with env labels; 200 random wins each
$P - <<'PY'
import json, random
O="reports/alfworld/step_selectors"; rng=random.Random(0)
for name, traj, lab in (("s0","reports/alfworld/student3b_low100_train_b50.jsonl","reports/alfworld/self_attr_labels/r0/labels.jsonl"),
                        ("s1","reports/alfworld/self_attr_rounds/selfattr/r2/traj.jsonl","reports/alfworld/self_attr_rounds/selfattr/r2/envlab/labels.jsonl")):
    eps=[json.loads(l) for l in open(traj)]; fail_ids={json.loads(l)["id"] for l in open(lab)}
    wins=[e["id"] for e in eps if e["won"]]; rng.shuffle(wins)
    open(f"{O}/{name}_fail_ids.jsonl","w").writelines(json.dumps({"id":i})+"\n" for i in fail_ids)
    open(f"{O}/{name}_win_ids.jsonl","w").writelines(json.dumps({"id":i})+"\n" for i in wins[:200])
    print(name, "failures", len(fail_ids), "wins sampled", min(200,len(wins)))
PY
# S1's refreshed M5 (as the chain would build it)
until [ -f $O/m5_s1/adapter/adapter_config.json ]; do $(S 20000) $P -u scripts/alfworld_progress_model.py train --traj $T1 --adapter $M5_0 --n 4000 --out $O/m5_s1 </dev/null > $O/m5_s1.train.log 2>&1 || sleep 60; done
for spec in s0_fail:$T0:$M5_0 s0_win:$T0:$M5_0 s1_fail:$T1:$O/m5_s1/adapter s1_win:$T1:$O/m5_s1/adapter; do IFS=: read -r name traj ad <<< "$spec"; ids=$O/${name%_*}_${name#*_}_ids.jsonl
  until grep -q "DONE score" $O/$name.score.log 2>/dev/null; do $(S 10000) $P -u scripts/alfworld_progress_model.py score --traj $traj --adapter $ad --ids_from $ids --out $O/$name.scores.jsonl </dev/null >> $O/$name.score.log 2>&1 || sleep 60; done; done
log "step-selector M5 scores done (s0/s1 failures + 200 wins each)"
$P scripts/compare_step_selectors.py 2>&1 | tee -a reports/alfworld/self_evolve3b/queue.log
