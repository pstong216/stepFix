#!/bin/bash
# Self-trained detour detector (labels from replay-verified pruning of the student's own wins) as a step-selection rule.
# Phase 1 (ee5239 hold, after the attr arm's eval): train, score the 319 failures, write pointers, compare with Model A / M5.
# Phase 2 (qianwen hold, after the compression-SFT runs): self-evolve arm "detour" = hindsight self-teacher at the detector's
# step, verified KL distill, eval on unseen 134 + seen 140.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; O=reports/alfworld/detour_detector; SE=reports/alfworld/self_evolve3b; M=$SE/queue.log
T=reports/alfworld/student3b_low100_train_b50.jsonl; STU=reports/alfworld/student3b_low100/final
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SE_DIR=$SE SE_TRAJ=$T SE_MODEL=Qwen/Qwen2.5-3B-Instruct SE_ADAPTER=$PWD/$STU
S1="srun --jobid=2543688 --overlap -n1 -c4 --mem=0"; S2="srun --jobid=2543689 --overlap -n1 -c4 --mem=0"
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
mkdir -p $O $SE/detour
until grep -q "3B self attr:" $M 2>/dev/null; do sleep 60; done
log "detour detector: train (6000 states) + score 319 failures"
$S1 $P -u scripts/alfworld_detour_detector.py train --out $O --n 6000 > $O/train.log 2>&1 || { log "detour detector TRAIN FAILED"; exit 1; }
$S1 $P -u scripts/alfworld_detour_detector.py score --adapter $O/adapter --ids_from $SE/ids319.jsonl --out $O/scores319.jsonl > $O/score.log 2>&1
$P scripts/alfworld_detour_detector.py pointer --scores $O/scores319.jsonl --out $SE/detour/steps.jsonl
$P - <<'PY' | tee -a $M
import json, statistics
SE="reports/alfworld/self_evolve3b"; A="reports/alfworld/opd/student3b_low100"
d={json.loads(l)["id"]:json.loads(l) for l in open(f"{SE}/detour/steps.jsonl")}
attr={json.loads(l)["id"]:json.loads(l)["steps"][0] for l in open(f"{A}/attr_nofb/steps.jsonl")}
m5={json.loads(l)["id"]:json.loads(l)["steps"][0] for l in open(f"{A}/progress_nofb/steps.jsonl")}
ids=list(d); ts=[d[i]["steps"][0] for i in ids]
print(f"=== detour pointer: mean t {statistics.mean(ts):.1f} median {statistics.median(ts):.0f} | t0 {sum(t==0 for t in ts)/len(ts):.2f} t<=2 {sum(t<=2 for t in ts)/len(ts):.2f} | "
      f"exact/within-1 vs Model A {sum(d[i]['steps'][0]==attr[i] for i in ids)/len(ids):.2f}/{sum(abs(d[i]['steps'][0]-attr[i])<=1 for i in ids)/len(ids):.2f} | vs M5 {sum(d[i]['steps'][0]==m5[i] for i in ids)/len(ids):.2f}/{sum(abs(d[i]['steps'][0]-m5[i])<=1 for i in ids)/len(ids):.2f} | mean P(detour) at pointer {statistics.mean(d[i]['p'] for i in ids):.2f}")
PY
# phase 2: self-evolve arm at the detector's step (qianwen hold, after the compression-SFT runs)
until grep -q "COMPRESS-SFT DONE" reports/alfworld/compress_traj/queue.log 2>/dev/null; do sleep 60; done
log "3B self-evolve arm detour: hindsight targets + verify/distill + eval"
D=$SE/detour
$S2 $P -u scripts/self_evolve.py hindsight_steps --steps $D/steps.jsonl > $D/h.log 2>&1
until grep -q "DONE distill" $D/run.log 2>/dev/null; do
  $S2 $P -u scripts/alfworld_attr_opd.py distill --traj $T --targets $D/targets.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter $STU --out $D/round1 --verify_k 2 --exclude_a_minus --max_examples_per_run 250 >> $D/run.log 2>&1 || sleep 30; done
for sp in valid_unseen valid_seen; do $S2 $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $D/round1/adapter --split $sp --budget 50 --temperature 0 --out $D/eval_${sp#valid_}.jsonl --tag se3b_detour >> $D/run.log 2>&1; done
log "3B self detour: verified $(wc -l < $D/round1/examples.jsonl)/319 | unseen $(grep -c '"won": true' $D/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen.jsonl)/140 (student 56/134, 80/140)"
