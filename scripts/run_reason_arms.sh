#!/bin/bash
# Training arms for the better-informed self-teacher (attr pointer states): candidates from the teacher that first writes an
# analysis (reason_hind / reason_plain) or is given Model A's diagnosis (modeA_hind); action chosen by env_top86 (no rollout), plus
# reason_hind selected by the teacher's own confidence (hind_top).  Hold resolved by name before every GPU step (survives expiry).
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; SE=reports/alfworld/self_evolve3b; M=$SE/queue.log
STU=reports/alfworld/student3b_low100/final; export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SE_DIR=$SE
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
hold() { squeue -u pan00389 -h -o "%i %j %T" | awk '$1!="" && $2=="error_attr_hold_igpu" && $3=="RUNNING"{print $1}' | sort -n | tail -n1; }
S() { local H; until H=$(hold); [ -n "$H" ]; do sleep 120; done; echo "srun --jobid=$H --overlap -n1 -c4 --mem=0"; }
free_mb() { $(S) nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1; }
guard() { until [ "$(free_mb)" -ge "$1" ] 2>/dev/null; do sleep 60; done; }
until grep -q "DONE teacher_reason" $SE/proxy/teacher_reason_attr.log 2>/dev/null; do sleep 60; done
[ -s $SE/modeA_hind/targets.jsonl ] || { guard 12000; $(S) $P -u scripts/build_reason_targets.py attr > $SE/proxy/build_reason_targets.log 2>&1; }
until grep -q "DONE onestep" $SE/proxy/onestep6.log 2>/dev/null; do srun --jobid=2239109 --overlap -n1 -c2 --mem=20G $P -u scripts/proxy_eval_se3b.py onestep --arms reason_hind reason_plain modeA_hind --limit 250 >> $SE/proxy/onestep6.log 2>&1; sleep 5; done
arm() { local A=$1 rule=$2; local D=$SE/proxy_arms/${A}_$rule
  $P scripts/proxy_arms_se3b.py --arm $A --rule $rule --k 86 | tee -a $M
  guard 24000; $(S) $P -u scripts/alfworld_opd_plus.py train --data $D/train.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter $PWD/$STU --lr 1e-5 --epochs 1 --batch 8 --out $D/model > $D/train.log 2>&1 || { log "proxy $A/$rule: TRAIN FAILED"; return; }
  guard 10000; for sp in valid_unseen valid_seen; do $(S) $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $D/model/adapter --split $sp --budget 50 --temperature 0 --out $D/eval_${sp#valid_}.jsonl --tag px_${A}_$rule >> $D/train.log 2>&1; done
  log "3B proxy $A/$rule (better-informed self-teacher): $(wc -l < $D/train.jsonl) targets ($(python3 -c "import json;s=json.load(open('$D/stats.json'));print(f\"{s['rollout_won']} would pass rollout\")")) | unseen $(grep -c '"won": true' $D/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen.jsonl)/140 (student 56/134, 80/140; hind cands env_top 172/169)"; }
log "reasoning-teacher arms: reason_hind/env_top, reason_hind/hind_top, modeA_hind/env_top, reason_plain/env_top"
arm reason_hind env_top; arm reason_hind hind_top; arm modeA_hind env_top; arm reason_plain env_top
log "REASONING-TEACHER ARMS DONE"
