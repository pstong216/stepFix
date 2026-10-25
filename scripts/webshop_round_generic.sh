#!/bin/bash
# WebShop round 1 with the SAME generic recipe as ALFWorld (Oct 5 23:20): generic evidence, late log rule (tentative, user), K=100 +
# 100 replay; main comparison env top-100 vs rollout (all winners; and random 100 of them); placebo as the floor.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; O=reports/webshop; R=$O/round1; G=$R/generic; mkdir -p $G; M=$O/queue.log
K=100; STU=$O/student100/final; T=$R/traj_train.jsonl; export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WEBSHOP_SERVER=http://acn216:38300
log() { echo "=== $(date '+%F %T') $*" | tee -a $M reports/alfworld/self_evolve3b/queue.log; }
pick() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge ${1:-10000} ] 2>/dev/null && echo $best; }
S() { local H=""; until H=$(pick $1); [ -n "$H" ]; do sleep 60; done; echo "srun --jobid=$H --overlap -n1 -c4 --mem=0"; }
[ -s $G/pointer.jsonl ] || $P scripts/webshop_generic.py pointer --traj $T --rule late --out $G/pointer.jsonl | tee -a $M
until grep -q "DONE gscan" $G/scan.log 2>/dev/null; do $(S 12000) $P -u scripts/webshop_generic.py scan --traj $T --pointer $G/pointer.jsonl --adapter $STU --out $G/scan.jsonl >> $G/scan.log 2>&1 || sleep 60; done
until grep -q "DONE grollout" $G/rollout.log 2>/dev/null; do $(S 12000) $P -u scripts/webshop_generic.py rollout --traj $T --scan $G/scan.jsonl --adapter $STU --budget 15 --out $G/rollout.jsonl >> $G/rollout.log 2>&1 || sleep 60; done
log "webshop generic: $(grep 'DONE grollout' $G/rollout.log | tail -n1) | rollout winners $(grep -c '"won": true' $G/rollout.jsonl) candidates of $(grep -c . $G/rollout.jsonl)"
for rule in env_top rollout rollsub placebo; do D=$G/arm_$rule; mkdir -p $D
  [ -s $D/train.jsonl ] || $P scripts/webshop_generic.py filter --traj $T --scan $G/scan.jsonl --roll $G/rollout.jsonl --rule $rule --k $K --out $D/train.jsonl | tee -a $M
  cat $D/train.jsonl $R/replay.jsonl > $D/train_replay.jsonl
  until [ -f $D/model/adapter/adapter_config.json ]; do $(S 16000) $P -u scripts/webshop_evidence.py train --data $D/train_replay.jsonl --adapter $STU --out $D/model > $D/train.log 2>&1 || { log "webshop generic arm $rule: TRAIN FAILED"; sleep 120; }; done
  until grep -q "DONE collect" $D/eval.log 2>/dev/null; do $(S 10000) $P -u scripts/webshop_agent.py collect --adapter $D/model/adapter --split dev --budget 15 --temperature 0 --out $D/eval_dev.jsonl --tag wsg_$rule >> $D/eval.log 2>&1 || sleep 60; done
  log "WEBSHOP generic r1 arm $rule: $(wc -l < $D/train.jsonl) targets | dev $(grep DONE $D/eval.log | tail -n1 | sed 's/DONE collect //')"
done
log "WEBSHOP GENERIC ROUND 1 DONE"
