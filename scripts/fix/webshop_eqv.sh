#!/bin/bash
# WebShop equal-count arm: env top-N with N = number of states that have a rollout winner (same candidates), + 100 replay.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; O=reports/webshop; R=$O/round1; G=$R/generic; STU=$O/student100/final; T=$R/traj_train.jsonl; M=$O/queue.log
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WEBSHOP_SERVER=http://acn216:38300
pick() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge ${1:-10000} ] 2>/dev/null && echo $best; }
S() { local H=""; until H=$(pick $1); [ -n "$H" ]; do sleep 60; done; echo "srun --jobid=$H --overlap -n1 -c4 --mem=0"; }
D=$G/arm_env_top_eqV; mkdir -p $D
$P scripts/webshop_generic.py filter --traj $T --scan $G/scan.jsonl --roll $G/rollout.jsonl --rule rollout --k 100000 --out $G/rollout_all_count.jsonl > /dev/null
N=$(wc -l < $G/rollout_all_count.jsonl); [ -s $D/train.jsonl ] || $P scripts/webshop_generic.py filter --traj $T --scan $G/scan.jsonl --rule env_top --k $N --out $D/train.jsonl | tee -a $M
echo "=== $(date '+%F %T') webshop generic: equal-count arm env top-$N (rollout winner states = $N)" | tee -a $M
cat $D/train.jsonl $R/replay.jsonl > $D/train_replay.jsonl
until [ -f $D/model/adapter/adapter_config.json ]; do $(S 16000) $P -u scripts/webshop_evidence.py train --data $D/train_replay.jsonl --adapter $STU --out $D/model > $D/train.log 2>&1 || sleep 120; done
until grep -q "DONE collect" $D/eval.log 2>/dev/null; do $(S 10000) $P -u scripts/webshop_agent.py collect --adapter $D/model/adapter --split dev --budget 15 --temperature 0 --out $D/eval_dev.jsonl --tag wsg_eqv >> $D/eval.log 2>&1 || sleep 60; done
echo "=== $(date '+%F %T') WEBSHOP generic r1 arm env_top_eqV: $N targets | dev $(grep DONE $D/eval.log | tail -n1 | sed 's/DONE collect //')" | tee -a $M reports/alfworld/self_evolve3b/queue.log
