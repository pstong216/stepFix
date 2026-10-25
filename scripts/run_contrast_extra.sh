#!/bin/bash
# Success/failure-contrast arms on the original collection's divergence pairs (409 states): DPO vs one-hot SFT, all pairs vs the
# env-evidence-consistent subset.  Trained from the SFT student; eval 274 held-out games greedy.  GPU = hold with most free memory.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; O=reports/alfworld/self_evolve3b/dpo_arms; mkdir -p $O
M=reports/alfworld/self_evolve3b/queue.log; STU=reports/alfworld/student3b_low100/final; export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
pick() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge "${1:-10000}" ] 2>/dev/null && echo $best; }
S() { local H=""; until H=$(pick $1); [ -n "$H" ]; do sleep 60; done; echo "srun --jobid=$H --overlap -n1 -c4 --mem=0"; }
until grep -q "DONE dpo_prep" $O/prep.log 2>/dev/null; do $(S 12000) $P -u scripts/dpo_arms_prep.py --pairs reports/alfworld/self_attr_labels/div/dpo_pairs.jsonl --adapter $STU --out_dir $O > $O/prep.log 2>&1 || sleep 60; done
log "dpo arms: $(grep DONE $O/prep.log)"
for arm in sft_86; do D=$O/$arm; mkdir -p $D
  until [ -f $D/model/adapter/adapter_config.json ]; do $(S 16000) $P -u scripts/alfworld_opd_plus.py train --data $O/$arm.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter $PWD/$STU --lr 1e-5 --epochs 1 --batch 8 --beta 0.1 --out $D/model > $D/train.log 2>&1 || { log "dpo arm $arm: TRAIN FAILED"; sleep 120; }; done
  for sp in valid_unseen valid_seen; do until [ "$(grep -c . $D/eval_${sp#valid_}.jsonl 2>/dev/null)" -ge $([ $sp = valid_unseen ] && echo 134 || echo 140) ]; do
    $(S 10000) $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $D/model/adapter --split $sp --budget 50 --temperature 0 --out $D/eval_${sp#valid_}.jsonl --tag dpo_$arm >> $D/eval.log 2>&1 || sleep 30; done; done
  log "3B contrast arm (budget-matched) $arm: $(wc -l < $O/$arm.jsonl) pairs | unseen $(grep -c '"won": true' $D/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen.jsonl)/140 (student 56/134, 80/140; env_top86 172/169)"
done
log "DPO ARMS DONE"
