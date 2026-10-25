#!/bin/bash
# OPD vs SFT, round 1 from the original student, same offline trainer and greedy 774-game eval as the OPD+ arms (M0 = offline KL).
#  S1 SFT on the same 125 verified states, hard label a+ (vs M0: teacher distribution)
#  S2 SFT on 125 fresh expert states (13 trajectories, none from any eval set)
#  S3 SFT on 1,176 fresh expert states (132 trajectories)
H=$1; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SR="srun --jobid=$H --overlap -n1 -c4 --mem=0"; O=reports/alfworld/sft_compare; D=data/opd_plus/sft_compare; M=$O/queue.log; mkdir -p $O
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
run_arm() {
  local n=$1; shift; local A=$O/$n
  if [ ! -f $A/adapter/adapter_config.json ]; then
    log "$n train"; $SR $P -u scripts/alfworld_opd_plus.py train --data "$@" --out $A >> $O/$n.log 2>&1 || { log "$n train FAILED"; return; }
  fi
  $SR $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $A/adapter --split valid_unseen --budget 50 --temperature 0 --out $A/eval_unseen.jsonl --tag sftc_$n >> $O/$n.log 2>&1
  log "$n unseen: $(grep -c '"won": true' $A/eval_unseen.jsonl)/$(wc -l < $A/eval_unseen.jsonl)"
  scripts/alfworld_eval_extended.sh $H sftc_$n $A/adapter; log "$n extended eval done"
}
run_arm S1_sft_aplus $D/sft_aplus.jsonl
run_arm S2_sft_expert_matched $D/sft_expert_matched.jsonl
run_arm S3_sft_expert_1k $D/sft_expert_1k.jsonl
log "SFT COMPARE FINISHED"
