#!/bin/bash
# Round-1 variants that use success data, all from the original student; greedy eval on unseen/seen/heldout.
H=$1; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SR="srun --jobid=$H --overlap -n1 -c4 --mem=0"; O=reports/alfworld/opd_plus; D=data/opd_plus/attr_nofb_r1; M=$O/queue.log; mkdir -p $O
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
until [ "$(squeue -j $H -h -o %T)" = "RUNNING" ]; do sleep 30; done; log "hold $H running"
if ! grep -q "DONE prep" $O/prep.log 2>/dev/null; then
  log "prep start"; $SR $P -u scripts/alfworld_opd_plus.py prep >> $O/prep.log 2>&1 || { log "prep FAILED"; exit 1; }; log "prep done"
fi
run_arm() {  # name, then data specs
  local n=$1; shift; local A=$O/$n
  if [ ! -f $A/adapter/adapter_config.json ]; then
    log "$n train ($*)"; $SR $P -u scripts/alfworld_opd_plus.py train --data "$@" --out $A >> $O/$n.log 2>&1 || { log "$n train FAILED"; return; }
  fi
  $SR $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $A/adapter --split valid_unseen --budget 50 --temperature 0 --out $A/eval_unseen.jsonl --tag opd_plus_$n >> $O/$n.log 2>&1
  log "$n unseen: $(grep -c '"won": true' $A/eval_unseen.jsonl)/$(wc -l < $A/eval_unseen.jsonl)"
  scripts/alfworld_eval_extended.sh $H opd_plus_$n $A/adapter; log "$n extended eval done"
}
run_arm M0_kl_offline      $D/kl.jsonl
run_arm M2_kl_anchor       $D/kl.jsonl $D/anchor.jsonl
run_arm M3_rft             $D/rft.jsonl
run_arm M4_dpo             $D/dpo.jsonl
run_arm M6_kl_continuation $D/kl.jsonl $D/cont.jsonl
log "OPD+ QUEUE FINISHED"
