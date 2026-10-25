#!/bin/bash
# Round-1 "always step 0" control: same start, same 319 failures, same 3B SFT teacher and recipe as attr_nofb / M5.
H=$1; cd /users/8/pan00389/multi-agent/error-attribution; A=reports/alfworld/opd/student3b_low100; M=$A/const0_queue.log
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
until [ "$(squeue -j $H -h -o %T)" = "RUNNING" ]; do sleep 30; done; log "hold $H running"
TAG=student3b_low100 TEACHER=Qwen/Qwen2.5-3B-Instruct TEACHER_ADAPTER=reports/alfworld/student3b_sft/final ROUND=round1 IDS_FROM=$A/attr/steps.jsonl \
  bash scripts/run_alfworld_opd_arm.sh $H const0_nofb 1000 2 && [ -f $A/const0_nofb/round1/adapter/adapter_config.json ] || { log "const0 R1 FAILED"; exit 1; }
log "const0_nofb R1 verified $(wc -l < $A/const0_nofb/round1/examples.jsonl)/319, unseen $(grep -c '"won": true' $A/const0_nofb/eval_unseen_round1.jsonl)/134"
scripts/alfworld_eval_extended.sh $H opd_const0_nofb $A/const0_nofb/round1/adapter; log "const0 R1 done"
