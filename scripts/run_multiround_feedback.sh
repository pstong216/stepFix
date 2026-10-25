#!/bin/bash
# Does debating for more rounds after the attributor's feedback make the team revise / get it right more often?
H=$1; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; SR="srun --jobid=$H --overlap -n1 -c8 --mem=0"
O=reports/selfplay_chain_debate_multiround; mkdir -p $O; M=$O/queue.log
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
until [ "$(squeue -j $H -h -o %T)" = "RUNNING" ]; do sleep 30; done; log "hold $H running on $(squeue -j $H -h -o %N)"
IDS=reports/selfplay_chain_debate_round_150/true_failure_ids.json
$P -c "import json; json.dump(json.load(open('$IDS'))[:2], open('$O/smoke_ids.json','w'))"
log "smoke"; $SR $P -u scripts/run_selfplay_feedback_chain.py --framework debate --protocol debate_round --ids_file $O/smoke_ids.json --conditions no_feedback,incorrect_only --n_extra_rounds 2 --out_dir $O/smoke > $O/smoke.log 2>&1 \
  && $P -c "import json; r=[json.loads(l) for l in open('$O/smoke/corrections.jsonl')]; assert all(len(x['by_round'])==2 for x in r), 'by_round missing'; print('smoke ok', len(r))" >> $M 2>&1 || { log "smoke FAILED"; exit 1; }
log "full run: 118 true failures x 5 conditions x 4 rounds"
$SR $P -u scripts/run_selfplay_feedback_chain.py --framework debate --protocol debate_round --ids_file $IDS \
  --conditions no_feedback,incorrect_only,precise_reason,expert_incorrect_only,expert_precise_reason --n_extra_rounds 4 --out_dir $O/full > $O/full.log 2>&1 && log "full done" || log "full FAILED"
