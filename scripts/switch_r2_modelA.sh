#!/bin/bash
# One-off: run Model A on selfattr/r2 trajectories, then restart round 2 at the teacher stage with the Model A diagnosis.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; SA=reports/alfworld/self_attr_rounds; D=$SA/selfattr/r2
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pick() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge 18000 ] 2>/dev/null && echo $best; }
until grep -q DONE $D/modelA.log 2>/dev/null; do H=""; until H=$(pick); [ -n "$H" ]; do sleep 60; done
  srun --jobid=$H --overlap -n1 -c4 --mem=0 $P -u scripts/alfworld_attribute.py --attributor reports/attributor_7b_no_observations/final --variant no_observations --release $D/traj.release.jsonl --out $D/modelA_pred.jsonl >> $D/modelA.log 2>&1 && echo DONE >> $D/modelA.log || sleep 60; done
$P -c "import json; rows=[json.loads(l) for l in open('$D/modelA_pred.jsonl')]; open('$D/diag_modelA.raw.jsonl','w').writelines(json.dumps({'id':r['id'],'t_hat':r['t_hat'],'type':'Model-A mode '+str(r['pred'].get('error_mode'))})+'\n' for r in rows); print('Model A diagnoses', len(rows))"
until [ -f $D/attr/adapter/adapter_config.json ] && [ -s $D/pointer.raw.jsonl ]; do sleep 30; done
tmux kill-session -t selfattr_v3 2>/dev/null; sleep 2
for p in $(pgrep -f "self_attr_round.sh selfattr 2"); do kill $p 2>/dev/null; done
for p in $(pgrep -f "self_teacher_informed.py --traj $D/traj.jsonl"); do kill $p 2>/dev/null; done
sleep 5; rm -f $D/teacher.log $D/self/targets.jsonl $D/onestep.log; rm -rf $D/proxy/onestep.jsonl $D/targets_env_top
echo "=== $(date '+%F %T') selfattr R2 restarted at the teacher stage with Model A diagnosis (hindsight + Model A + own analysis)" | tee -a $SA/queue.log
S=$(cat $SA/selfattr/r1/ADAPTER); AT=$(cat $SA/selfattr/r1/ATTR)
for R in 2 3; do bash scripts/self_attr_round.sh selfattr $R $S $AT error_attr_hold_igpu_qw || break; S=$(cat $SA/selfattr/r$R/ADAPTER); AT=$(cat $SA/selfattr/r$R/ATTR); done
echo "=== $(date '+%F %T') SELFATTR V3 CHAIN DONE" | tee -a $SA/queue.log
