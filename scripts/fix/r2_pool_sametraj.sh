#!/bin/bash
# Same-trajectory R2 test of ENVIRONMENT ADJUDICATION: on the main final chain's R2 trajectories, combine the stall-entry scan
# (selfattr_hind/r2) and the first-step scan (selfattr_hindfix/r2); filter env_pool; train from the same R1 adapter; eval.
cd /users/8/pan00389/multi-agent/error-attribution; SA=reports/alfworld/self_attr_rounds; L=$SA/selfattr_hind/r2; F=$SA/selfattr_hindfix/r2; N=$SA/selfattr_hindpool/r2
until grep -q "DONE onestep" $L/onestep.log 2>/dev/null && grep -q "DONE onestep" $F/onestep.log 2>/dev/null && [ -s $L/replay.jsonl ]; do sleep 120; done
mkdir -p $N/self $N/proxy $N/envlab $N/attr_labels
for f in traj.jsonl traj.release.jsonl collect.log replay.jsonl ids_fail.jsonl; do cp $L/$f $N/; done
cat $L/self/steps.jsonl $F/self/steps.jsonl > $N/pointer.jsonl; cp $N/pointer.jsonl $N/self/steps.jsonl
cat $L/self/targets.jsonl $F/self/targets.jsonl > $N/self/targets.jsonl; cat $L/proxy/onestep.jsonl $F/proxy/onestep.jsonl > $N/proxy/onestep.jsonl
/users/8/pan00389/miniconda3/envs/error_attr/bin/python3 -c "
import json; rows=[json.loads(l) for l in open('$N/pointer.jsonl')]
open('$N/pointer.raw.jsonl','w').writelines(json.dumps({'id':r['id'],'t_hat':r['steps'][0],'type':None})+'\n' for r in rows)
open('$N/diag.raw.jsonl','w').writelines(json.dumps({'id':r['id'],'t_hat':r['steps'][0],'type':None})+'\n' for r in rows)"
echo "DONE informed_teacher (combined)" > $N/teacher.log; echo "DONE onestep (combined)" > $N/onestep.log
echo "=== $(date '+%F %T') R2 same-trajectory control: ENVIRONMENT ADJUDICATION (selfattr_hindpool) built from stall-entry + first-step scans" | tee -a $SA/queue.log
EVIDENCE=generic STEP=logfix STEP_POOL=1 TEACHER_NO_DIAG=1 K=100 CHAIN_SUFFIX=_hindpool SEED=0 bash scripts/self_attr_round.sh selfattr 2 $(cat $SA/selfattr_hind/r1/ADAPTER) reports/alfworld/self_attr/r0/both/adapter any
Q=reports/alfworld/ext/heldout_queue.txt; for t in hind:selfattr_hind hindfix:selfattr_hindfix hindpool:selfattr_hindpool; do n=${t%%:*}; d=${t##*:}; grep -q "^selfattr_${n}_r2	" $Q || printf "selfattr_%s_r2\t%s/%s/r2/model/adapter\n" $n $SA $d >> $Q; done
echo "=== $(date '+%F %T') R2 same-trajectory control DONE (selfattr_hindpool)" | tee -a $SA/queue.log
