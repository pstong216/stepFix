#!/bin/bash
# Model A check (user Oct 6 11:00): same R2 trajectories and the same pool proposals as selfattr_hindpool/r2 (first-step R1 lineage);
# teacher = hindsight + MODEL A diagnosis + own analysis (instead of hindsight + analysis); environment adjudication.
# Two budgets from the same teacher/scan: K=40%+replay100 (vs selfattr_hindpoolk40r100: 189 / 774 536) and K=100 (vs selfattr_hindpool: 182 / 514).
cd /users/8/pan00389/multi-agent/error-attribution; SA=reports/alfworld/self_attr_rounds; R1A=$(cat $SA/selfattr_hind/r1/ADAPTER)
EVIDENCE=generic STEP=logfix STEP_POOL=1 K_FRAC=0.4 REPLAY_N=100 CHAIN_SUFFIX=_hindpoolA SEED=0 bash scripts/self_attr_round.sh selfattr 2 $R1A reports/alfworld/self_attr/r0/both/adapter any
echo "=== $(date '+%F %T') Model A check (K40+replay100) DONE (selfattr_hindpoolA)" | tee -a $SA/queue.log
S=$SA/selfattr_hindpoolA/r2; N=$SA/selfattr_hindpoolA100/r2; mkdir -p $N/self $N/proxy $N/envlab $N/attr_labels
for f in traj.jsonl traj.release.jsonl collect.log ids_fail.jsonl pointer.jsonl pointer.raw.jsonl diag.raw.jsonl diag_modelA.raw.jsonl modelA_pred.jsonl modelA.log teacher.log onestep.log; do cp $S/$f $N/ 2>/dev/null; done; cp $S/self/steps.jsonl $S/self/targets.jsonl $N/self/; cp $S/proxy/onestep.jsonl $N/proxy/
EVIDENCE=generic STEP=logfix STEP_POOL=1 K=100 CHAIN_SUFFIX=_hindpoolA100 SEED=0 bash scripts/self_attr_round.sh selfattr 2 $R1A reports/alfworld/self_attr/r0/both/adapter any
echo "=== $(date '+%F %T') Model A check (K100) DONE (selfattr_hindpoolA100)" | tee -a $SA/queue.log
