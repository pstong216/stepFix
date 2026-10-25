#!/bin/bash
# Final-recipe chains (user priority, Oct 4 evening): steps = self-attributor A_k; teacher = hindsight + Model A diagnosis + own
# analysis; filter = env one-step top-86 (+86 replay); R1 -> R3.  Runs in parallel on any free card:
#   selfattr_s1, selfattr_s2   : two more seeds of the main chain (seed 0 = selfattr, already running)
#   selfattr_rollout           : same recipe, filter = rollout verification        (the "vs rollout" comparison for the final method)
#   selfattr_placebo           : same recipe, filter = random state + random action (noise floor)
cd /users/8/pan00389/multi-agent/error-attribution; SA=reports/alfworld/self_attr_rounds; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
A0=reports/alfworld/self_attr/r0/both/adapter; S0=reports/alfworld/student3b_low100/final; export ERROR_ATTR_ROOT=$PWD
# Model A diagnoses for the original collection (round 1 of every chain reuses it)
for d in selfattr_s1 selfattr_s2 selfattr_rollout selfattr_placebo; do mkdir -p $SA/$d/r1
  $P -c "import json; rows=[json.loads(l) for l in open('reports/alfworld/attr7bnoobs_student3b_low100_train_b50.jsonl')]; open('$SA/$d/r1/diag_modelA.raw.jsonl','w').writelines(json.dumps({'id':r['id'],'t_hat':r['t_hat'],'type':'Model-A mode '+str(r['pred'].get('error_mode'))})+'\n' for r in rows)"; done
chain() { local CH=$1 SUF=$2 SEED=$3; local S=$S0 AT=$A0
  for R in 1 2 3; do CHAIN_SUFFIX=$SUF SEED=$SEED bash scripts/self_attr_round.sh $CH $R $S $AT any || break; S=$(cat $SA/$CH$SUF/r$R/ADAPTER); AT=$(cat $SA/$CH$SUF/r$R/ATTR); done
  echo "=== $(date '+%F %T') FINAL CHAIN $CH$SUF DONE" | tee -a $SA/queue.log; }
chain selfattr _s1 1 &
sleep 120; chain selfattr _s2 2 &
sleep 120; chain selfattr_rollout "" 0 &
sleep 120; chain selfattr_placebo "" 0 &
wait
