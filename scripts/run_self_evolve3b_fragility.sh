#!/bin/bash
# Fragility alongside 3B self-evolve: placebo (random targets at the attr arm's verified states, same lr), flip probe for every
# trained arm + placebo + the old external-teacher adapter; placebo eval on 274.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; O=reports/alfworld/self_evolve3b; M=$O/queue.log
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; T=reports/alfworld/student3b_low100_train_b50.jsonl; STU=reports/alfworld/student3b_low100/final
until grep -q "DONE distill" $O/attr/run.log 2>/dev/null; do sleep 60; done
$P - <<'PY'
import json, random, os
O="reports/alfworld/self_evolve3b"; rng=random.Random(0)
ex=[json.loads(l) for l in open(f"{O}/attr/round1/examples.jsonl")]; tg={(json.loads(l)["id"],json.loads(l)["t"]):json.loads(l) for l in open(f"{O}/attr/targets.jsonl")}
os.makedirs(f"{O}/placebo",exist_ok=True)
with open(f"{O}/placebo/train.jsonl","w") as f:
    for r in ex:
        T=tg[(r["id"],r["t"])]; cmds=T["admissible"]; c=rng.choice([x for x in cmds if x!=T["a_minus"]])
        f.write(json.dumps({"kind":"kl","src":"reports/alfworld/student3b_low100_train_b50.jsonl","id":r["id"],"t":r["t"],"cmds":cmds,"target":[1.0 if x==c else 0.0 for x in cmds]})+"\n")
print("=== placebo: random targets at", len(ex), "attr-verified states")
PY
J=$(squeue -u pan00389 -h -o "%i %j %T" | awk '$2=="error_attr_hold_igpu_qw" && $3=="RUNNING"{print $1; exit}')
srun --jobid=$J --overlap -n1 -c4 --mem=0 $P -u scripts/alfworld_opd_plus.py train --data $O/placebo/train.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter $STU --lr 1e-5 --epochs 1 --out $O/placebo/model > $O/placebo/train.log 2>&1
for sp in valid_unseen valid_seen; do srun --jobid=$J --overlap -n1 -c4 --mem=0 $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $O/placebo/model/adapter --split $sp --budget 50 --temperature 0 --out $O/placebo/eval_${sp#valid_}.jsonl --tag se3b_placebo >> $O/placebo/train.log 2>&1; done
echo "=== $(date '+%F %T') 3B placebo: unseen $(grep -c '"won": true' $O/placebo/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $O/placebo/eval_seen.jsonl)/140 (student 56/134, 80/140)" >> $M
until grep -q "3B SELF-EVOLVE ROUND 1 DONE" $M; do sleep 60; done
SPECS="placebo=$O/placebo/model/adapter external-teacher-attr_nofb=reports/alfworld/opd/student3b_low100/attr_nofb/round1/adapter"
for n in attr m5 selfattr entropy rand_early; do SPECS="$SPECS self-$n=$O/$n/round1/adapter"; done
srun --jobid=$J --overlap -n1 -c4 --mem=0 $P -u scripts/measure_flip_generic.py --base_model Qwen/Qwen2.5-3B-Instruct --base_adapter $STU --traj $T --states $O/flip_states150.jsonl --cache $O/flip_base150.json $SPECS > $O/flip.log 2>&1
echo "=== $(date '+%F %T') 3B fragility probe done" >> $M
