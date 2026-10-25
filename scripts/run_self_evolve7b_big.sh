#!/bin/bash
# 7B self-evolve scale-up: 1500 zero-shot games -> Model A step on every failure -> hindsight self-teacher candidates ->
# one-step env evidence -> veto-priority ordered rollout verification (resumable across GPU holds) -> DPO on REAL fixes vs
# matched placebo, 2 seeds, lr 1e-6, 1 epoch -> eval 274 games.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; O=reports/alfworld/self_evolve7b; M=$O/queue.log
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SE_DIR=$O SE_TRAJ=$O/r1_big.jsonl SE_MODEL=Qwen/Qwen2.5-7B-Instruct SE_ADAPTER=
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
hold() { squeue -u pan00389 -h -o "%i %j %T" | awk -v n="$1" '$2==n && $3=="RUNNING"{print $1; exit}'; }
J() { local j=""; while [ -z "$j" ]; do j=$(hold $1); [ -z "$j" ] && sleep 60; done; echo $j; }
G1() { J error_attr_hold_igpu; }; G2() { J error_attr_hold_igpu_qw; }
until grep -q "^DONE" <(tail -3 $O/collect_c.log) && grep -q "^DONE" <(tail -3 $O/collect_d.log); do sleep 60; done
cat $O/r1_train_{a,b,c,d}.jsonl > $O/r1_big.jsonl; cat $O/r1_train_{a,b,c,d}.release.jsonl > $O/r1_big.release.jsonl
$P - <<'PY' >> $M
import json
O="reports/alfworld/self_evolve7b"; E=[json.loads(l) for l in open(f"{O}/r1_big.jsonl")]; F=[e["id"] for e in E if not e["won"]]
have={json.loads(l)["id"] for l in open(f"{O}/attr7b.jsonl")}; need=set(F)-have
open(f"{O}/fail_big_need.release.jsonl","w").writelines(l for l in open(f"{O}/r1_big.release.jsonl") if json.loads(l)["id"] in need)
open(f"{O}/ids_big.jsonl","w").writelines(json.dumps({"id":i})+"\n" for i in F)
print(f"=== big collection: {len(E)} games, {len(E)-len(F)} won, {len(F)} failures; Model A needed for {len(need)}")
PY
srun --jobid=$(G1) --overlap -n1 -c4 --mem=0 $P -u scripts/alfworld_attribute.py --attributor reports/attributor_7b_no_observations/final --variant no_observations --release $O/fail_big_need.release.jsonl --out $O/attr7b.jsonl > $O/attr_big.log 2>&1
$P - <<'PY'
import json, os
O="reports/alfworld/self_evolve7b"; A={json.loads(l)["id"]:json.loads(l)["t_hat"] for l in open(f"{O}/attr7b.jsonl")}
E={json.loads(l)["id"]:json.loads(l) for l in open(f"{O}/r1_big.jsonl")}; ids=[json.loads(l)["id"] for l in open(f"{O}/ids_big.jsonl")]
for h, part in (("a", ids[0::2]), ("b", ids[1::2])):
    d=f"{O}/attr_big_{h}"; os.makedirs(f"{d}/round1", exist_ok=True); open(f"{d}/round1/examples.jsonl","a").close()
    open(f"{d}/steps.jsonl","w").writelines(json.dumps({"id":i,"steps":[min(A[i],len(E[i]["steps"])-1)],"mode":None})+"\n" for i in part)
PY
log "big: hindsight self-teacher candidates at Model A steps"
srun --jobid=$(G1) --overlap -n1 -c4 --mem=0 $P -u scripts/self_evolve.py hindsight_steps --steps $O/attr_big_a/steps.jsonl > $O/attr_big_a/h.log 2>&1 &
srun --jobid=$(G2) --overlap -n1 -c4 --mem=0 $P -u scripts/self_evolve.py hindsight_steps --steps $O/attr_big_b/steps.jsonl > $O/attr_big_b/h.log 2>&1 &
wait
for h in a b; do ( until grep -q "DONE onestep" <(tail -2 $O/attr_big_$h/onestep.log 2>/dev/null); do srun --jobid=2239109 --overlap -n1 -c2 --mem=20G $P -u scripts/env_evidence_filter.py --teacher_dir $O/attr_big_$h --traj $O/r1_big.jsonl onestep >> $O/attr_big_$h/onestep.log 2>&1; sleep 3; done ) & done; wait
mkdir -p $O/attr_big/round1; touch $O/attr_big/round1/examples.jsonl
cat $O/attr_big_{a,b}/steps.jsonl > $O/attr_big/steps.jsonl; cat $O/attr_big_{a,b}/targets.jsonl > $O/attr_big/targets.jsonl; cat $O/attr_big_{a,b}/onestep.jsonl > $O/attr_big/onestep.jsonl
$P scripts/self_evolve.py order --steps $O/attr_big/steps.jsonl >> $M 2>&1
log "big: veto-priority rollout verification (resumable across holds)"
N=$(wc -l < $O/attr_big/targets_ordered.jsonl)
vdone() { cat $O/attr_big/verify_s0.jsonl $O/attr_big/verify_s1.jsonl 2>/dev/null | wc -l; }
until [ "$(vdone)" -ge "$N" ]; do
  srun --jobid=$(G1) --overlap -n1 -c4 --mem=0 $P -u scripts/verify_arm_candidates.py attr_big 0 2 targets_ordered.jsonl >> $O/attr_big/verify_s0.log 2>&1 &
  srun --jobid=$(G2) --overlap -n1 -c4 --mem=0 $P -u scripts/verify_arm_candidates.py attr_big 1 2 targets_ordered.jsonl >> $O/attr_big/verify_s1.log 2>&1 &
  wait; sleep 30
done
log "big: verified $(vdone)/$N states"
$P - <<'PY' >> $M
import json, random, collections
O="reports/alfworld/self_evolve7b"; T="reports/alfworld/self_evolve7b/r1_big.jsonl"
V=[json.loads(l) for s in (0,1) for l in open(f"{O}/attr_big/verify_s{s}.jsonl")]
TG={(json.loads(l)["id"],json.loads(l)["t"]):json.loads(l) for l in open(f"{O}/attr_big/targets_ordered.jsonl")}
rng=random.Random(0); pairs=[]; plac=[]; ok=collections.Counter()
for v in V:
    fx=[c for c in v["cands"] if v["won"][c] and not v["won"][v["a_minus"]]]
    ok[(TG[(v["id"],v["t"])]["proxy_ok"], bool(fx))]+=1
    if fx:
        pairs.append({"kind":"dpo","src":T,"id":v["id"],"t":v["t"],"chosen":fx[0],"rejected":v["a_minus"]})
        adm=[c for c in TG[(v["id"],v["t"])]["admissible"] if c!=v["a_minus"]]
        plac.append({"kind":"dpo","src":T,"id":v["id"],"t":v["t"],"chosen":rng.choice(adm),"rejected":v["a_minus"]})
import os
for n,r in (("big_verified",pairs),("big_placebo",plac)):
    os.makedirs(f"{O}/{n}",exist_ok=True); open(f"{O}/{n}/dpo_pairs.jsonl","w").writelines(json.dumps(x)+"\n" for x in r)
pv=ok[(True,True)]/max(1,ok[(True,True)]+ok[(True,False)]); pn=ok[(False,True)]/max(1,ok[(False,True)]+ok[(False,False)])
print(f"=== big: {len(V)} states verified, REAL fixes {len(pairs)} ({len(pairs)/len(V):.0%}); real-fix rate passing veto {pv:.0%} vs vetoed {pn:.0%}")
PY
export SE_TRAJ=$O/r1_big.jsonl
srun --jobid=$(G1) --overlap -n1 -c4 --mem=0 $P -u scripts/self_evolve.py refs --arms big_verified,big_placebo > $O/refs_big.log 2>&1
run() { local a=$1 sd=$2 g=$3; local D=$O/$a; local j; j=$($g)
  srun --jobid=$j --overlap -n1 -c4 --mem=0 $P -u scripts/alfworld_opd_plus.py train --data $D/dpo_pairs.jsonl --student Qwen/Qwen2.5-7B-Instruct --adapter "" --lr 1e-6 --epochs 1 --beta 0.1 --seed $sd --out $D/m_s$sd > $D/train_s$sd.log 2>&1
  for sp in valid_unseen valid_seen; do j=$($g); srun --jobid=$j --overlap -n1 -c4 --mem=0 $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-7B-Instruct --adapter $D/m_s$sd/adapter --split $sp --budget 50 --temperature 0 --out $D/eval_${sp#valid_}_s$sd.jsonl --tag big_${a}_$sd >> $D/train_s$sd.log 2>&1; done
  echo "=== $(date '+%F %T') big $a seed$sd: $(wc -l < $D/dpo_pairs.jsonl) pairs | unseen $(grep -c '"won": true' $D/eval_unseen_s$sd.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen_s$sd.jsonl)/140 (zero-shot 60, 50)" >> $M; }
(run big_verified 0 G1; run big_placebo 1 G1) & (run big_placebo 0 G2; run big_verified 1 G2) & wait
log "BIG ALL DONE"
