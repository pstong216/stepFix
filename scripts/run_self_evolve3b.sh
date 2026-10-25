#!/bin/bash
# 3B self-evolution, round 1: the SFT'd 3B student is its own teacher (sees the full failed log). Same 319 failures and the
# same verified-only OPD recipe as the external-teacher arms (attr_nofb 64.3% pooled); only the teacher (and step rule) change.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; O=reports/alfworld/self_evolve3b; M=$O/queue.log
A=reports/alfworld/opd/student3b_low100; T=reports/alfworld/student3b_low100_train_b50.jsonl; STU=reports/alfworld/student3b_low100/final
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SE_DIR=$O SE_TRAJ=$T SE_MODEL=Qwen/Qwen2.5-3B-Instruct SE_ADAPTER=$PWD/$STU
G1=2543688; G2=2543689; S1="srun --jobid=$G1 --overlap -n1 -c4 --mem=0"; S2="srun --jobid=$G2 --overlap -n1 -c4 --mem=0"
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
cp $A/attr_nofb/steps.jsonl $O/ids319.jsonl
$P - <<'PY'
import json
O="reports/alfworld/self_evolve3b"; ids=[json.loads(l)["id"] for l in open(f"{O}/ids319.jsonl")]
open(f"{O}/ids319.jsonl","w").writelines(json.dumps({"id":i})+"\n" for i in ids)
open(f"{O}/ids_a.jsonl","w").writelines(json.dumps({"id":i})+"\n" for i in ids[0::2]); open(f"{O}/ids_b.jsonl","w").writelines(json.dumps({"id":i})+"\n" for i in ids[1::2])
PY
log "3B self-evolve: per-step entropy (319 failures) + self-attribution"
$S1 $P -u scripts/student_entropy_all_steps.py $O/ids_a.jsonl $T $O/entropy_a.jsonl > $O/entropy_a.log 2>&1 &
$S2 $P -u scripts/student_entropy_all_steps.py $O/ids_b.jsonl $T $O/entropy_b.jsonl > $O/entropy_b.log 2>&1 &
wait
$S1 $P -u scripts/self_evolve.py selfattr --ids $O/ids319.jsonl > $O/selfattr.log 2>&1
$P - <<'PY'
import json, random, os
O="reports/alfworld/self_evolve3b"; A="reports/alfworld/opd/student3b_low100"
E={json.loads(l)["id"]:json.loads(l) for l in open("reports/alfworld/student3b_low100_train_b50.jsonl")}
H={}
for f in ("entropy_a.jsonl","entropy_b.jsonl"):
    for l in open(f"{O}/{f}"): r=json.loads(l); H[r["id"]]=r["entropy"]
ids=[json.loads(l)["id"] for l in open(f"{O}/ids319.jsonl")]
sa={json.loads(l)["id"]:json.loads(l)["t_hat"] for l in open(f"{O}/selfattr.jsonl")}
attr={json.loads(l)["id"]:json.loads(l)["steps"][0] for l in open(f"{A}/attr_nofb/steps.jsonl")}
m5={json.loads(l)["id"]:json.loads(l)["steps"][0] for l in open(f"{A}/progress_nofb/steps.jsonl")}
rng=random.Random(0)
rules={"attr":lambda i:attr[i],"m5":lambda i:m5[i],"selfattr":lambda i:min(sa[i],len(E[i]["steps"])-1),
       "entropy":lambda i:max(range(len(H[i])),key=lambda k:H[i][k]),"rand_early":lambda i:rng.randrange(min(3,len(E[i]["steps"])))}
for n,f in rules.items():
    os.makedirs(f"{O}/{n}",exist_ok=True)
    open(f"{O}/{n}/steps.jsonl","w").writelines(json.dumps({"id":i,"steps":[f(i)],"mode":None})+"\n" for i in ids)
print("steps written")
PY
log "3B self-evolve: hindsight self-teacher targets"
for n in attr m5 selfattr; do $S1 $P -u scripts/self_evolve.py hindsight_steps --steps $O/$n/steps.jsonl > $O/$n/h.log 2>&1; done &
for n in entropy rand_early; do $S2 $P -u scripts/self_evolve.py hindsight_steps --steps $O/$n/steps.jsonl > $O/$n/h.log 2>&1; done &
wait
arm() { local n=$1 S=$2; local D=$O/$n
  until grep -q "DONE distill" $D/run.log 2>/dev/null; do
    $S $P -u scripts/alfworld_attr_opd.py distill --traj $T --targets $D/targets.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter $STU --out $D/round1 --verify_k 2 --exclude_a_minus --max_examples_per_run 250 >> $D/run.log 2>&1 || sleep 30; done
  for sp in valid_unseen valid_seen; do $S $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $D/round1/adapter --split $sp --budget 50 --temperature 0 --out $D/eval_${sp#valid_}.jsonl --tag se3b_$n >> $D/run.log 2>&1; done
  echo "=== $(date '+%F %T') 3B self $n: verified $(wc -l < $D/round1/examples.jsonl)/319 | unseen $(grep -c '"won": true' $D/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen.jsonl)/140 (student 56/134, 80/140)" >> $M; }
log "3B self-evolve: verify + distill + eval (unseen + seen)"
(arm attr "$S1"; arm selfattr "$S1"; arm rand_early "$S1") & (arm m5 "$S2"; arm entropy "$S2") & wait
log "3B SELF-EVOLVE ROUND 1 DONE"
