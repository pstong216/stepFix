#!/bin/bash
# 3B attributor (Qwen2.5-3B LoRA, SFT on Who&When Pro, agent-text-only) as the step rule, then the best filter
# (plain candidates + env_top86, no rollout).  Compare with the same filter on Model-A (7B) steps (172/169) and M5 steps (167).
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; SE=reports/alfworld/self_evolve3b; M=$SE/queue.log
T=reports/alfworld/student3b_low100_train_b50.jsonl; STU=reports/alfworld/student3b_low100/final; export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S="srun --jobid=2644179 --overlap -n1 -c4 --mem=0"; A3=reports/attributor_no_observations_keepids/final; PTR=reports/alfworld/attr3bnoobs_student3b_low100_train_b50.jsonl
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
free_mb() { $S nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1; }
guard() { until [ "$(free_mb)" -ge "$1" ] 2>/dev/null; do sleep 60; done; }
mkdir -p $SE/attr3b_plain
[ -s $PTR ] || { guard 12000; $S $P -u scripts/alfworld_attribute.py --attributor $A3 --variant no_observations --release ${T%.jsonl}.release.jsonl --out $PTR > $SE/attr3b_plain/attribute.log 2>&1; }
$P -u scripts/alfworld_attr_opd.py choose --traj $T --arm attr --attr $PTR --ids_from $SE/ids319.jsonl --max_failures 1000 --out $SE/attr3b_plain/steps.jsonl > $SE/attr3b_plain/choose.log 2>&1
$P - <<'PY' | tee -a $M
import json, statistics
SE="reports/alfworld/self_evolve3b"; A="reports/alfworld/opd/student3b_low100"
d={json.loads(l)["id"]:json.loads(l)["steps"][0] for l in open(f"{SE}/attr3b_plain/steps.jsonl")}
attr={json.loads(l)["id"]:json.loads(l)["steps"][0] for l in open(f"{A}/attr_nofb/steps.jsonl")}
m5={json.loads(l)["id"]:json.loads(l)["steps"][0] for l in open(f"{A}/progress_nofb/steps.jsonl")}
ids=[i for i in d if i in attr]; ts=[d[i] for i in ids]
print(f"=== 3B attributor pointer ({len(ids)} states): mean t {statistics.mean(ts):.1f} median {statistics.median(ts):.0f} | t0 {sum(t==0 for t in ts)/len(ts):.2f} t<=2 {sum(t<=2 for t in ts)/len(ts):.2f} | exact/within-1 vs Model A {sum(d[i]==attr[i] for i in ids)/len(ids):.2f}/{sum(abs(d[i]-attr[i])<=1 for i in ids)/len(ids):.2f} | vs M5 {sum(d[i]==m5[i] for i in ids)/len(ids):.2f}/{sum(abs(d[i]-m5[i])<=1 for i in ids)/len(ids):.2f}")
PY
export SE_DIR=$SE SE_TRAJ=$T SE_MODEL=Qwen/Qwen2.5-3B-Instruct SE_ADAPTER=$PWD/$STU
guard 10000; $S $P -u scripts/self_evolve.py hindsight_steps --steps $SE/attr3b_plain/steps.jsonl > $SE/attr3b_plain/h.log 2>&1
$P - <<'PY'
import json
SE="reports/alfworld/self_evolve3b"; rows=[json.loads(l) for l in open(f"{SE}/attr3b_plain/targets.jsonl")]
with open(f"{SE}/attr3b_plain/targets.jsonl","w") as f:
    for r in rows: r["hind_logprobs"]=r["teacher_logprobs"]; r["teacher_logprobs"]=r["plain_logprobs"]; r["mode"]="plain"; f.write(json.dumps(r)+"\n")
print("plain targets", len(rows))
PY
until grep -q "DONE onestep" $SE/proxy/onestep5.log 2>/dev/null; do srun --jobid=2239109 --overlap -n1 -c2 --mem=20G $P -u scripts/proxy_eval_se3b.py onestep --arms attr3b_plain --limit 250 >> $SE/proxy/onestep5.log 2>&1; sleep 5; done
D=$SE/proxy_arms/attr3b_plain_env_top; $P scripts/proxy_arms_se3b.py --arm attr3b_plain --rule env_top --k 86 | tee -a $M
guard 24000; $S $P -u scripts/alfworld_opd_plus.py train --data $D/train.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter $PWD/$STU --lr 1e-5 --epochs 1 --batch 8 --out $D/model > $D/train.log 2>&1 || { log "attr3b_plain/env_top: TRAIN FAILED"; exit 1; }
guard 10000; for sp in valid_unseen valid_seen; do $S $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $D/model/adapter --split $sp --budget 50 --temperature 0 --out $D/eval_${sp#valid_}.jsonl --tag px_attr3b >> $D/train.log 2>&1; done
log "3B proxy attr3b_plain/env_top (3B attributor steps, plain candidates, env_top86, no rollout): 86 targets | unseen $(grep -c '"won": true' $D/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen.jsonl)/140 (Model-A steps 78/94 & 74/95; M5 steps 72/95; student 56/134, 80/140)"
