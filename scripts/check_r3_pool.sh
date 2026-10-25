#!/bin/bash
# User-approved (Oct 8 16:00): check the final recipe's R3 (a3bfinal_k20/r3, 774 = 591) after the teacher-shard resume bug.
# R3's selection pool had teacher candidates for only 171 of the 249 Model-A states. Here (in a separate dir, originals untouched):
#  (0) control: re-run the env_top filter on the ORIGINAL pool -> must reproduce r3/targets_env_top exactly;
#  (1) fill the 78 missing states with the same teacher (R2 adapter, same script, now fixed) + 6 already-covered states as a determinism check;
#  (2) one-step evidence for the new candidates only (cached results reused); (3) env_top with the same K = 50 over all 249 states;
#  (4) compare with the 50 targets actually used. If they differ: retrain R3 (same start adapter, same replay, same trainer/seed) -> r3_fixpool, 774 eval.
source scripts/lib_baselines.sh; export EVIDENCE=generic
D0=$SA/a3bfinal_k20/r3; C=$SA/a3bfinal_k20/r3_poolcheck; AD=$SA/a3bfinal_k20/r2/model/adapter; mkdir -p $C/self $C/proxy $C/control/self $C/control/proxy
for x in $C $C/control; do cp $D0/self/steps.jsonl $x/self/; cp $D0/ids_fail.jsonl $D0/pointer.jsonl $D0/pointer.raw.jsonl $D0/diag_modelA.raw.jsonl $x/; [ -s $x/proxy/onestep.jsonl ] || cp $D0/proxy/onestep.jsonl $x/proxy/; done
cp $D0/self/targets.jsonl $C/control/self/targets.jsonl
[ -s $C/control/targets_env_top/train.jsonl ] || SE_DIR=$C/control SE_TRAJ=$D0/traj.jsonl $P scripts/proxy_arms_se3b.py --arm self --rule env_top --k 50 --out $C/control/targets_env_top | tail -n 2
$P - <<PY
import json
C="$C"; D0="$D0"; steps=[(json.loads(l)["id"], json.loads(l)["steps"][0]) for l in open(f"{D0}/self/steps.jsonl")]
have={}
for l in open(f"{D0}/self/targets.jsonl"):
    r=json.loads(l); have.setdefault((r["id"], r["t"]), l)
miss=[s for s in steps if s not in have]; ctrl=[s for s in steps if s in have][:6]
open(f"{C}/self/steps_missing.jsonl","w").writelines(json.dumps({"id":i,"steps":[t],"mode":None})+"\n" for i,t in miss)
open(f"{C}/self/steps_control.jsonl","w").writelines(json.dumps({"id":i,"steps":[t],"mode":None})+"\n" for i,t in ctrl)
open(f"{C}/self/targets_have.jsonl","w").writelines(have[s] for s in steps if s in have)
a=[json.loads(l) for l in open(f"{D0}/targets_env_top/train.jsonl")]; b=[json.loads(l) for l in open(f"{C}/control/targets_env_top/train.jsonl")]
print(f"control: original pool reproduces the original selection: {a == b} ({len(a)} vs {len(b)} rows)")
print(f"states {len(steps)}, covered {len(have)}, missing {len(miss)}")
PY
for w in missing control; do until grep -q "DONE informed_teacher" $C/teacher_$w.log 2>/dev/null; do $(S 12000) $P -u scripts/self_teacher_informed.py --traj $D0/traj.jsonl --steps $C/self/steps_$w.jsonl --diag $D0/diag_modelA.raw.jsonl --adapter $AD --out $C/self/targets_$w.jsonl >> $C/teacher_$w.log 2>&1 || sleep 30; done; done
$P - <<PY
import json, math
C="$C"; D0="$D0"; steps=[(json.loads(l)["id"], json.loads(l)["steps"][0]) for l in open(f"{D0}/self/steps.jsonl")]
rows={}
for f in ["targets_have", "targets_missing"]:
    for l in open(f"{C}/self/{f}.jsonl"): r=json.loads(l); rows.setdefault((r["id"], r["t"]), l)
open(f"{C}/self/targets.jsonl","w").writelines(rows[s] for s in steps if s in rows)
old={(json.loads(l)["id"], json.loads(l)["t"]): json.loads(l) for l in open(f"{C}/self/targets_have.jsonl")}
dmax=0; same_top=0; ctl=[json.loads(l) for l in open(f"{C}/self/targets_control.jsonl")]
for r in ctl:
    o=old[(r["id"], r["t"])]; dmax=max(dmax, max(abs(r["teacher_logprobs"][c]-o["teacher_logprobs"][c]) for c in r["teacher_logprobs"]))
    top=lambda T: [c for c in sorted(T["teacher_logprobs"], key=lambda c: -T["teacher_logprobs"][c]) if c != T["a_minus"]][:2]
    same_top+= top(r)==top(o)
print(f"determinism: {same_top}/{len(ctl)} control states give the same top-2 candidates, max |logprob diff| {dmax:.4f}")
print(f"pool now {sum(1 for s in steps if s in rows)}/{len(steps)} states")
PY
until grep -q "DONE onestep" $C/onestep.log 2>/dev/null; do SE_DIR=$C SE_TRAJ=$D0/traj.jsonl $P -u scripts/proxy_eval_se3b.py onestep --arms self --limit 400 >> $C/onestep.log 2>&1; sleep 3; done
[ -s $C/targets_env_top/train.jsonl ] || SE_DIR=$C SE_TRAJ=$D0/traj.jsonl $P scripts/proxy_arms_se3b.py --arm self --rule env_top --k 50 --out $C/targets_env_top | tail -n 2
$P - <<PY | tee -a $SA/queue.log
import json
a=[json.loads(l) for l in open("$D0/targets_env_top/train.jsonl")]; b=[json.loads(l) for l in open("$C/targets_env_top/train.jsonl")]
key=lambda r: (r["id"], r["t"], tuple(r["target"]))
A={key(r) for r in a}; B={key(r) for r in b}
print(f"=== R3 POOL CHECK (a3bfinal_k20/r3): full pool 249 states vs used pool 171; top-50 overlap {len(A & B)}/50, replaced {len(A - B)}")
open("$C/verdict.txt","w").write("SAME\n" if A == B else "DIFFERENT\n")
PY
if grep -q DIFFERENT $C/verdict.txt; then
  F=$SA/a3bfinal_k20/r3_fixpool; mkdir -p $F; log "a3bfinal_k20 r3_fixpool: retrain R3 on the full-pool top-50 (+ same 100 replay) from $AD"
  train1 $F $AD "$C/targets_env_top/train.jsonl:1.0 $D0/replay.jsonl:1.0"; [ -s $E/eval_heldout_a3bk20_r3fix.jsonl ] || eval774 $F a3bk20_r3fix
  log "a3bfinal_k20 r3_fixpool DONE (a3bk20_r3fix; original R3 = 591)"
else log "a3bfinal_k20 r3: full-pool selection identical to the one used -> 591 stands"; fi
