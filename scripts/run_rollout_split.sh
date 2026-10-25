#!/bin/bash
# User-approved (Oct 8 ~16:40): "rollout-split" mechanism ablation on R1 of the final recipe (why does one-step evidence beat rollout verification?).
# Pool = all one-step-approved corrections at the final recipe's R1 states (rule env_filter on selfattr_a3bfinal/r1: same Model-A states, same
# hindsight-teacher candidates, same generic one-step evidence as our R1), ranked by the env_top key (env score, then hindsight prob).
# Rollout labels = selfattr_rollsub_a3b/r1 (full student continuation from the corrected state, fixed SFT policy; a state absent from
# examples.jsonl had no winning candidate). Two arms, K = 82 each (= our R1 budget), each + the SAME 100 replay steps as our R1, same trainer/seed,
# from the SFT student, 774 eval:
#   pass82 : top-82 (by evidence) among corrections that PASS rollout  -> vs rollout 20% R1 (483, random 82 among rollout winners): does ranking matter?
#   rej82  : top-82 (by evidence) among corrections REJECTED by rollout -> vs student (416): are rollout-rejected corrections still useful?
source scripts/lib_baselines.sh; export EVIDENCE=generic
X=$SA/a3b_rolloutsplit; F=$SA/selfattr_a3bfinal/r1; RV=$SA/selfattr_rollsub_a3b/r1; REP=$SA/a3bfinal_k20/r1/replay.jsonl; mkdir -p $X/pool $X/pass82 $X/rej82
[ -s $X/pool/env_filter/train.jsonl ] || SE_DIR=$F SE_TRAJ=$T1 $P scripts/proxy_arms_se3b.py --arm self --rule env_filter --out $X/pool/env_filter | tail -n 1
[ -s $X/pool/env_top_all/train.jsonl ] || SE_DIR=$F SE_TRAJ=$T1 $P scripts/proxy_arms_se3b.py --arm self --rule env_top --k 100000 --out $X/pool/env_top_all | tail -n 1
[ -s $X/rej82/targets.jsonl ] || $P - <<PY | tee -a $SA/queue.log
import json, glob
X="$X"; L=lambda p: [json.loads(l) for l in open(p)]
approved={(r["id"], r["t"], r["cand"]) for r in L(f"{X}/pool/env_filter/train.jsonl")}
ranked=[r for r in L(f"{X}/pool/env_top_all/train.jsonl") if (r["id"], r["t"], r["cand"]) in approved]
roll={}
for f in glob.glob("$RV/rollout_s*/examples.jsonl"):
    for l in open(f): r=json.loads(l); roll[(r["id"], r["t"])]=r.get("alternatives") or {}
P=[r for r in ranked if roll.get((r["id"], r["t"]), {}).get(r["cand"])]; Rj=[r for r in ranked if not roll.get((r["id"], r["t"]), {}).get(r["cand"])]
ours={(r["id"], r["t"]) for r in L("$SA/a3bfinal_k20/r1/targets.jsonl")}; top82={(r["id"], r["t"]) for r in ranked[:82]}
for name, S in (("pass82", P[:82]), ("rej82", Rj[:82])):
    open(f"{X}/{name}/targets.jsonl","w").writelines(json.dumps(r)+"\n" for r in S)
print(f"=== rollout-split (R1): one-step-approved {len(ranked)} (pool check: {len(approved)}), rollout-pass {len(P)}, rejected {len(Rj)}; "
      f"top-82 of the ranked pool == our R1 targets: {len(top82 & ours)}/82; pass82 overlap with ours {len({(r['id'],r['t']) for r in P[:82]} & ours)}, rej82 overlap {len({(r['id'],r['t']) for r in Rj[:82]} & ours)}")
PY
for A in pass82 rej82; do ( train1 $X/$A $STU "$X/$A/targets.jsonl:1.0 $REP:1.0"; [ -s $E/eval_heldout_rsplit_$A.jsonl ] || eval774 $X/$A rsplit_$A; log "rollout-split $A DONE (rsplit_$A)" ) & sleep 60; done; wait
log "rollout-split R1 ablation DONE (compare: student 416, rollout 20% R1 483, rand20 R1 505, ours R1 537)"
