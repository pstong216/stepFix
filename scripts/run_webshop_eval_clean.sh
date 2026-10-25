#!/bin/bash
# Oct 8 15:25 (user-approved rerun): clean WebShop 700-session evaluation (test 500 + dev 200, greedy, budget 15) of the SFT student and the
# fixed-Model-A R1 model under prompt v1 (original) and v3 (one "remember to click[buy now]" system sentence), no training.
# Earlier eval700 / eval700_p2 / eval700_p3 runs were INVALID: all shards of a model shared one server worker id, so concurrent shards reset
# and stepped each other's sessions. Here every concurrent process has its own worker id (tag includes prompt, model, split and offset).
source scripts/lib_baselines.sh; export WEBSHOP_SERVER=${WEBSHOP_SERVER:-http://agd07:38300} WEBSHOP_BUDGET=15; W=reports/webshop; O=$W/eval700c
ev() { local pr=$1 name=$2 AD=$3 split=$4 off=$5 n=$6 out=$O/$1/${2}_${4}_${5}.jsonl; mkdir -p $O/$pr
  until grep -q "DONE collect" $out.log 2>/dev/null; do WEBSHOP_PROMPT=$pr $(S 10000) $P -u scripts/webshop_agent.py collect --adapter $AD --split $split --offset $off --n $n --budget 15 --temperature 0 --out $out --tag evc_${pr}_${name}_${split}${off} >> $out.log 2>&1 || sleep 60; done; }
for pr in v1 v3; do for spec in "student $W/student100/final" "r1 $W/final_modelA/r1/model/adapter"; do set -- $spec
  for s in "test 0 175" "test 175 175" "test 350 150" "dev 0 200"; do ev $pr $1 $2 $s & sleep 15; done
done; done; wait
$P - <<PY | tee -a $W/queue.log $SA/queue.log
import json, glob, re
from scipy.stats import binomtest
def instr(o):
    m=re.search(r"Instruction:\s*\[SEP\]\s*(.*?)\s*\[SEP\]", o); return m.group(1).strip().lower() if m else None
def load(pat):
    R={}; bad=0
    for f in glob.glob(pat):
        for l in open(f):
            r=json.loads(l); R[(r["split"], r["session"])]=r; g=r["goal"].lower().replace("instruction:","").strip()
            bad+=any(i and i!=g and not g.startswith(i[:60]) for i in [instr(s.get("next_obs","")) for s in r["steps"]])
    return R, bad
def stat(R): n=len(R); s=sum(r["won"] for r in R.values()); return f"{s}/{n} success ({100*s/max(1,n):.1f}%), mean reward {sum(r['final_score'] for r in R.values())/100/max(1,n):.3f}, no purchase {sum(not any(st['action']=='click[buy now]' for st in r['steps']) for r in R.values())}, mean steps {sum(len(r['steps']) for r in R.values())/max(1,n):.1f}"
A={}
for pr in ["v1","v3"]:
    for m in ["student","r1"]:
        R,bad=load(f"$O/{pr}/{m}_*.jsonl"); A[f"{m}_{pr}"]=R; print(f"=== WEBSHOP EVAL700 CLEAN {m}_{pr}: {stat(R)} | contaminated sessions {bad}")
for x,y in [("r1_v1","student_v1"),("r1_v3","student_v3"),("student_v3","student_v1"),("r1_v3","r1_v1")]:
    a,b=A[x],A[y]; g=[k for k in a if k in b]; p=sum(a[k]["won"] and not b[k]["won"] for k in g); q=sum(b[k]["won"] and not a[k]["won"] for k in g)
    print(f"=== CLEAN {x} vs {y}: {x}-only {p}, {y}-only {q}, p={binomtest(p,p+q).pvalue if p+q else 1:.3g}")
PY
