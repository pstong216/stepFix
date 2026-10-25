#!/bin/bash
# User (Oct 8 13:35): test the ALREADY-TRAINED models with the new student prompt v2 (budget + "only a purchase is scored" + actions
# used), no retraining. Models: R1 of the fixed-Model-A chain (reports/webshop/final_modelA/r1, trained with prompt v1) and, as the
# reference under the same prompt, the SFT student R0. test 500 + dev 200, greedy, budget 15.
export WEBSHOP_PROMPT=v2 WEBSHOP_BUDGET=15
source scripts/lib_baselines.sh; export WEBSHOP_SERVER=http://agd07:38300; W=reports/webshop; O=$W/eval700_p2; mkdir -p $O
ev() { local name=$1 AD=$2 split=$3 off=$4 out=$O/${1}_${3}_${4}.jsonl
  until grep -q "DONE collect" $out.log 2>/dev/null; do $(S 10000) $P -u scripts/webshop_agent.py collect --adapter $AD --split $split --offset $off --n 100 --budget 15 --temperature 0 --out $out --tag evp2_$name >> $out.log 2>&1 || sleep 60; done; }
for spec in "r1 $W/final_modelA/r1/model/adapter" "student $W/student100/final"; do set -- $spec
  for s in "test 0" "test 100" "test 200" "test 300" "test 400" "dev 0" "dev 100"; do ev $1 $2 $s & sleep 15; done
done; wait
$P - <<PY | tee -a $W/queue.log $SA/queue.log
import json, glob, collections
from scipy.stats import binomtest
def load(pat):
    R={}
    for f in glob.glob(pat):
        for l in open(f): r=json.loads(l); R[(r["split"], r["session"])]=r
    return R
def stat(R): n=len(R); s=sum(r["won"] for r in R.values()); return f"{s}/{n} success ({100*s/max(1,n):.1f}%), mean reward {sum(r['final_score'] for r in R.values())/100/max(1,n):.3f}, no purchase {sum(not any(st['action']=='click[buy now]' for st in r['steps']) for r in R.values())}"
A={"r1_v2": load("$O/r1_*.jsonl"), "student_v2": load("$O/student_*.jsonl"), "r1_v1": load("$W/eval700/r1_b15_*.jsonl"), "student_v1": load("$W/eval700/student_b15_*.jsonl")}
for k,v in A.items(): print(f"=== WEBSHOP EVAL700 {k}: {stat(v)}")
for x,y in [("r1_v2","r1_v1"),("r1_v2","student_v2"),("student_v2","student_v1"),("r1_v2","student_v1")]:
    a,b=A[x],A[y]; g=[k for k in a if k in b]; p=sum(a[k]["won"] and not b[k]["won"] for k in g); q=sum(b[k]["won"] and not a[k]["won"] for k in g)
    print(f"=== {x} vs {y}: {x}-only {p}, {y}-only {q}, p={binomtest(p,p+q).pvalue if p+q else 1:.3g}")
PY
