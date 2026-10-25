#!/bin/bash
# WebShop larger evaluation (Oct 8): test 500 (sessions 0-499) + dev 200, greedy, budget 15 (standard) and 20, for the SFT student and the
# fixed-Model-A R1 model; later rounds can be added with the same function. Shards of 100 sessions on the GPU holds.
source scripts/lib_baselines.sh; export WEBSHOP_SERVER=http://agd07:38300; W=reports/webshop; O=$W/eval700; mkdir -p $O
ev() { local name=$1 AD=$2 B=$3 split=$4 off=$5 n=$6 out=$O/${1}_b${3}_${4}_${5}.jsonl
  until grep -q "DONE collect" $out.log 2>/dev/null; do $(S 10000) $P -u scripts/webshop_agent.py collect --adapter $AD --split $split --offset $off --n $n --budget $B --temperature 0 --out $out --tag ev_${name}_b${B} >> $out.log 2>&1 || sleep 60; done; }
for spec in "student $W/student100/final" "r1 $W/final_modelA/r1/model/adapter"; do set -- $spec
  for B in 15; do for off in 0 100 200 300 400; do ev $1 $2 $B test $off 100 & sleep 20; done; for off in 0 100; do ev $1 $2 $B dev $off 100 & sleep 20; done; done
done; wait
$P - <<PY | tee -a $W/queue.log $SA/queue.log
import json, glob, collections
R=collections.defaultdict(dict)
for f in glob.glob("$O/*_b*_*.jsonl"):
    name, b = f.split("/")[-1].split("_")[0], f.split("_b")[-1].split("_")[0]
    for l in open(f): r=json.loads(l); R[(name,b)][(r["split"], r["session"])]=r
for k,v in sorted(R.items()):
    n=len(v); s=sum(r["won"] for r in v.values()); m=sum(r["final_score"] for r in v.values())/100/max(1,n)
    nb=sum(not any(st["action"]=="click[buy now]" for st in r["steps"]) for r in v.values())
    print(f"=== WEBSHOP EVAL700 {k[0]} budget {k[1]}: {s}/{n} success ({s/max(1,n):.1%}), mean reward {m:.3f}, no purchase {nb}")
PY
