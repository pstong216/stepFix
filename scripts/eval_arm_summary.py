"""Pooled greedy score, retention, and McNemar vs a reference, for arms given as name=unseen,seen,heldout file triples."""
import json, os, sys
from scipy.stats import binomtest
E = "reports/alfworld/ext"; A = "reports/alfworld/opd/student3b_low100"
def load(p): return {json.loads(l)["game_file"]: bool(json.loads(l).get("won")) for l in open(p)} if os.path.exists(p) else {}
def pooled(u, s, h): d = {}; [d.update(load(p)) for p in (u, s, h)]; return d
BASE = pooled("reports/alfworld/eval_unseen_student3b_low100.jsonl", f"{E}/eval_seen_student_low100.jsonl", f"{E}/eval_heldout_student_low100.jsonl")
REF = pooled(f"{A}/attr_nofb/eval_unseen_round1.jsonl", f"{E}/eval_seen_opd_attr_nofb.jsonl", f"{E}/eval_heldout_opd_attr_nofb.jsonl")
def report(name, d):
    g = [x for x in BASE if x in d]
    if len(g) < 770: print(f"{name:28s} incomplete ({len(g)} games)"); return
    won = [x for x in g if BASE[x]]; kept = sum(d[x] for x in won)
    def mc(o):
        ab = sum(d[x] and not o[x] for x in g); ba = sum(o[x] and not d[x] for x in g); return f"{ab}/{ba} p={binomtest(ab, ab + ba).pvalue:.2g}" if ab + ba else "--"
    un = [x for x in g if "valid_unseen" in x]
    print(f"{name:28s} unseen {sum(d[x] for x in un)/len(un)*100:5.1f}% pooled {sum(d[x] for x in g)/len(g)*100:5.1f}% | kept {kept/len(won)*100:5.1f}% | vs student {mc(BASE)} | vs attr_nofb {mc(REF)}")
report("student", BASE); report("attr_nofb (online verify)", REF)
for spec in sys.argv[1:]:
    name, files = spec.split("="); report(name, pooled(*files.split(",")))
