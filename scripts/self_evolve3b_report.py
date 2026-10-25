"""Read-out for the 3B self-evolution round 1 (reports/alfworld/self_evolve3b): per arm the number of verified corrections,
unseen/seen/pooled wins on the 274 eval games, and exact-binomial McNemar vs the SFT student, vs the external-teacher arm
(attr_nofb, 3B teacher + Model A pointers), vs the no-hindsight self arm (self_nofb) and vs the placebo. Appends the flip
probe table when flip.log exists. Usage: python scripts/self_evolve3b_report.py [--markdown]"""
import json, os, re, sys
from scipy.stats import binomtest
O = "reports/alfworld/self_evolve3b"; A = "reports/alfworld/opd/student3b_low100"; E = "reports/alfworld/ext"


def load(p): return {json.loads(l)["game_file"]: bool(json.loads(l).get("won")) for l in open(p)} if os.path.exists(p) else {}
def pooled(u, s): d = load(u); d.update(load(s)); return d
def mc(d, o, g):
    ab = sum(d[x] and not o[x] for x in g); ba = sum(o[x] and not d[x] for x in g)
    return f"{ab:3d}/{ba:<3d} p={binomtest(ab, ab + ba).pvalue:.2g}" if ab + ba else "   --      "


REF = {"student": pooled("reports/alfworld/eval_unseen_student3b_low100.jsonl", f"{E}/eval_seen_student_low100.jsonl"),
       "ext-attr": pooled(f"{A}/attr_nofb/eval_unseen_round1.jsonl", f"{E}/eval_seen_opd_attr_nofb.jsonl"),
       "self_nofb": pooled(f"{A}/self_nofb/eval_unseen_round1.jsonl", f"{E}/eval_seen_opd_self_nofb.jsonl"),
       "placebo": pooled(f"{O}/placebo/eval_unseen.jsonl", f"{O}/placebo/eval_seen.jsonl")}
ARMS = [("attr", "Model A pointer"), ("m5", "M5 progress pointer"), ("selfattr", "student self-attribution"),
        ("entropy", "max-entropy step"), ("rand_early", "random step in first 3"), ("placebo", "random target @attr states")]
md = "--markdown" in sys.argv
hdr = f"{'arm':12s} {'step rule':26s} {'verified':>8s} {'unseen':>7s} {'seen':>6s} {'pooled':>7s} | vs student      | vs ext-attr     | vs self_nofb    | vs placebo"
print(hdr); print("-" * len(hdr))
for name, d in [("student", REF["student"]), ("ext-attr (attr_nofb)", REF["ext-attr"]), ("self_nofb", REF["self_nofb"])]:
    g = sorted(d); un = [x for x in g if "valid_unseen" in x]
    if len(g) < 274: print(f"{name:39s} incomplete ({len(g)})"); continue
    print(f"{name:39s} {'':>8s} {sum(d[x] for x in un):4d}/134 {sum(d[x] for x in g if x not in un):3d}/140 {sum(d[x] for x in g)/len(g)*100:6.1f}% | "
          + " | ".join(mc(d, REF[r], g) for r in ("student", "ext-attr", "self_nofb", "placebo") if len(REF[r]) >= 274))
print()
for n, rule in ARMS:
    d = pooled(f"{O}/{n}/eval_unseen.jsonl", f"{O}/{n}/eval_seen.jsonl")
    ex = f"{O}/{n}/round1/examples.jsonl"; nv = sum(1 for _ in open(ex)) if os.path.exists(ex) else (sum(1 for _ in open(f"{O}/placebo/train.jsonl")) if n == "placebo" and os.path.exists(f"{O}/placebo/train.jsonl") else 0)
    g = sorted(d); un = [x for x in g if "valid_unseen" in x]
    if len(g) < 274: print(f"{n:12s} {rule:26s} {nv:8d} incomplete ({len(g)} games)"); continue
    print(f"{n:12s} {rule:26s} {nv:8d} {sum(d[x] for x in un):4d}/134 {sum(d[x] for x in g if x not in un):3d}/140 {sum(d[x] for x in g)/len(g)*100:6.1f}% | "
          + " | ".join(mc(d, REF[r], g) if len(REF[r]) >= 274 else "   (pending)   " for r in ("student", "ext-attr", "self_nofb", "placebo")))
import glob
REF["self-attr-verified"] = pooled(f"{O}/attr/eval_unseen.jsonl", f"{O}/attr/eval_seen.jsonl")
px = sorted(glob.glob(f"{O}/proxy_arms/*/eval_seen.jsonl"))
if px:
    print(f"\nproxy-selected arms (no rollout; hindsight self-teacher candidates unless *_plain):")
    print(f"{'arm':22s} {'targets':>7s} {'pass%':>6s} {'unseen':>7s} {'seen':>6s} {'pooled':>7s} | vs student      | vs placebo      | vs self-verified| vs self_nofb    | vs ext-attr")
    for p in px:
        n = os.path.basename(os.path.dirname(p)); D = os.path.dirname(p); d = pooled(f"{D}/eval_unseen.jsonl", p)
        g = sorted(d); un = [x for x in g if "valid_unseen" in x]
        if len(g) < 274: print(f"{n:22s} incomplete ({len(g)})"); continue
        st = json.load(open(f"{D}/stats.json")) if os.path.exists(f"{D}/stats.json") else {}
        nt = sum(1 for _ in open(f"{D}/train.jsonl")); ps = f"{st['rollout_won'] / max(1, st['n_targets']) * 100:5.0f}%" if st else "  n/a "
        print(f"{n:22s} {nt:7d} {ps:>6s} {sum(d[x] for x in un):4d}/134 {sum(d[x] for x in g if x not in un):3d}/140 {sum(d[x] for x in g) / len(g) * 100:6.1f}% | "
              + " | ".join(mc(d, REF[r], g) if len(REF[r]) >= 274 else "   (pending)   " for r in ("student", "placebo", "self-attr-verified", "self_nofb", "ext-attr")))
if os.path.exists(f"{O}/flip.log"):
    print("\nflip probe (150 student states, argmax flips vs SFT student; KL(base||trained)):")
    for l in open(f"{O}/flip.log"):
        if "argmax flips" in l: print("  " + l.rstrip())
