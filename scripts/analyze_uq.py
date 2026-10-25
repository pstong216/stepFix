"""Analyse attributor and student uncertainty on attr_nofb's round-2 failures (outputs of alfworld_uq.py).
Planner labels (t_first_regress) are used only as the ruler."""
import json, math, os, statistics as st
from pathlib import Path
R = Path(__file__).resolve().parent.parent
U = R / "reports/alfworld/uq"; A = R / "reports/alfworld/opd/student3b_low100"


def load(p):
    p = Path(p); return [json.loads(l) for l in open(p)] if p.exists() else []


def auc(xs, ys):
    pos = [x for x, y in zip(xs, ys) if y]; neg = [x for x, y in zip(xs, ys) if not y]
    if not pos or not neg: return float("nan")
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def spearman(a, b):
    def rk(x):
        o = sorted(range(len(x)), key=lambda i: x[i]); r = [0] * len(x)
        for k, i in enumerate(o): r[i] = k
        return r
    ra, rb = rk(a), rk(b); n = len(a)
    if n < 3: return float("nan")
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb)); va = sum((x - ma) ** 2 for x in ra); vb = sum((y - mb) ** 2 for y in rb)
    return cov / math.sqrt(va * vb) if va and vb else float("nan")


lab = {r["id"]: r.get("t_first_regress") for r in load(R / "reports/alfworld/progress_student3b_low100_round2_attr_nofb.jsonl")}
stu = {r["id"]: r["steps"] for r in load(U / "student_r2_attr_nofb.jsonl")}
atts = {n: {r["id"]: r for r in load(U / f"attributor_{n}_r2_attr_nofb.jsonl")} for n in ("stale", "verify", "pl450", "v1")}
ids = [i for i in lab if lab[i] is not None]


def agree(pred, name, sub=None):
    sub = [i for i in (sub or ids) if i in pred]; n = len(sub)
    ex = sum(pred[i] == lab[i] for i in sub) / n * 100; w2 = sum(abs(pred[i] - lab[i]) <= 2 for i in sub) / n * 100
    print(f"  {name:48s} n={n:3d} exact {ex:5.1f}%  within2 {w2:5.1f}%")


print("== 1. attributor confidence vs agreement with the planner")
for n, d in atts.items():
    sub = [i for i in ids if i in d]
    if not sub: print(f"  {n}: not available"); continue
    y_ex = [d[i]["argmax_t"] == lab[i] for i in sub]; y_w2 = [abs(d[i]["argmax_t"] - lab[i]) <= 2 for i in sub]
    pm = [d[i]["p_max"] for i in sub]; ne = [-d[i]["entropy"] for i in sub]
    ptrue = [d[i]["p"][lab[i]] if lab[i] < d[i]["n"] else 0.0 for i in sub]
    print(f"  {n:7s} n={len(sub)} argmax exact {sum(y_ex)/len(sub)*100:5.1f}% | median p_max {st.median(pm):.2f} | median mass on planner step {st.median(ptrue):.3f}"
          f" | AUC(p_max->exact) {auc(pm, y_ex):.3f} AUC(p_max->within2) {auc(pm, y_w2):.3f} AUC(-entropy->exact) {auc(ne, y_ex):.3f}")
    q = sorted(sub, key=lambda i: -d[i]["p_max"]); h = len(q) // 3
    for lbl, s in (("most confident third", q[:h]), ("least confident third", q[-h:])):
        print(f"      {lbl:22s} exact {sum(d[i]['argmax_t']==lab[i] for i in s)/len(s)*100:5.1f}%  within2 {sum(abs(d[i]['argmax_t']-lab[i])<=2 for i in s)/len(s)*100:5.1f}%")

if stu:
    print("\n== 2. student uncertainty as an attributor (vs planner)")
    def first(i, cond):
        return next((s["t"] for s in stu[i] if cond(s)), stu[i][-1]["t"])
    ptr = {
        "argmin p(chosen action)": {i: min(stu[i], key=lambda s: s["p_chosen"])["t"] for i in ids if i in stu},
        "argmax student entropy": {i: max(stu[i], key=lambda s: s["entropy"])["t"] for i in ids if i in stu},
        "first step where student did NOT pick its top action": {i: first(i, lambda s: s["rank_chosen"] > 1) for i in ids if i in stu},
        "first step with p(chosen) < 0.5": {i: first(i, lambda s: s["p_chosen"] < 0.5) for i in ids if i in stu},
        "first step with p(chosen) < 0.2": {i: first(i, lambda s: s["p_chosen"] < 0.2) for i in ids if i in stu},
    }
    for k, v in ptr.items(): agree(v, k)
    for n, d in atts.items():
        if d: agree({i: d[i]["argmax_t"] for i in ids if i in d}, f"Model A ({n}) argmax")
    t = [lab[i] for i in ids]
    for c in (1, 3): agree({i: c for i in ids}, f"constant step {c}")

    print("\n   student state AT the planner's harmful step vs the other steps of the same failure")
    at, other, at_rank1, oth_rank1 = [], [], [], []
    for i in ids:
        if i not in stu or lab[i] >= len(stu[i]): continue
        for s in stu[i]:
            (at if s["t"] == lab[i] else other).append(s["p_chosen"]); (at_rank1 if s["t"] == lab[i] else oth_rank1).append(s["rank_chosen"] == 1)
    print(f"   p(chosen): harmful step median {st.median(at):.3f} vs other steps {st.median(other):.3f} | chosen = student's top action: {sum(at_rank1)/len(at_rank1)*100:.0f}% vs {sum(oth_rank1)/len(oth_rank1)*100:.0f}%")
    print(f"   AUC(1 - p(chosen) separates the harmful step from other steps) {auc([1-x for x in at]+[1-x for x in other], [1]*len(at)+[0]*len(other)):.3f}")

    print("\n== 3. attributor vs student")
    for n, d in atts.items():
        sub = [i for i in ids if i in d and i in stu and len(stu[i]) == d[i]["n"]]
        if not sub: continue
        pct, rho = [], []
        for i in sub:
            unc = [1 - s["p_chosen"] for s in stu[i]]; am = d[i]["argmax_t"]
            pct.append(sum(1 for u in unc if u < unc[am]) / len(unc))
            rho.append(spearman(d[i]["p"], unc))
        rho = [x for x in rho if x == x]
        conf = [d[i]["p_max"] for i in sub]; su = [1 - stu[i][d[i]["argmax_t"]]["p_chosen"] for i in sub]
        print(f"  {n:7s} n={len(sub)} | student-uncertainty percentile of the blamed step: median {st.median(pct):.2f} (0.5 = typical)"
              f" | per-failure Spearman(attributor p_k, student 1-p_chosen_k): median {st.median(rho):.3f}"
              f" | Spearman(attributor p_max, student uncertainty at blamed step) {spearman(conf, su):.3f}")

    print("\n== 4. predicting verify success of the state that was actually trained on (round 2)")
    for arm, an in (("attr_nofb", "stale"), ("attrvl_nofb", "verify")):
        ex = {(r["id"], r["t"]) for r in load(A / arm / "round2/examples.jsonl")}
        tried = ex | {(r["id"], r["t"]) for r in load(A / arm / "round2/skipped.jsonl")}
        rows = [(i, t) for (i, t) in tried if i in stu and t < len(stu[i])]
        y = [(i, t) in ex for (i, t) in rows]
        feats = {"student 1-p(chosen)": [1 - stu[i][t]["p_chosen"] for i, t in rows],
                 "student entropy": [stu[i][t]["entropy"] for i, t in rows],
                 "student margin": [stu[i][t]["margin"] for i, t in rows],
                 "chosen was not student's top": [stu[i][t]["rank_chosen"] > 1 for i, t in rows]}
        d = atts.get(an) or {}
        if d:
            feats[f"attributor ({an}) p_max"] = [d[i]["p_max"] if i in d else 0 for i, t in rows]
            feats[f"attributor ({an}) mass at this step"] = [d[i]["p"][t] if i in d and t < d[i]["n"] else 0 for i, t in rows]
        print(f"  {arm}: {len(rows)} states, verified {sum(y)} ({sum(y)/len(rows)*100:.0f}%)")
        for k, v in feats.items(): print(f"      {k:40s} AUC {auc(v, y):.3f}")
