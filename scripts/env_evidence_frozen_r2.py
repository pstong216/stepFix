"""Frozen round-1 filter applied to round 2 (stronger student, refreshed pointer): which signals stay valid?"""
import json, math, random, collections, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
D = "reports/alfworld/env_evidence"
F1 = json.load(open(f"{D}/features.json")); F2 = json.load(open(f"{D}/r2_vl/features.json"))
HS = ["contradicts", "silent", "supports", "n/a"]
env = lambda f: [f["plus_goal_revealed"], math.log1p(f["plus_n_new"]), f["plus_no_info"], f["gain_new"]] + [f["hindsight"] == h for h in HS]
dist = lambda f: [f["H_student"], f["H_teacher"], f["KL_ts"], f["KL_st"], math.log(f["p_s_cand"] + 1e-6), math.log(f["p_t_cand"] + 1e-6), math.log(f["p_s_minus"] + 1e-6), math.log(f["p_t_minus"] + 1e-6)]
both = lambda f: env(f) + dist(f)
sk = lambda f: (f["id"], f["t"])
def model(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
rng = np.random.default_rng(0)
def ci_auc(F, s, label=lambda f: f["won"]):
    rows = [(f, v) for f, v in zip(F, s) if label(f) is not None]; F, s = [r[0] for r in rows], np.array([r[1] for r in rows])
    y = np.array([label(f) for f in F], int); st = sorted({sk(f) for f in F}); idx = collections.defaultdict(list)
    for i, f in enumerate(F): idx[sk(f)].append(i)
    bs = []
    for _ in range(1000):
        ii = [j for k in rng.choice(len(st), len(st)) for j in idx[st[k]]]
        if len(set(y[ii])) > 1: bs.append(roc_auc_score(y[ii], s[ii]))
    return roc_auc_score(y, s), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
def cv_scores(F, fx):
    keys = sorted({sk(f) for f in F}); random.Random(0).shuffle(keys); fold = {k: i % 5 for i, k in enumerate(keys)}; sc = np.zeros(len(F))
    for i in range(5):
        tr = [j for j, f in enumerate(F) if fold[sk(f)] != i]; te = [j for j, f in enumerate(F) if fold[sk(f)] == i]
        m = model().fit(np.array([fx(F[j]) for j in tr], float), [F[j]["won"] for j in tr]); sc[te] = m.predict_proba(np.array([fx(F[j]) for j in te], float))[:, 1]
    return sc
print(f"round 1: {len(F1)} candidates, win {np.mean([f['won'] for f in F1]):.0%} | round 2: {len(F2)} candidates ({len({sk(f) for f in F2})} states), win {np.mean([f['won'] for f in F2]):.0%}")
print("\n== AUC for 'candidate wins' on round 2 (95% CI, state-clustered bootstrap)")
print(f"  {'':34s} {'frozen round-1 model':>28s}   {'refit on round 2 (5-fold CV)':>28s}")
for name, fx in [("env (one-step + hindsight)", env), ("dist (entropy/KL/probs)", dist), ("env + dist", both)]:
    m = model().fit(np.array([fx(f) for f in F1], float), [f["won"] for f in F1]); p = m.predict_proba(np.array([fx(f) for f in F2], float))[:, 1]
    a, lo, hi = ci_auc(F2, p); a2, lo2, hi2 = ci_auc(F2, cv_scores(F2, fx))
    print(f"  {name:34s} {a:.3f} [{lo:.3f}, {hi:.3f}]          {a2:.3f} [{lo2:.3f}, {hi2:.3f}]")
print("\n== single signals, round 1 vs round 2 (direction fixed as in round 1)")
for name, fx in [("goal object revealed", lambda f: f["plus_goal_revealed"]), ("#new entities", lambda f: f["plus_n_new"]), ("new info", lambda f: -f["plus_no_info"]),
                 ("hindsight (contradicts<silent<supports)", lambda f: {"contradicts": 0, "silent": 1, "n/a": 1, "supports": 2}[f["hindsight"]]),
                 ("student entropy, high first", lambda f: f["H_student"]), ("teacher p(cand)", lambda f: f["p_t_cand"]), ("KL(teacher||student)", lambda f: f["KL_ts"])]:
    a1 = roc_auc_score([f["won"] for f in F1], [fx(f) for f in F1]); a2, lo, hi = ci_auc(F2, [fx(f) for f in F2])
    print(f"  {name:42s} R1 {a1:.3f} | R2 {a2:.3f} [{lo:.3f}, {hi:.3f}]")
print("\n== same state, one wins / one loses (round 2): picks winner")
by = collections.defaultdict(list)
for f in F2: by[sk(f)].append(f)
pairs = [v for v in by.values() if len(v) == 2 and v[0]["won"] != v[1]["won"]]
m_env = model().fit(np.array([env(f) for f in F1], float), [f["won"] for f in F1])
fz = {(sk(f), f["cand"]): p for f, p in zip(F2, m_env.predict_proba(np.array([env(f) for f in F2], float))[:, 1])}
for sname, s in [("teacher's first choice", lambda f: -f["rank"]), ("teacher p(cand)", lambda f: f["p_t_cand"]), ("frozen round-1 env model", lambda f: fz[(sk(f), f["cand"])]),
                 ("goal object revealed", lambda f: f["plus_goal_revealed"]), ("hindsight", lambda f: {"contradicts": 0, "silent": 1, "n/a": 1, "supports": 2}[f["hindsight"]])]:
    acc = [1.0 if s(w) > s(l) else 0.5 if s(w) == s(l) else 0.0 for v in pairs for w, l in [(v[0], v[1]) if v[0]["won"] else (v[1], v[0])]]
    print(f"  {sname:28s} {np.mean(acc):.2f}  (n pairs {len(pairs)})")
print("\n== state level (any candidate wins), round 2")
S = list(by.values()); y = [any(x["won"] for x in v) for v in S]
for sname, fn in [("student entropy, high first", lambda v: v[0]["H_student"]), ("position, early first", lambda v: -v[0]["t"])]:
    print(f"  {sname:28s} AUC {roc_auc_score(y, [fn(v) for v in S]):.3f}  (R2 salvageable {np.mean(y):.0%})")
con = [f for f in F2 if f["hindsight"] == "contradicts"]; W = sum(f["won"] for f in F2)
print(f"\n== hindsight veto on round 2: removes {len(con)}/{len(F2)} ({len(con) / len(F2):.0%}), keeps {(W - sum(f['won'] for f in con)) / W:.0%} of winners; vetoed win {np.mean([f['won'] for f in con]):.0%} vs kept {np.mean([f['won'] for f in F2 if f['hindsight'] != 'contradicts']):.0%}")
