"""Does the filter identify easy-to-save STATES or better ACTIONS?
 (1) paired analysis: states where one candidate wins and the other loses -> accuracy of picking the winner (ties 0.5)
 (2) hindsight veto within action type
 (3) state-disjoint 3B->7B transfer with state-clustered bootstrap CIs
"""
import json, math, random, collections, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
D = "reports/alfworld/env_evidence"
F3 = json.load(open(f"{D}/features.json")); F7 = json.load(open(f"{D}/teacher7b/features.json"))
HS = ["contradicts", "silent", "supports", "n/a"]
env = lambda f: [f["plus_goal_revealed"], math.log1p(f["plus_n_new"]), f["plus_no_info"], f["gain_new"]] + [f["hindsight"] == h for h in HS]
dist = lambda f: [f["H_student"], f["H_teacher"], f["KL_ts"], f["KL_st"], math.log(f["p_s_cand"] + 1e-6), math.log(f["p_t_cand"] + 1e-6), math.log(f["p_s_minus"] + 1e-6), math.log(f["p_t_minus"] + 1e-6)]
both = lambda f: env(f) + dist(f)
hs_score = lambda f: {"contradicts": 0, "silent": 1, "n/a": 1, "supports": 2}[f["hindsight"]]
def model(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
def sk(f): return (f["id"], f["t"])

print("== (1) same state, one candidate wins and the other loses: accuracy of picking the winner (ties = 0.5)")
for name, F in [("3B", F3), ("7B", F7)]:
    by = collections.defaultdict(list)
    for f in F: by[sk(f)].append(f)
    pairs = [v for v in by.values() if len(v) == 2 and v[0]["won"] != v[1]["won"]]
    nst = len(by); both_w = sum(len(v) == 2 and v[0]["won"] and v[1]["won"] for v in by.values()); none_w = sum(all(not x["won"] for x in v) for v in by.values())
    print(f" {name}: {nst} states -> one-wins pairs {len(pairs)}, both win {both_w}, neither wins {none_w}")
    # CV model restricted to candidate-varying features, fit on all candidates of other states
    keys = list(by); random.Random(0).shuffle(keys); fold = {k: i % 5 for i, k in enumerate(keys)}
    def cvscore(fx):
        sc = {}
        for i in range(5):
            tr = [f for f in F if fold[sk(f)] != i]; te = [f for f in F if fold[sk(f)] == i]
            m = model().fit(np.array([fx(f) for f in tr], float), [f["won"] for f in tr])
            for f, p in zip(te, m.predict_proba(np.array([fx(f) for f in te], float))[:, 1]): sc[(sk(f), f["cand"])] = p
        return lambda f: sc[(sk(f), f["cand"])]
    scorers = [("teacher's first choice (rank 0)", lambda f: -f["rank"]), ("teacher p(cand)", lambda f: f["p_t_cand"]), ("student p(cand)", lambda f: f["p_s_cand"]),
               ("student entropy (state-level)", lambda f: f["H_student"]), ("KL teacher||student (state-level)", lambda f: f["KL_ts"]),
               ("one-step: goal object revealed", lambda f: f["plus_goal_revealed"]), ("one-step: #new entities", lambda f: f["plus_n_new"]),
               ("one-step: new info (not no-info)", lambda f: -f["plus_no_info"]), ("hindsight (contradicts<silent<supports)", hs_score),
               ("CV model: env", cvscore(env)), ("CV model: dist", cvscore(dist)), ("CV model: env+dist", cvscore(both))]
    for sname, s in scorers:
        acc = [1.0 if s(w) > s(l) else 0.5 if s(w) == s(l) else 0.0 for v in pairs for w, l in [(v[0], v[1]) if v[0]["won"] else (v[1], v[0])]]
        ties = sum(1 for v in pairs if s(v[0]) == s(v[1]))
        print(f"   {sname:42s} picks winner {np.mean(acc):.2f}  (ties {ties}/{len(pairs)})")

print("\n== (2) hindsight veto: candidates the failed log contradicts, overall and within action type")
for name, F in [("3B", F3), ("7B", F7)]:
    n = len(F); W = sum(f["won"] for f in F); con = [f for f in F if f["hindsight"] == "contradicts"]
    print(f" {name}: veto removes {len(con)}/{n} ({len(con) / n:.0%}) candidates, keeps {(W - sum(f['won'] for f in con)) / W:.0%} of winners; win rate vetoed {np.mean([f['won'] for f in con]):.0%} vs kept {np.mean([f['won'] for f in F if f['hindsight'] != 'contradicts']):.0%}")
    for verb in ["go", "open"]:
        v = [f for f in F if f["cand"].split()[0] == verb]
        c = [f for f in v if f["hindsight"] == "contradicts"]; k = [f for f in v if f["hindsight"] != "contradicts"]
        if c and k: print(f"    within '{verb}' candidates (n={len(v)}): vetoed {len(c)} win {np.mean([f['won'] for f in c]):.0%} vs kept {len(k)} win {np.mean([f['won'] for f in k]):.0%}")
    gv = [f for f in F if f["cand"].startswith("go to")]
    for t0, t1 in [(0, 0), (1, 99)]:
        v = [f for f in gv if t0 <= f["t"] <= t1]; c = [f for f in v if f["hindsight"] == "contradicts"]; k = [f for f in v if f["hindsight"] != "contradicts"]
        if c and k: print(f"    'go to' at step {t0}-{t1}: vetoed {len(c)} win {np.mean([f['won'] for f in c]):.0%} vs kept {len(k)} win {np.mean([f['won'] for f in k]):.0%}")

print("\n== (3) 3B->7B transfer, train states DISJOINT from the 7B test states; 95% CI by state-clustered bootstrap")
test_states = {sk(f) for f in F7}; tr = [f for f in F3 if sk(f) not in test_states]
print(f" train: {len(tr)} 3B candidates from {len({sk(f) for f in tr})} states (the 100 test states removed); test: {len(F7)} 7B candidates")
rng = np.random.default_rng(0); states = sorted(test_states); idx = collections.defaultdict(list)
for i, f in enumerate(F7): idx[sk(f)].append(i)
y7 = np.array([f["won"] for f in F7])
for sname, fx in [("env", env), ("dist", dist), ("env+dist", both)]:
    p = model().fit(np.array([fx(f) for f in tr], float), [f["won"] for f in tr]).predict_proba(np.array([fx(f) for f in F7], float))[:, 1]
    bs = []
    for _ in range(2000):
        ii = [j for s in rng.choice(len(states), len(states)) for j in idx[states[s]]]
        if len(set(y7[ii])) > 1: bs.append(roc_auc_score(y7[ii], p[ii]))
    print(f"   {sname:9s} AUC {roc_auc_score(y7, p):.3f}  95% CI [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]")
for sname, fx in [("student entropy, high first", lambda f: f["H_student"]), ("goal object revealed", lambda f: f["plus_goal_revealed"])]:
    s = np.array([fx(f) for f in F7], float); bs = []
    for _ in range(2000):
        ii = [j for k in rng.choice(len(states), len(states)) for j in idx[states[k]]]
        if len(set(y7[ii])) > 1: bs.append(roc_auc_score(y7[ii], s[ii]))
    print(f"   single: {sname:28s} AUC {roc_auc_score(y7, s):.3f}  95% CI [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]")
