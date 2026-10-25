"""Combine teacher-action filter signals (logistic regression, 5-fold CV grouped by state; plus 3B->7B transfer)."""
import json, math, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
D = "reports/alfworld/env_evidence"
F3 = json.load(open(f"{D}/features.json")); F7 = json.load(open(f"{D}/teacher7b/features.json"))
HS = ["contradicts", "silent", "supports", "n/a"]
SETS = {
    "env (one-step + hindsight)": lambda f: [f["plus_goal_revealed"], math.log1p(f["plus_n_new"]), f["plus_no_info"], f["gain_new"]] + [f["hindsight"] == h for h in HS],
    "one-step only": lambda f: [f["plus_goal_revealed"], math.log1p(f["plus_n_new"]), f["plus_no_info"], f["gain_new"]],
    "hindsight only": lambda f: [f["hindsight"] == h for h in HS],
    "dist (entropy/KL/probs)": lambda f: [f["H_student"], f["H_teacher"], f["KL_ts"], f["KL_st"], math.log(f["p_s_cand"] + 1e-6), math.log(f["p_t_cand"] + 1e-6), math.log(f["p_s_minus"] + 1e-6), math.log(f["p_t_minus"] + 1e-6)],
    "position": lambda f: [math.log1p(f["t"])],
}
SETS["env + dist"] = lambda f: SETS["env (one-step + hindsight)"](f) + SETS["dist (entropy/KL/probs)"](f)
SETS["env + dist + position"] = lambda f: SETS["env + dist"](f) + SETS["position"](f)
def model(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
def cv(F, fx, label):
    rows = [f for f in F if label(f) is not None]
    X = np.array([fx(f) for f in rows], float); y = np.array([label(f) for f in rows], int); g = [f["id"] + str(f["t"]) for f in rows]
    p = np.zeros(len(y)); 
    for tr, te in GroupKFold(5).split(X, y, g):
        p[te] = model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return roc_auc_score(y, p), p, y, rows
def prec_at(p, y, frac):
    k = max(1, int(len(p) * frac)); idx = np.argsort(-p)[:k]; return y[idx].mean(), y[idx].sum() / max(1, y.sum())
WIN = lambda f: int(f["won"]); CAUSAL = lambda f: None if f["orig_won"] is None else int(f["causal"])
for name, F in [("3B teacher", F3), ("7B teacher", F7)]:
    print(f"\n== {name}: {len(F)} candidates, win rate {np.mean([f['won'] for f in F]):.0%}")
    print(f"  {'feature set':28s} {'AUC win':>8s} {'AUC causal':>11s}   keep top 50%: win-precision / recall")
    for s, fx in SETS.items():
        a1, p, y, _ = cv(F, fx, WIN); a2, *_ = cv(F, fx, CAUSAL); pr, rc = prec_at(p, y, 0.5)
        print(f"  {s:28s} {a1:8.3f} {a2:11.3f}   {pr:.0%} / {rc:.0%}")
print("\n== transfer: fit on 3B-teacher candidates, test on 7B-teacher candidates (different teacher, unseen candidates)")
for s in ["env (one-step + hindsight)", "dist (entropy/KL/probs)", "env + dist", "env + dist + position"]:
    fx = SETS[s]; X3 = np.array([fx(f) for f in F3], float); y3 = np.array([f["won"] for f in F3], int)
    X7 = np.array([fx(f) for f in F7], float); y7 = np.array([f["won"] for f in F7], int)
    p = model().fit(X3, y3).predict_proba(X7)[:, 1]; pr, rc = prec_at(p, y7, 0.5)
    print(f"  {s:28s} AUC {roc_auc_score(y7, p):.3f} | keep top 50%: {pr:.0%} / {rc:.0%}")
m = model().fit(np.array([SETS['env + dist + position'](f) for f in F3], float), np.array([f['won'] for f in F3], int))
names = ["goal_revealed", "log#new", "no_info", "gain_new"] + [f"hind_{h}" for h in HS] + ["H_student", "H_teacher", "KL_ts", "KL_st", "logp_s_cand", "logp_t_cand", "logp_s_minus", "logp_t_minus", "log_t"]
print("\n== standardized coefficients (3B, all features):", ", ".join(f"{n} {c:+.2f}" for n, c in sorted(zip(names, m[-1].coef_[0]), key=lambda x: -abs(x[1]))))
