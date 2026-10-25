"""Offline simulation of active verification: each round the learner verifies (one rollout) one candidate in each of B
states; the proxy (logistic on env + dist features) is refit on everything verified so far. Compare strategies by
winning corrections found per rollout, and by proxy AUC on the not-yet-verified candidates.
  S1 round 1 (3B student) from scratch;  S2 round 2 starting from the frozen round-1 proxy (student has changed)."""
import json, math, random, collections, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
D = "reports/alfworld/env_evidence"; HS = ["contradicts", "silent", "supports", "n/a"]
fx = lambda f: [f["plus_goal_revealed"], math.log1p(f["plus_n_new"]), f["plus_no_info"], f["gain_new"]] + [f["hindsight"] == h for h in HS] + \
               [f["H_student"], f["H_teacher"], f["KL_ts"], f["KL_st"], math.log(f["p_s_cand"] + 1e-6), math.log(f["p_t_cand"] + 1e-6)]
def model(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
def run(F, strategy, B, rounds, seed, init=None):
    rng = random.Random(seed); by = collections.defaultdict(list)
    for i, f in enumerate(F): by[(f["id"], f["t"])].append(i)
    X = np.array([fx(f) for f in F], float); y = np.array([f["won"] for f in F], int)
    open_states = set(by); lab = []; found = []; aucs = []; m = init
    for r in range(rounds):
        if not open_states: break
        cand = [i for s in open_states for i in by[s]]
        p = m.predict_proba(X[cand])[:, 1] if m is not None else np.full(len(cand), 0.5)
        score = dict(zip(cand, p)); best_in_state = {s: max(by[s], key=lambda i: score[i]) for s in open_states}
        if strategy == "random" or m is None:
            pick = rng.sample(sorted(open_states), min(B, len(open_states))); chosen = [min(by[s], key=lambda i: F[i]["rank"]) for s in pick]
        elif strategy in ("greedy", "frozen"):
            pick = sorted(open_states, key=lambda s: -score[best_in_state[s]])[:B]; chosen = [best_in_state[s] for s in pick]
        elif strategy == "greedy+explore":   # 75% exploit, 25% most uncertain (p closest to 0.5) to keep the proxy calibrated
            ex = sorted(open_states, key=lambda s: -score[best_in_state[s]]); k = int(0.75 * B); pick = ex[:k]
            rest = sorted(set(open_states) - set(pick), key=lambda s: abs(score[best_in_state[s]] - 0.5))[: B - k]; pick += rest
            chosen = [best_in_state[s] for s in pick]
        for s in pick: open_states.discard(s)
        lab += chosen; found.append(sum(y[i] for i in chosen))
        if strategy != "frozen" and len(set(y[lab])) > 1:
            m = model().fit(X[lab] if init is None else np.vstack([X[lab]]), y[lab]) if init is None else model().fit(np.vstack([INIT_X, X[lab]]), np.concatenate([INIT_Y, y[lab]]))
        rest_idx = [i for s in open_states for i in by[s]]
        if m is not None and rest_idx and len(set(y[rest_idx])) > 1: aucs.append(roc_auc_score(y[rest_idx], m.predict_proba(X[rest_idx])[:, 1]))
    return np.cumsum(found), aucs
F1 = json.load(open(f"{D}/features.json")); F2 = json.load(open(f"{D}/r2_vl/features.json"))
print("== S1: round 1 (3B), 319 states, 20 verifications per round, 8 rounds (160 rollouts); mean over 20 seeds")
for st in ["random", "greedy", "greedy+explore"]:
    R = [run(F1, st, 20, 8, s) for s in range(20)]; c = np.mean([r[0] for r in R], 0); a = np.mean([r[1][-1] for r in R if r[1]])
    print(f"  {st:16s} wins after 40/80/160 rollouts: {c[1]:.1f} / {c[3]:.1f} / {c[7]:.1f}   proxy AUC on unverified at the end {a:.3f}")
INIT_X = np.array([fx(f) for f in F1], float); INIT_Y = np.array([f["won"] for f in F1], int)
m1 = model().fit(INIT_X, INIT_Y)
print("== S2: round 2 (stronger student), 528 states, start from the round-1 proxy; 20 per round, 8 rounds")
for st in ["random", "frozen", "greedy", "greedy+explore"]:
    R = [run(F2, st, 20, 8, s, init=m1) for s in range(20)]; c = np.mean([r[0] for r in R], 0); aa = [r[1][-1] for r in R if r[1]]
    print(f"  {st:16s} wins after 40/80/160 rollouts: {c[1]:.1f} / {c[3]:.1f} / {c[7]:.1f}   proxy AUC on unverified at the end {np.mean(aa) if aa else float('nan'):.3f}")
