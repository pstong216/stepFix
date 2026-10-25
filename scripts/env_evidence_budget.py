"""Two-stage selection under a verification budget: WHERE (state-level, student entropy) x WHICH (candidate-level,
environment evidence). Each verification = one full rollout of one candidate. Count winning corrections found."""
import json, math, random, collections, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
D = "reports/alfworld/env_evidence"
F3 = json.load(open(f"{D}/features.json")); F7 = json.load(open(f"{D}/teacher7b/features.json"))
HS = ["contradicts", "silent", "supports", "n/a"]
env = lambda f: [f["plus_goal_revealed"], math.log1p(f["plus_n_new"]), f["plus_no_info"], f["gain_new"]] + [f["hindsight"] == h for h in HS]
def model(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
sk = lambda f: (f["id"], f["t"])
def env_scores(F, train=None):
    sc = {}
    if train is not None:   # fit on another set (state-disjoint), apply to F
        m = model().fit(np.array([env(f) for f in train], float), [f["won"] for f in train])
        for f, p in zip(F, m.predict_proba(np.array([env(f) for f in F], float))[:, 1]): sc[(sk(f), f["cand"])] = p
        return sc
    keys = sorted({sk(f) for f in F}); random.Random(0).shuffle(keys); fold = {k: i % 5 for i, k in enumerate(keys)}
    for i in range(5):
        tr = [f for f in F if fold[sk(f)] != i]; te = [f for f in F if fold[sk(f)] == i]
        m = model().fit(np.array([env(f) for f in tr], float), [f["won"] for f in tr])
        for f, p in zip(te, m.predict_proba(np.array([env(f) for f in te], float))[:, 1]): sc[(sk(f), f["cand"])] = p
    return sc
def run(name, F, sc):
    by = collections.defaultdict(list)
    for f in F: by[sk(f)].append(f)
    states = list(by); n = len(states); W = sum(any(x["won"] for x in v) for v in by.values())
    ent = {s: by[s][0]["H_student"] for s in states}
    rng = random.Random(1)
    def policy(order_states, pick):
        # verify one candidate per state in the given order; returns cumulative wins after each rollout
        found = []; c = 0
        for s in order_states:
            c += pick(by[s])["won"]; found.append(c)
        return found
    teacher_first = lambda v: min(v, key=lambda f: f["rank"])
    env_best = lambda v: max(v, key=lambda f: sc[(sk(f), f["cand"])])
    rand_order = lambda: rng.sample(states, n)
    ent_order = sorted(states, key=lambda s: -ent[s])
    res = {}
    res["random states + teacher's 1st choice"] = np.mean([policy(rand_order(), teacher_first) for _ in range(200)], 0)
    res["high-entropy states + teacher's 1st choice"] = np.array(policy(ent_order, teacher_first), float)
    res["random states + env-best candidate"] = np.mean([policy(rand_order(), env_best) for _ in range(200)], 0)
    res["high-entropy states + env-best candidate"] = np.array(policy(ent_order, env_best), float)
    oracle = sorted([any(x["won"] for x in v) for v in by.values()], reverse=True); res["(oracle: perfect state + candidate choice)"] = np.cumsum(oracle).astype(float)
    print(f"\n== {name}: {n} states, {W} salvageable (some candidate wins); current practice verifies both candidates of every state = {2 * n} rollouts -> {W} wins")
    print(f"  {'policy (1 rollout per chosen state)':46s} " + "  ".join(f"budget {int(b * n):3d}" for b in (0.25, 0.5, 1.0)))
    for k, v in res.items():
        print(f"  {k:46s} " + "  ".join(f"   {v[int(b * n) - 1]:5.1f}  " for b in (0.25, 0.5, 1.0)))
run("3B teacher (env model: 5-fold CV by state)", F3, env_scores(F3))
test_states = {sk(f) for f in F7}; run("7B teacher (env model fit on 3B, state-disjoint)", F7, env_scores(F7, train=[f for f in F3 if sk(f) not in test_states]))
