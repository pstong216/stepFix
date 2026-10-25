"""Step-selector arms, judged ONLY by the student trained from them (user decision Oct 5): same 319 original failures, same candidate
source (the student's top-2 alternatives at the step, from the r0 scan where every one of the first 8 steps was already executed one
step in the environment), same filter (env one-step ranking, veto no-info, top K=86), same training.  Only the STEP differs:
  sel_log       first step whose own action produced non-positive verifiable evidence (log only, zero cost)
  sel_log_fix   same, shifted one step earlier when the previous action was a move ("go to") -- the systematic off-by-one
  sel_m5        M5 progress-estimator pointer (first predicted regress)
  sel_envattr   env-evidence attribution (first step with a clearly better executed alternative)
  sel_a0        learned attributor A0 pointer
  sel_random    random step in 0..7
  pool_multi    all non-positive log steps (<= 2 per failure) -> the env filter ranks across the pool
  pool_all      every scanned step (0..7) -> the env filter alone decides (no attribution at all)
Writes reports/alfworld/rigorous/data/<arm>.jsonl and appends the arms to reports/alfworld/rigorous/selector_queue.txt.
"""
import json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from env_evidence_filter import evidence
from alfworld_relaxed_verify import parse_goal
from compare_step_selectors import log_rule, taken_scores
L = lambda p: [json.loads(l) for l in open(ROOT / p)]
TRAJ = "reports/alfworld/student3b_low100_train_b50.jsonl"; R0 = "reports/alfworld/self_attr_labels/r0"; K = 86
eps = {e["id"]: e for e in L(TRAJ)}
cands = {(r["id"], r["t"]): r for r in L(f"{R0}/cands.jsonl")}; one = {(r["id"], r["t"], r["cand"]): r for r in L(f"{R0}/onestep.jsonl")}
ids = sorted({i for i, _ in cands}); rng = random.Random(0)
m5 = {r["id"]: r["t_hat"] for r in L("reports/alfworld/progress_model/pointer_r1.jsonl")}
envl = {r["id"]: r["t_star"] for r in L(f"{R0}/labels.jsonl")}
a0 = {r["id"]: r["t_hat"] for r in L("reports/alfworld/self_attr/r0/both/pointer_real319.raw.jsonl")}


def sc(e, t, c, o):
    ev = evidence(e, t, c, o, parse_goal(e["goal"])["obj"]); return 2 * ev["goal_revealed"] + 2 * ev["subgoal"] + min(ev["n_new"], 5) / 5 - 2 * ev["no_info"], ev["no_info"]


def best_at(i, t):
    """best executed alternative at (i,t) by one-step evidence; None if no scan / all vetoed."""
    r = cands.get((i, t)); e = eps[i]
    if not r: return None
    ok = []
    for c in r["cands"]:
        o = one.get((i, t, c))
        if not o: continue
        s, ni = sc(e, t, c, o["obs_plus"]); ok.append((s, c)) if (not ni and s > 0) else None
    return max(ok) if ok else None


def log_fix(e):
    t = log_rule(e)
    return t - 1 if t > 0 and e["steps"][t - 1]["action"].startswith("go to") else t


def nonpos_steps(e):
    s = taken_scores(e); return [t for t in range(min(8, len(s))) if s[t] <= 0]


selectors = {"sel_log": lambda i: [log_rule(eps[i])], "sel_log_fix": lambda i: [log_fix(eps[i])], "sel_m5": lambda i: [m5[i]] if i in m5 else [],
             "sel_envattr": lambda i: [envl[i]] if i in envl else [], "sel_a0": lambda i: [a0[i]] if i in a0 else [], "sel_random": lambda i: [rng.randrange(0, min(8, len(eps[i]["steps"])))],
             "pool_multi": lambda i: nonpos_steps(eps[i])[:2] or [log_rule(eps[i])], "pool_all": lambda i: list(range(min(8, len(eps[i]["steps"]))))}
out = ROOT / "reports/alfworld/rigorous/data"; out.mkdir(parents=True, exist_ok=True); q = open(ROOT / "reports/alfworld/rigorous/selector_queue.txt", "a")
for name, fn in selectors.items():
    picks = []; n_states = 0; n_in_scan = 0
    for i in ids:
        for t in fn(i):
            n_states += 1
            if (i, t) not in cands: continue
            n_in_scan += 1; b = best_at(i, t)
            if b: picks.append((b[0], i, t, b[1]))
    picks.sort(reverse=True)
    if name.startswith("pool"):   # at most 2 states per failure after ranking
        seen = {}; kept = []
        for p in picks:
            if seen.get(p[1], 0) < 2: kept.append(p); seen[p[1]] = seen.get(p[1], 0) + 1
        picks = kept
    picks = picks[:K]
    with open(out / f"{name}.jsonl", "w") as f:
        for s, i, t, c in picks:
            cmds = list(eps[i]["steps"][t]["admissible"]); f.write(json.dumps({"kind": "kl", "src": TRAJ, "id": i, "t": t, "cmds": cmds, "target": [1.0 if x == c else 0.0 for x in cmds], "cand": c}) + "\n")
    ts = [t for _, _, t, _ in picks]
    print(f"{name:12s} states {n_states:4d} (in scan {n_in_scan:4d}) -> {len(picks)} targets | chosen t: mean {sum(ts) / len(ts):.1f}, t0 {sum(t == 0 for t in ts) / len(ts):.2f} | mean evidence {sum(p[0] for p in picks) / len(picks):.2f}")
    q.write(f"{name}\treports/alfworld/rigorous/data/{name}.jsonl:1.0\n")
q.close(); print("queue written: reports/alfworld/rigorous/selector_queue.txt")
