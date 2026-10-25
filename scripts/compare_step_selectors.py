"""Offline comparison of cheap step selectors against the replay-based env-evidence labels (and, for S0, the planner oracle):
  log_rule   first step whose OWN action produced non-positive verifiable evidence (from the log; zero cost)
  m5_fixed   M5: first step with E[k]_{t+1} - E[k]_t > 1 (current chain rule)
  m5_z       M5: first step whose delta exceeds the 95th percentile of deltas at the same step index on WON trajectories
  m5_cusum   M5: CUSUM on z-scored deltas (k=0.5), alarm threshold calibrated to 5% false alarms on wins; pointer = run start
  union      steps proposed by log_rule OR m5_z (pool size per failure)
Metrics: +-1 agreement with the env-evidence label, exact agreement, share at step 0, mean, and "actionable" = the chosen step has
an alternative whose one-step evidence beats the taken action by >= 2 (only where the alternatives were executed: first 8 steps).
"""
import json, os, statistics as st, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from env_evidence_filter import evidence
from alfworld_relaxed_verify import parse_goal
O = ROOT / "reports/alfworld/step_selectors"
SETS = {"s0": ("reports/alfworld/student3b_low100_train_b50.jsonl", "reports/alfworld/self_attr_labels/r0/labels.jsonl", "reports/alfworld/progress_student3b_low100_train_b50.jsonl"),
        "s1": ("reports/alfworld/self_attr_rounds/selfattr/r2/traj.jsonl", "reports/alfworld/self_attr_rounds/selfattr/r2/envlab/labels.jsonl", "reports/alfworld/step_selectors/s1_oracle.jsonl")}


def L(p): return [json.loads(l) for l in open(p)] if p and os.path.exists(p) else []


def taken_scores(e):
    if os.environ.get("EVIDENCE") == "generic":   # one generic template for every environment (Oct 5 23:10)
        from generic_evidence import alfworld_step
        return [alfworld_step(e, t, s["next_obs"])["score"] for t, s in enumerate(e["steps"])]
    obj = parse_goal(e["goal"])["obj"]; out = []
    for t, s in enumerate(e["steps"]):
        ev = evidence(e, t, s["action"], s["next_obs"], obj); out.append(2 * ev["goal_revealed"] + 2 * ev["subgoal"] + min(ev["n_new"], 5) / 5 - 2 * ev["no_info"])
    return out


def log_rule(e):
    s = taken_scores(e); return next((t for t in range(len(s)) if s[t] <= 0), min(range(len(s)), key=lambda t: s[t]))


def deltas(ex): return [ex[t + 1] - ex[t] for t in range(len(ex) - 1)]


def m5_fixed(ex, delta=1.0):
    d = deltas(ex)
    if not d: return 0
    return next((t for t in range(len(d)) if d[t] > delta), max(range(len(d)), key=lambda t: d[t]))


def calib(win_scores):
    """per-step-index 95th percentile of win deltas and (mu, sigma) for z-scoring; indices >= 8 pooled."""
    by = {}
    for r in win_scores:
        for t, d in enumerate(deltas(r["exp_remaining"])): by.setdefault(min(t, 8), []).append(d)
    q = {}; mu = {}; sd = {}
    for t, v in by.items():
        v = sorted(v); q[t] = v[int(0.95 * (len(v) - 1))]; mu[t] = st.mean(v); sd[t] = st.pstdev(v) or 1.0
    return q, mu, sd


def m5_z(ex, cal):
    q, mu, sd = cal; d = deltas(ex)
    if not d: return 0
    z = [(d[t] - mu[min(t, 8)]) / sd[min(t, 8)] for t in range(len(d))]
    return next((t for t in range(len(d)) if d[t] > q[min(t, 8)]), max(range(len(d)), key=lambda t: z[t]))


def cusum_stat(ex, cal, k=0.5):
    q, mu, sd = cal; d = deltas(ex); S = 0.0; start = 0; out = []
    for t in range(len(d)):
        z = (d[t] - mu[min(t, 8)]) / sd[min(t, 8)]
        if S == 0: start = t
        S = max(0.0, S + z - k); out.append((S, start))
    return out


def m5_cusum(ex, cal, h):
    cs = cusum_stat(ex, cal)
    for S, start in cs:
        if S > h: return start
    return max(range(len(cs)), key=lambda t: cs[t][0]) if cs else 0


def main():
    for name, (traj, lab, oracle) in SETS.items():
        eps = {e["id"]: e for e in L(ROOT / traj)}; labs = {r["id"]: r for r in L(ROOT / lab)}
        if not labs: continue
        orc = {r["id"]: r.get("t_first_regress") for r in L(ROOT / oracle) if oracle} if oracle else {}
        fs = {r["id"]: r["exp_remaining"] for r in L(O / f"{name}_fail.scores.jsonl")}
        if name == "s0" and not fs: fs = {r["id"]: r["exp_remaining"] for r in L(ROOT / "reports/alfworld/progress_model/scores_r1.jsonl")}
        ws = L(O / f"{name}_win.scores.jsonl"); cal = calib(ws) if ws else None
        h = None
        if cal:   # CUSUM threshold: 95th percentile of the max CUSUM statistic on wins
            mx = sorted(max((S for S, _ in cusum_stat(r["exp_remaining"], cal)), default=0.0) for r in ws); h = mx[int(0.95 * (len(mx) - 1))]
        rules = {"log_rule": lambda i, e: log_rule(e)}
        if fs: rules["m5_fixed"] = lambda i, e: m5_fixed(fs[i]) if i in fs else None
        if fs and cal: rules["m5_z"] = lambda i, e: m5_z(fs[i], cal) if i in fs else None; rules["m5_cusum"] = lambda i, e: m5_cusum(fs[i], cal, h) if i in fs else None
        print(f"\n=== step selectors on {name}: {len(labs)} failures with env-evidence labels" + (f" | M5 scores {len(fs)} | win calibration {len(ws)}" if fs else " | (no M5 scores yet)"))
        ref = {i: labs[i]["t_star"] for i in labs}
        print(f"{'selector':10s} {'cov':>4s} {'+-1 env':>8s} {'exact':>6s} {'t0':>5s} {'mean':>5s} {'+-1 oracle':>10s} {'actionable':>10s}")
        rows_out = {}
        for rn, fn in rules.items():
            pred = {i: fn(i, eps[i]) for i in labs if i in eps}; pred = {i: t for i, t in pred.items() if t is not None}; rows_out[rn] = pred
            if not pred: continue
            ag = st.mean(abs(pred[i] - ref[i]) <= 1 for i in pred); ex = st.mean(pred[i] == ref[i] for i in pred)
            t0 = st.mean(pred[i] == 0 for i in pred); mn = st.mean(pred.values())
            oa = st.mean(abs(pred[i] - orc[i]) <= 1 for i in pred if orc.get(i) is not None) if orc else float("nan")
            act = [labs[i]["per_step"].get(str(pred[i]), {}).get("margin") for i in pred]; act = [a for a in act if a is not None]
            actionable = st.mean(a >= 2 for a in act) if act else float("nan")
            print(f"{rn:10s} {len(pred):4d} {ag:8.2f} {ex:6.2f} {t0:5.2f} {mn:5.1f} {oa:10.2f} {actionable:10.2f}  (actionable measured on {len(act)} steps within the scanned window)")
        if "log_rule" in rows_out and "m5_z" in rows_out:
            u = {i: {rows_out["log_rule"][i], rows_out["m5_z"][i]} for i in rows_out["m5_z"] if i in rows_out["log_rule"]}
            hit = st.mean(any(abs(t - ref[i]) <= 1 for t in s) for i, s in u.items()); size = st.mean(len(s) for s in u.values())
            print(f"union      {len(u):4d} {hit:8.2f}  (pool hits the env label within +-1) | mean pool size {size:.2f}")
        ref_t0 = st.mean(ref[i] == 0 for i in ref); print(f"env-evidence label itself: t0 {ref_t0:.2f}, mean {st.mean(ref.values()):.1f}" + (f" | +-1 agreement with planner oracle {st.mean(abs(ref[i]-orc[i])<=1 for i in ref if orc.get(i) is not None):.2f}" if orc else ""))


if __name__ == "__main__":
    main()


def log_fix(e):
    """log rule, shifted one step earlier when the previous action was a move ('go to'): the first non-positive step is usually
    the wrong *take* after arriving at a wrong place; the actionable mistake is the move (Oct 5: plain rule broke round 2 of the
    chains -- 76-80 of 86 corrections replaced a take with 'go elsewhere' and the student stopped picking things up, 175 -> 137)."""
    t = log_rule(e)
    return t - 1 if t > 0 and e["steps"][t - 1]["action"].startswith("go to") else t


def log_late(e, window=3):
    """LATE log rule (user request Oct 5 21:40, to fix 'R2 does not rise'): point at the move that led into the plateau where the log
    stops producing new verifiable information, instead of the first non-positive step.  p = first index such that `window`
    consecutive steps from p all have non-positive evidence (the wandering/loop regime begins); pointer = last 'go to' step at or
    before p (the move that entered the plateau); falls back to log_fix when no plateau or no move is found."""
    s = taken_scores(e); n = len(s)
    p = next((t for t in range(n) if all(s[u] <= 0 for u in range(t, min(n, t + window))) and t + window <= n), None)
    if p is None: return log_fix(e)
    moves = [t for t in range(p + 1) if e["steps"][t]["action"].startswith("go to")]
    return moves[-1] if moves else log_fix(e)


def log_late2(e, window=2):
    """LATE rule, final form (Oct 5 21:50): the first move after the first take (after the student committed to an object, the next
    move is where the trajectory goes wrong; best agreement with the attributor that worked in R2: within-1 0.51, within-2 0.78);
    no take in the log -> plateau rule log_late(window); fallback log_fix."""
    s = e["steps"]; ft = next((t for t, x in enumerate(s) if x["action"].startswith("take")), None)
    if ft is not None:
        m = next((t for t in range(ft + 1, len(s)) if s[t]["action"].startswith("go to")), None)
        if m is not None: return m
    return log_late(e, window)


def plateau_start(scores, window=2):
    n = len(scores)
    return next((t for t in range(n) if t + window <= n and all(scores[u] <= 0 for u in range(t, t + window))), None)


def late_generic(scores, window=2):
    """GENERIC late rule (no action vocabulary): p = start of the first run of `window` consecutive steps whose own one-step evidence
    is non-positive (the log stops producing verifiable information); pointer = p-1, the decision that led into the stall (p if p=0).
    No stall: the step with the lowest evidence."""
    p = plateau_start(scores, window)
    if p is None: return min(range(len(scores)), key=lambda t: scores[t])
    return max(p - 1, 0)


def first_nonpos_generic(scores):
    """GENERIC first-step rule: first step whose own evidence is non-positive, minus one (the decision before it)."""
    t = next((t for t in range(len(scores)) if scores[t] <= 0), None)
    return 0 if not t else t - 1
