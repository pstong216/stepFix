"""Can a proxy replace the rollout in the 3B self-evolve run?  Every arm rolled out its two hindsight-teacher candidates at
each of 319 states (verified states keep per-candidate outcomes in round1/examples.jsonl; a state absent from it had no winner),
so each candidate has a rollout label.  Proxies: hindsight-teacher / plain-student probability, rank, state entropies and
KL(hindsight || plain), the hindsight audit of the failed log, one-step environment evidence (replay prefix, execute the
candidate once), and the detour detector's P(necessary) when scored.
  onestep  -- execute each candidate once in the env (CPU), cached in reports/alfworld/self_evolve3b/proxy/onestep.jsonl
  cands    -- dump (id, t, cand) rows for the detector's score_cands
  auc      -- per arm and pooled: AUC of each proxy vs rollout success, within-state accuracy (states where exactly one of
              the two candidates won), and the veto rule's recall / pass rate
"""
import argparse, json, math, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from env_evidence_filter import evidence, hindsight, softmax, H, KL, auc
from alfworld_relaxed_verify import parse_goal
import os
SE = ROOT / os.environ.get("SE_DIR", "reports/alfworld/self_evolve3b"); PX = SE / "proxy"
TRAJ = ROOT / os.environ.get("SE_TRAJ", "reports/alfworld/student3b_low100_train_b50.jsonl")
ARMS = ["attr", "m5", "selfattr", "entropy", "rand_early", "detour"]


def L(p): return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def arm_done(arm): return (SE / arm / "run.log").exists() and "DONE distill" in open(SE / arm / "run.log").read()


_MH = {}


def future_feats(e, t, c, obj):
    """The candidate against the failed trajectory's own future (steps > t): does the agent eventually take it, how many steps
    later (the detour it would skip), was it productive when taken (one-step evidence at that later step), did the milestone level
    rise afterwards, and does it leave the locations the agent cycles through right after t."""
    from analyze_pointer_action_types import classify_full
    if e["id"] not in _MH: _MH[e["id"]] = classify_full(e)[1]
    mh = _MH[e["id"]]; n = len(e["steps"])
    later = [j for j in range(t + 1, n) if e["steps"][j]["action"] == c]
    f = {"appears_later": float(bool(later)), "gap": float(later[0] - t) if later else -1.0, "productive_later": 0.0, "progress_after": 0.0, "fut_score": 0.0}
    if later:
        t2 = later[0]; ev = evidence(e, t2, c, e["steps"][t2]["next_obs"], obj)
        f["productive_later"] = float(ev["goal_revealed"] or ev["subgoal"]); f["noinfo_later"] = float(ev["no_info"])
        f["progress_after"] = float(max(m for _, m in mh[t2:]) > mh[t2][0])
        f["fut_score"] = 2.0 * ev["goal_revealed"] + 2.0 * ev["subgoal"] + min(ev["n_new"], 5) / 5.0 - 2.0 * ev["no_info"] + f["progress_after"]
    m = re.match(r"(?:go to|open) (.+)$", c)
    locs = {re.match(r"(?:go to|open) (.+)$", s["action"]).group(1) for s in e["steps"][t: t + 10] if re.match(r"(?:go to|open) (.+)$", s["action"])}
    f["loop_break"] = float(bool(m) and m.group(1) not in locs)
    return f


def candidates(arm, eps):
    tg = {(r["id"], r["t"]): r for r in L(SE / arm / "targets.jsonl")}
    ex = {(r["id"], r["t"]): r for r in L(SE / arm / "round1/examples.jsonl")}
    rows = []
    for r in L(SE / arm / "steps.jsonl"):
        k = (r["id"], r["steps"][0])
        if k not in tg: continue
        T = tg[k]; lp = T["teacher_logprobs"]; pl = T["plain_logprobs"]; ph, pp = softmax(lp), softmax(pl)
        alts = [c for c in sorted(lp, key=lambda c: -lp[c]) if c != T["a_minus"]][:2]
        e = eps[k[0]]; obj = parse_goal(e["goal"])["obj"]
        for rank, c in enumerate(alts):
            won = bool(ex[k]["alternatives"].get(c, False)) if k in ex else False
            hs = hindsight(e, k[1], c, obj)
            rows.append({"arm": arm, "id": k[0], "t": k[1], "cand": c, "won": won, "verified_state": k in ex, "rank": rank,
                         "hind_p": ph[c], "plain_p": pp[c], "hind_lp": lp[c], "plain_lp": pl[c], "H_hind": H(ph), "H_plain": H(pp),
                         "KL_hp": KL(ph, pp), "hind_minus_plain": math.log(ph[c] + 1e-12) - math.log(pp[c] + 1e-12),
                         "audit": {"supports": 1.0, "silent": 0.5, "n/a": 0.5, "contradicts": 0.0}[hs], "audit_str": hs, **future_feats(e, k[1], c, obj)})
    return rows


def cmd_onestep(a):
    from error_attr.env.alfworld_env import make_env
    eps = {e["id"]: e for e in L(TRAJ)}; PX.mkdir(parents=True, exist_ok=True)
    out = PX / (f"onestep_s{a.shard[0]}.jsonl" if a.shard else "onestep.jsonl")   # shards write their own file (Oct 6 14:40)
    done = set()
    for f_ in sorted(PX.glob("onestep*.jsonl")):
        for l in open(f_):
            try: r = json.loads(l); done.add((r["id"], r["t"], r["cand"]))
            except Exception: pass
    todo = {}
    for arm in a.arms:
        for r in candidates(arm, eps):
            k = (r["id"], r["t"], r["cand"])
            if k not in done: todo[k] = r
    if a.shard: todo = {k: v for j, (k, v) in enumerate(sorted(todo.items())) if j % a.shard[1] == a.shard[0]}
    print(f"{len(todo)} candidates to execute once", flush=True)
    if not todo: print("DONE onestep", flush=True); return
    with open(out, "a") as f:
        for n, (k, r) in enumerate(todo.items()):
            if a.limit and n >= a.limit: print(f"chunk limit {a.limit} reached, {len(todo) - n} left", flush=True); return   # make_env leaks ~50 MB per env
            e = eps[r["id"]]; env = make_env(e["game_file"], e["budget"] + 2)
            try:
                obs, info = env.reset()
                for s in e["steps"][: r["t"]]: obs, _, _, info = env.step([s["action"]])
                o, _, dn, info = env.step([r["cand"]])
                f.write(json.dumps({"id": r["id"], "t": r["t"], "cand": r["cand"], "obs_plus": o[0], "won_now": bool(info["won"][0])}) + "\n"); f.flush()
            finally:
                env.close()
            if n % 100 == 0: print(f"[{n}/{len(todo)}]", flush=True)
    print("DONE onestep", flush=True)


def cmd_cands(a):
    eps = {e["id"]: e for e in L(TRAJ)}; PX.mkdir(parents=True, exist_ok=True); seen = set()
    with open(PX / "cands.jsonl", "w") as f:
        for arm in a.arms:
            for r in candidates(arm, eps):
                k = (r["id"], r["t"], r["cand"])
                if k in seen: continue
                seen.add(k); f.write(json.dumps({"id": r["id"], "t": r["t"], "cand": r["cand"]}) + "\n")
    print(len(seen), "unique candidates", flush=True)


def within_state(rows, key, higher_better=True):
    by = {}
    for r in rows: by.setdefault((r["id"], r["t"]), []).append(r)
    ok = n = 0
    for v in by.values():
        if len(v) != 2 or v[0]["won"] == v[1]["won"]: continue
        w = v[0] if v[0]["won"] else v[1]; l = v[1] if v[0]["won"] else v[0]
        if w[key] == l[key]: ok += 0.5
        elif (w[key] > l[key]) == higher_better: ok += 1
        n += 1
    return ok / n if n else float("nan"), n


def cmd_auc(a):
    eps = {e["id"]: e for e in L(TRAJ)}
    one = {(r["id"], r["t"], r["cand"]): r for r in L(PX / "onestep.jsonl")}
    det = {(r["id"], r["t"], r["cand"]): r["p_necessary"] for r in L(PX / "detector_cands.jsonl")}
    arms = [x for x in a.arms if arm_done(x)]; allrows = []
    for arm in arms:
        rows = candidates(arm, eps)
        for r in rows:
            e = eps[r["id"]]; obj = parse_goal(e["goal"])["obj"]; o = one.get((r["id"], r["t"], r["cand"]))
            if o:
                ev = evidence(e, r["t"], r["cand"], o["obs_plus"], obj)
                r.update({"n_new": ev["n_new"], "goal_revealed": float(ev["goal_revealed"]), "subgoal": float(ev["subgoal"]), "no_info": ev["no_info"],
                          "env_score": 2.0 * ev["goal_revealed"] + 2.0 * ev["subgoal"] + min(ev["n_new"], 5) / 5.0 - 2.0 * ev["no_info"]})
            if (r["id"], r["t"], r["cand"]) in det: r["p_necessary"] = det[(r["id"], r["t"], r["cand"])]
        allrows += rows
    proxies = [("hind_p", True), ("plain_p", True), ("hind_minus_plain", True), ("rank", False), ("H_hind", True), ("H_plain", True), ("KL_hp", True),
               ("audit", True), ("appears_later", True), ("gap", True), ("productive_later", True), ("progress_after", True), ("fut_score", True), ("loop_break", True),
               ("n_new", True), ("goal_revealed", True), ("env_score", True), ("p_necessary", True)]
    def report(name, rows):
        if not rows: return
        print(f"\n[{name}] {len(rows)} candidates, {sum(r['won'] for r in rows)} won ({sum(r['won'] for r in rows) / len(rows):.1%}); "
              f"{len({(r['id'], r['t']) for r in rows if r['verified_state']})} states with a winner")
        print(f"  {'proxy':18s} {'AUC':>6s} {'within-state acc (n)':>22s}")
        for key, hb in proxies:
            rr = [r for r in rows if key in r]
            if not rr: continue
            y = [r["won"] for r in rr]; s = [r[key] if hb else -r[key] for r in rr]
            w, n = within_state(rr, key, hb)
            print(f"  {key:18s} {auc(y, s):6.3f} {w:14.3f} ({n:3d})" + ("" if len(rr) == len(rows) else f"   [{len(rr)} scored]"))
        rr = [r for r in rows if "no_info" in r]
        if rr:
            keep = [r for r in rr if not r["no_info"]]; W = [r for r in rr if r["won"]]
            print(f"  veto 'no_info': keeps {len(keep)}/{len(rr)} candidates ({len(keep) / len(rr):.0%}), recall of winners {sum(not r['no_info'] for r in W)}/{len(W)}"
                  f" | win rate kept {sum(r['won'] for r in keep) / max(1, len(keep)):.1%} vs vetoed {sum(r['won'] and r['no_info'] for r in rr) / max(1, len(rr) - len(keep)):.1%}")
    for arm in arms: report(arm, [r for r in allrows if r["arm"] == arm])
    report("pooled " + "+".join(arms), allrows)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("onestep", "cands", "auc"):
        s = sub.add_parser(n); s.add_argument("--arms", nargs="+", default=ARMS)
        if n == "onestep": s.add_argument("--limit", type=int, default=0, help="stop after this many new executions (re-run to resume)"); s.add_argument("--shard", type=int, nargs=2, default=None)
    a = ap.parse_args(); {"onestep": cmd_onestep, "cands": cmd_cands, "auc": cmd_auc}[a.cmd](a)


if __name__ == "__main__":
    main()
