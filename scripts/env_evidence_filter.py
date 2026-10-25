"""Environment-evidence filter for teacher corrections (no student rollout), evaluated against full verification.
For every round-1 attr_nofb pointed state and each teacher candidate (top-2 excluding the student's action a-):
  onestep  -- (CPU) replay the logged prefix, execute the candidate ONE step, record the observation o+;
              the student's own one-step observation o- is already in the log
  analyze  -- features and their power to predict (a) candidate wins under full verification, (b) causal fix
              (candidate wins AND a- loses), (c) luck (a- also wins):
              one-step evidence : state changed / new entities revealed / goal object revealed / subgoal event, vs o-
              hindsight audit   : the failed trajectory's own future (steps > t) contradicts / supports / is silent on
                                  the candidate (e.g. "go to L" when the student later found L empty of the goal object)
              distributions     : student & teacher entropy at the state, KL(teacher||student), KL(student||teacher),
                                  student / teacher probability of the candidate and of a-
"""
import argparse, json, math, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
A = ROOT / "reports/alfworld/opd/student3b_low100/attr_nofb"
TRAJ = ROOT / "reports/alfworld/student3b_low100_train_b50.jsonl"
OUT = ROOT / "reports/alfworld/env_evidence"


def L(p):
    return [json.loads(l) for l in open(p)]


CF_OVERRIDE = None   # set by --cf: counterfactual file giving a-'s outcome (round-2 data)
TEACHER_DIR = None   # set by --teacher_dir: another teacher's targets + verification on a subset of the same states


def candidates():
    cf = {(r["id"], r["t"]): r for r in L(A / "round1/counterfactual.jsonl")}
    rx = {(r["id"], r["t"]): r for r in L(ROOT / "reports/alfworld/relaxed_verify/attr_nofb_r1_skipped.jsonl")}
    orig_of = {**{k: v["orig"]["won"] for k, v in rx.items()}, **{k: v["a_minus_won"] for k, v in cf.items()}}
    if CF_OVERRIDE is not None: orig_of = {(r["id"], r["t"]): r["a_minus_won"] for r in L(CF_OVERRIDE)}
    D = TEACHER_DIR or A
    tg = {(r["id"], r["t"]): r for r in L(D / "targets.jsonl")}
    ex = {(r["id"], r["t"]): r for r in L(D / "round1/examples.jsonl")}
    rows = []
    for r in L(D / "steps.jsonl"):
        k = (r["id"], r["steps"][0]); T = tg[k]; lp = T["teacher_logprobs"]
        alts = [c for c in sorted(lp, key=lambda c: -lp[c]) if c != T["a_minus"]][:2]
        orig = orig_of.get(k)
        if k in ex:
            won = ex[k]["alternatives"]
        elif TEACHER_DIR is not None:
            won = {c: False for c in alts}   # skipped by verification: no candidate won
        else:
            won = {rx[k][f"alt{j + 1}"]["action"]: rx[k][f"alt{j + 1}"]["won"] for j in range(2) if f"alt{j + 1}" in rx[k]}
        for j, c in enumerate(alts):
            if c in won:
                rows.append({"id": k[0], "t": k[1], "rank": j, "cand": c, "a_minus": T["a_minus"], "won": bool(won[c]), "orig_won": orig})
    return rows, tg


def cmd_onestep(a):
    import signal
    from error_attr.env.alfworld_env import make_env
    eps = {e["id"]: e for e in L(TRAJ)}; rows, _ = candidates(); OUT.mkdir(parents=True, exist_ok=True)
    out = (TEACHER_DIR or OUT) / "onestep.jsonl"; done = {(json.loads(l)["id"], json.loads(l)["t"], json.loads(l)["cand"]) for l in open(out)} if out.exists() else set()
    with open(out, "a") as f:
        for k, r in enumerate(rows):
            if (r["id"], r["t"], r["cand"]) in done: continue
            e = eps[r["id"]]; env = make_env(e["game_file"], e["budget"] + 2)
            try:
                obs, info = env.reset()
                for s in e["steps"][: r["t"]]: obs, _, _, info = env.step([s["action"]])
                o, _, dn, info = env.step([r["cand"]])
                f.write(json.dumps({"id": r["id"], "t": r["t"], "cand": r["cand"], "obs_plus": o[0], "won_now": bool(info["won"][0])}) + "\n"); f.flush()
            finally:
                env.close()
            if k % 50 == 0: print(f"[{k}/{len(rows)}]", flush=True)
    print("DONE onestep", flush=True)


ENT = re.compile(r"\b([a-z]+ \d+)\b")


def ents(o):
    return set(ENT.findall(o.lower()))


def softmax(lp):
    z = max(lp.values()); Z = z + math.log(sum(math.exp(v - z) for v in lp.values())); return {c: math.exp(v - Z) for c, v in lp.items()}


def H(p):
    return -sum(q * math.log(q + 1e-12) for q in p.values())


def KL(p, q):
    return sum(pv * (math.log(pv + 1e-12) - math.log(q.get(c, 1e-12) + 1e-12)) for c, pv in p.items())


def evidence(e, t, act, obs_after, obj):
    """One-step evidence of an action from its resulting observation, relative to everything seen up to step t."""
    known = ents(e["initial_obs"])
    for s in e["steps"][:t]: known |= ents(s["obs"]) | ents(s["next_obs"])
    ol = obs_after.lower(); new = ents(obs_after) - known
    # potential-based progress must not reward UNDOING: picking an object back up after the agent itself put it down earlier in
    # this trajectory returns to a previous state (Oct 5 22:00: such 'take X back' corrections ranked 101-205 and taught take/put loops)
    pm = re.match(rf"you pick up the ({obj} \d+)", ol); undo = False
    if pm:
        inst = pm.group(1); undo = any(f"you put the {inst}" in s2["next_obs"].lower() or f"you move the {inst}" in s2["next_obs"].lower() for s2 in e["steps"][:t])
    return {"nothing_happens": ol.startswith("nothing happens"), "n_new": len(new), "undo": undo,
            "goal_revealed": any(x.split()[0] == obj for x in new),
            "subgoal": (bool(re.match(rf"you (pick up|clean|heat|cool|move) the {obj}\b", ol)) and not undo) or ol.startswith("you turn on the desklamp"),
            "no_info": ol.startswith("nothing happens") or (len(new) == 0 and not re.match(r"you (pick up|clean|heat|cool|move|open|turn on)", ol))}


def hindsight(e, t, act, obj):
    """What the failed trajectory's own future (steps >= t) says about the candidate's target."""
    m = re.match(r"(?:go to|open) (.+)$", act)
    if not m: return "n/a"
    loc = m.group(1); seen_there = None
    for s in e["steps"][t:]:
        a2 = s["action"]
        if a2 in (f"go to {loc}", f"open {loc}"):
            objs = {x.split()[0] for x in ents(s["next_obs"])}
            if "closed" in s["next_obs"].lower() and a2.startswith("go to"): continue
            seen_there = obj in objs if seen_there is None else (seen_there or obj in objs)
    if seen_there is None: return "silent"
    return "supports" if seen_there else "contradicts"


def auc(y, s):
    P = [v for v, l in zip(s, y) if l]; N = [v for v, l in zip(s, y) if not l]
    if not P or not N: return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in P for n in N) / (len(P) * len(N))


def cmd_analyze(a):
    from alfworld_relaxed_verify import parse_goal
    eps = {e["id"]: e for e in L(TRAJ)}; rows, tg = candidates()
    one = {(r["id"], r["t"], r["cand"]): r for r in L(OUT / "onestep.jsonl")} if TEACHER_DIR is None or TRAJ == ROOT / "reports/alfworld/student3b_low100_train_b50.jsonl" else {}
    if TEACHER_DIR is not None and (TEACHER_DIR / "onestep.jsonl").exists():
        one.update({(r["id"], r["t"], r["cand"]): r for r in L(TEACHER_DIR / "onestep.jsonl")})
    sp = (TEACHER_DIR / "student_dists.jsonl") if TEACHER_DIR is not None and (TEACHER_DIR / "student_dists.jsonl").exists() else OUT / "student_dists.jsonl"
    stu = {(r["id"], r["t"]): r["student_logprobs"] for r in L(sp)} if sp.exists() else {}
    feats = []
    for r in rows:
        k = (r["id"], r["t"]); e = eps[r["id"]]; obj = parse_goal(e["goal"])["obj"]
        if (r["id"], r["t"], r["cand"]) not in one: continue
        pt = softmax(tg[k]["teacher_logprobs"]); ps = softmax({c: v for c, v in stu[k].items() if c in pt}) if k in stu else None
        ev_p = evidence(e, r["t"], r["cand"], one[(r["id"], r["t"], r["cand"])]["obs_plus"], obj)
        ev_m = evidence(e, r["t"], r["a_minus"], e["steps"][r["t"]]["next_obs"], obj)
        hs = hindsight(e, r["t"] + 1, r["cand"], obj)
        feats.append({**r, "causal": bool(r["won"] and r["orig_won"] is False),
                      "plus_no_info": ev_p["no_info"], "plus_n_new": ev_p["n_new"], "plus_goal_revealed": ev_p["goal_revealed"], "plus_subgoal": ev_p["subgoal"],
                      "minus_n_new": ev_m["n_new"], "minus_goal_revealed": ev_m["goal_revealed"], "gain_new": ev_p["n_new"] - ev_m["n_new"],
                      "hindsight": hs, "H_teacher": H(pt), "p_t_cand": pt[r["cand"]], "p_t_minus": pt.get(r["a_minus"], 0),
                      **({"H_student": H(ps), "KL_ts": KL(pt, ps), "KL_st": KL(ps, pt), "p_s_cand": ps.get(r["cand"], 0), "p_s_minus": ps.get(r["a_minus"], 0)} if ps else {})})
    json.dump(feats, open((TEACHER_DIR or OUT) / "features.json", "w"))
    n = len(feats); y = [f["won"] for f in feats]; yc = [f["causal"] for f in feats if f["orig_won"] is not None]
    print(f"candidates {n} (states {len({(f['id'], f['t']) for f in feats})}); win rate {sum(y) / n:.0%}; causal among labelled {sum(yc) / len(yc):.0%}")
    print("\n-- single-feature AUC for 'candidate wins' / 'causal fix' (higher score -> predicted better)")
    sub = [f for f in feats if f["orig_won"] is not None]
    for name, fn in [("one-step: not no-info", lambda f: -f["plus_no_info"]), ("one-step: #new entities", lambda f: f["plus_n_new"]),
                     ("one-step: new entities vs a-", lambda f: f["gain_new"]), ("one-step: goal object revealed", lambda f: f["plus_goal_revealed"]),
                     ("hindsight: not contradicted", lambda f: {"contradicts": 0, "silent": 1, "n/a": 1, "supports": 2}[f["hindsight"]]),
                     ("teacher p(cand)", lambda f: f["p_t_cand"]), ("student p(cand)", lambda f: f["p_s_cand"]),
                     ("-student entropy", lambda f: -f["H_student"]), ("-teacher entropy", lambda f: -f["H_teacher"]),
                     ("KL(teacher||student)", lambda f: f["KL_ts"]), ("KL(student||teacher)", lambda f: f["KL_st"]),
                     ("student p(a-)", lambda f: f["p_s_minus"]), ("-position t", lambda f: -f["t"])]:
        try: [fn(f) for f in feats]
        except KeyError: print(f"  {name:32s} (student distributions not computed yet)"); continue
        print(f"  {name:32s} wins {auc(y, [fn(f) for f in feats]):.3f} | causal {auc([f['causal'] for f in sub], [fn(f) for f in sub]):.3f} | luck(a- wins) {auc([bool(f['orig_won']) for f in sub], [fn(f) for f in sub]):.3f}")
    print("\n-- groups")
    def grp(name, v):
        if not v: return
        c = [f for f in v if f["orig_won"] is not None]
        print(f"  {name:44s} n={len(v):3d} wins {sum(f['won'] for f in v) / len(v):4.0%} | causal {sum(f['causal'] for f in c) / max(1, len(c)):4.0%} | a- also wins {sum(bool(f['orig_won']) for f in c) / max(1, len(c)):4.0%}")
    grp("one-step: no new information", [f for f in feats if f["plus_no_info"]])
    grp("one-step: new information", [f for f in feats if not f["plus_no_info"]])
    grp("one-step: goal object revealed", [f for f in feats if f["plus_goal_revealed"]])
    for h in ["contradicts", "silent", "supports", "n/a"]: grp(f"hindsight: {h}", [f for f in feats if f["hindsight"] == h])
    keep = [f for f in feats if not f["plus_no_info"] and f["hindsight"] != "contradicts"]
    grp("FILTER: new info AND not contradicted", keep); grp("rejected by filter", [f for f in feats if f not in keep])
    tp = sum(f["won"] for f in keep); print(f"\n  filter vs full verification: keeps {len(keep)}/{n}; precision {tp / max(1, len(keep)):.0%} (base {sum(y) / n:.0%}); recall {tp / max(1, sum(y)):.0%}; env steps per candidate 1 vs full rollout")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    ap.add_argument("--teacher_dir", default=None); ap.add_argument("--traj", default=None); ap.add_argument("--cf", default=None)
    sub.add_parser("onestep"); sub.add_parser("analyze"); sub.add_parser("states")
    a = ap.parse_args()
    global TEACHER_DIR, TRAJ, CF_OVERRIDE
    if a.teacher_dir: TEACHER_DIR = ROOT / a.teacher_dir
    if a.traj: TRAJ = ROOT / a.traj
    if a.cf: CF_OVERRIDE = ROOT / a.cf
    if a.cmd == "states":
        rows, _ = candidates(); st = sorted({(r["id"], r["t"]) for r in rows}); D = TEACHER_DIR or OUT; D.mkdir(parents=True, exist_ok=True)
        json.dump([{"id": i, "t": t} for i, t in st], open(D / "states.json", "w")); print(len(st), "states")
    else:
        {"onestep": cmd_onestep, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
