"""Proxy-selected (NOT rollout-verified) self-correction targets for the 3B self-evolve states.
Hypothesis: a proxy's pick need not win the student's rollout, yet may be the better training signal.
At each pointer state the hindsight self-teacher's two candidates (a- excluded) are judged by a proxy; the chosen one becomes a
one-hot KL target (same trainer / lr as the verified and placebo arms).  Rules:
  env_filter  one-step env evidence: veto 'no_info' candidates, pick the higher env_score; keep the state only if env_score > 0
  env_top     as env_filter, then the K states with the highest env_score (tie: hindsight prob)      -- matched budget
  hind_top    self-teacher confidence: top-1 candidate, K states with the highest hindsight prob       -- confidence proxy
  all_top1    top-1 candidate at every state, no filter                                                -- unverified baseline
  det_top     detour detector: candidate with the higher P(necessary), K states with the highest P      -- learned proxy
  det_filter  detector pick at every state where P(necessary) > 0.5
The rollout outcome of every candidate is known (round1/examples.jsonl), so each set also reports how many of its targets would
have passed verification.  Usage: python scripts/proxy_arms_se3b.py --arm attr --rule env_filter [--k 86]
"""
import argparse, json, math, os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from proxy_eval_se3b import candidates, L, SE, PX, TRAJ
from env_evidence_filter import evidence
from alfworld_relaxed_verify import parse_goal

ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="attr"); ap.add_argument("--rule", required=True); ap.add_argument("--k", type=int, default=86)
ap.add_argument("--out", default=None); a = ap.parse_args()
eps = {e["id"]: e for e in L(TRAJ)}; rows = candidates(a.arm, eps)
one = {(r["id"], r["t"], r["cand"]): r for r in L(PX / "onestep.jsonl")}
det = {(r["id"], r["t"], r["cand"]): r["p_necessary"] for r in L(PX / "detector_cands.jsonl")}
tg = {(r["id"], r["t"]): r for r in L(SE / a.arm / "targets.jsonl")}
for r in rows:
    k = (r["id"], r["t"], r["cand"]); e = eps[r["id"]]
    if k in one:
        if os.environ.get("EVIDENCE") == "generic":   # one generic template for every environment (Oct 5 23:10)
            from generic_evidence import alfworld_step
            gv = alfworld_step(e, r["t"], one[k]["obs_plus"]); r["no_info"] = gv["no_info"]; r["env_score"] = gv["score"]
        else:
            ev = evidence(e, r["t"], r["cand"], one[k]["obs_plus"], parse_goal(e["goal"])["obj"]); r["no_info"] = ev["no_info"]
            r["env_score"] = 2.0 * ev["goal_revealed"] + 2.0 * ev["subgoal"] + min(ev["n_new"], 5) / 5.0 - 2.0 * ev["no_info"]
    if k in det: r["p_necessary"] = det[k]

def _known(e, t):
    from env_evidence_filter import ents
    k = ents(e["initial_obs"])
    for s in e["steps"][:t]: k |= ents(s["obs"]) | ents(s["next_obs"])
    return k


def achieved_delay(e, t, c, obs_plus, obj):
    """Layer 1 (state level): how many steps after t the failed episode's own future first achieves what candidate c achieves
    NOW (reveals the goal object / completes the subgoal / shows the same new entities).  None = never (the error at t was
    terminal), 0 = c brings nothing to achieve."""
    import re
    from env_evidence_filter import ents
    ev = evidence(e, t, c, obs_plus, obj); new = ents(obs_plus) - _known(e, t)
    if not (ev["goal_revealed"] or ev["subgoal"] or new): return 0
    for j in range(t + 1, len(e["steps"])):
        o = e["steps"][j]["next_obs"]; ol = o.lower()
        if ev["goal_revealed"] and any(x.split()[0] == obj for x in ents(o)): return j - t
        if ev["subgoal"] and (re.match(rf"you (pick up|clean|heat|cool|move) the {obj}\b", ol) or ol.startswith("you turn on")): return j - t
        if not ev["goal_revealed"] and not ev["subgoal"] and ents(o) & new: return j - t
    return None


def decisive_weight(delay):
    """never achieved 1.0 | >=10 steps later 0.7 | 3-9 later 0.4 | 1-2 later 0.1 (the agent corrected itself at once) | nothing 0."""
    if delay is None: return 1.0
    if delay == 0: return 0.0
    return 0.7 if delay >= 10 else 0.4 if delay >= 3 else 0.1

by = {}
for r in rows: by.setdefault((r["id"], r["t"]), []).append(r)
picks = []   # (state_key, chosen row, ranking score)
def _taken_score(key):   # one-step evidence of the action actually taken at the state (its observed next observation)
    e = eps[key[0]]; t = key[1]
    if os.environ.get("EVIDENCE") == "generic":
        from generic_evidence import alfworld_step
        return alfworld_step(e, t, e["steps"][t]["next_obs"])["score"]
    ev = evidence(e, t, e["steps"][t]["action"], e["steps"][t]["next_obs"], parse_goal(e["goal"])["obj"])
    return 2.0 * ev["goal_revealed"] + 2.0 * ev["subgoal"] + min(ev["n_new"], 5) / 5.0 - 2.0 * ev["no_info"]
for key, cs in by.items():
    if a.rule in ("env_filter", "env_top", "env_pool", "env_top_margin"):
        ok = [c for c in cs if "env_score" in c and not c["no_info"] and c["env_score"] > 0]
        if a.rule == "env_top_margin":   # Oct 9 (user): a correction must STRICTLY improve the one-step evidence over the taken action
            ft = _taken_score(key); ok = [c for c in ok if c["env_score"] > ft]
        if not ok: continue
        c = max(ok, key=lambda c: (c["env_score"], c["hind_p"])); picks.append((key, c, (c["env_score"], c["hind_p"])))
    elif a.rule in ("hind_top", "all_top1"):
        c = max(cs, key=lambda c: c["hind_p"]); picks.append((key, c, (c["hind_p"],)))
    elif a.rule in ("fut_filter", "fut_top", "fut_pass", "fut_fail"):
        # the failed log's own future as evidence: the agent eventually took the candidate, it was productive then, not contradicted
        ok = [c for c in cs if c["appears_later"] and c["productive_later"] and c["audit_str"] != "contradicts"]
        if not ok: continue
        c = max(ok, key=lambda c: (c["fut_score"], c["hind_p"]))
        # fut_pass / fut_fail split the fut_filter set by the (otherwise unused) rollout outcome: is the half that would FAIL
        # verification useful training signal on its own?
        if (a.rule == "fut_pass" and not c["won"]) or (a.rule == "fut_fail" and c["won"]): continue
        picks.append((key, c, (c["fut_score"], c["hind_p"])))
    elif a.rule in ("kl_top", "ent_top"):
        # state-level signals (identical for both candidates): KL(hindsight || plain) = how much seeing the full failed trajectory
        # changed the student's mind here; H_plain = how uncertain the plain student is here.  Action = hindsight top-1.
        c = max(cs, key=lambda c: c["hind_p"]); s = c["KL_hp"] if a.rule == "kl_top" else c["H_plain"]
        picks.append((key, c, (s, c["hind_p"])))
    elif a.rule in ("combo_top", "combo_filter"):
        # one-step env evidence + the failed log's future + hindsight audit; veto no_info and contradicted candidates
        ok = [c for c in cs if "env_score" in c and not c["no_info"] and c["audit_str"] != "contradicts"]
        if not ok: continue
        sc = lambda c: c["env_score"] + c["fut_score"] + 2.0 * (c["audit"] - 0.5)
        c = max(ok, key=lambda c: (sc(c), c["hind_p"]))
        if sc(c) <= 0: continue
        picks.append((key, c, (sc(c), c["hind_p"])))
    elif a.rule in ("det_top", "det_filter"):
        ok = [c for c in cs if "p_necessary" in c]
        if not ok: continue
        c = max(ok, key=lambda c: c["p_necessary"])
        if a.rule == "det_filter" and c["p_necessary"] <= 0.5: continue
        picks.append((key, c, (c["p_necessary"], c["hind_p"])))
    elif a.rule in ("two_layer", "fut_veto_env", "env_tie_fut", "wide_fut"):
        # layer 2 (action level) = one-step env evidence picks the action; layer 1 (state level) = the failed log's future says
        # whether the error at t was decisive (weight) and vetoes candidates the future contradicts
        ok = [c for c in cs if "env_score" in c and not c["no_info"] and c["audit_str"] != "contradicts" and c["env_score"] > 0]
        if not ok: continue
        c = max(ok, key=lambda c: (c["env_score"], c["fut_score"] if a.rule == "env_tie_fut" else c["hind_p"]))
        e = eps[c["id"]]; w = decisive_weight(achieved_delay(e, c["t"], c["cand"], one[(c["id"], c["t"], c["cand"])]["obs_plus"], parse_goal(e["goal"])["obj"]))
        c["decisive_w"] = w
        if a.rule == "two_layer":
            if w == 0: continue
            picks.append((key, c, (w * (1.0 + c["env_score"] / 4.0), c["env_score"])))
        elif a.rule == "wide_fut": picks.append((key, c, (c["env_score"], c["hind_p"])))
        else: picks.append((key, c, (c["env_score"], c["fut_score"] if a.rule == "env_tie_fut" else c["hind_p"])))
    elif a.rule in ("generic_top",):
        # de-tasked one-step evidence (no task parsing): new entities, overlap of the new observation with the goal TEXT, whether the
        # environment acknowledged a state change; veto no-info.  Tests whether env_top's effect depends on the ALFWorld-specific
        # goal_revealed/subgoal features.
        import re as _re
        from env_evidence_filter import ents as _ents
        def gscore(c):
            e = eps[c["id"]]; t = c["t"]; o = one[(c["id"], t, c["cand"])]["obs_plus"]; ol = o.lower()
            known = _known(e, t); new = _ents(o) - known
            gw = set(_re.findall(r"[a-z]+", e["goal"].lower())) - {"a", "an", "the", "in", "on", "some", "and", "put", "two", "with", "to", "of", "at", "look", "under"}
            overlap = any(x.split()[0] in gw for x in new)
            state_change = ol.startswith("you ") and not ol.startswith("you see") and not ol.startswith("you arrive")
            no_info = ol.startswith("nothing happens") or (len(new) == 0 and not state_change)
            return 2.0 * overlap + min(len(new), 5) / 5.0 - 2.0 * no_info, no_info
        ok = []
        for c in cs:
            if (c["id"], c["t"], c["cand"]) not in one: continue
            sc, ni = gscore(c); c["generic_score"] = sc
            if not ni and sc > 0: ok.append(c)
        if not ok: continue
        c = max(ok, key=lambda c: (c["generic_score"], c["hind_p"])); picks.append((key, c, (c["generic_score"], c["hind_p"])))
    elif a.rule in ("env_unit", "env_lex", "env_ig"):
        # weight-robustness variants of the one-step evidence (same veto; only the aggregation changes):
        #   env_unit: all weights 1;  env_lex: lexicographic (progress, goal revealed, novelty), no numeric weights;
        #   env_ig: information gain in bits about the goal object's location under a uniform prior over the K receptacles not yet
        #           visited (reveal -> log2 K, visit without reveal -> log2(K/(K-1)), known place -> 0) + 1[progress]*log2 K
        import math as _m, re as _re
        from env_evidence_filter import ents as _ents
        def feats(c):
            e = eps[c["id"]]; t = c["t"]; o = one[(c["id"], t, c["cand"])]["obs_plus"]; obj = parse_goal(e["goal"])["obj"]
            ev = evidence(e, t, c["cand"], o, obj); known = _known(e, t)
            recs = {x for x in _ents(e["initial_obs"]) if x.split()[0] not in ("mug", "cup", "knife", "apple", "tomato", "book", "pen", "cd", "key", "pillow", "plate", "bowl")}
            visited = {_re.match(r"go to (.+)$", s_["action"]).group(1) for s_ in e["steps"][:t] if _re.match(r"go to (.+)$", s_["action"])}
            K = max(2, len(recs - visited)); m = _re.match(r"(?:go to|open) (.+)$", c["cand"])
            ig = (_m.log2(K) if ev["goal_revealed"] else (_m.log2(K / (K - 1)) if (m and m.group(1) not in visited) else 0.0))
            return ev, ig, K
        ok = [c for c in cs if (c["id"], c["t"], c["cand"]) in one and not c["no_info"]]
        if not ok: continue
        def sc(c):
            ev, ig, K = feats(c)
            if a.rule == "env_unit": return 1.0 * ev["goal_revealed"] + 1.0 * ev["subgoal"] + min(ev["n_new"], 5) / 5.0
            if a.rule == "env_lex": return (float(ev["subgoal"]), float(ev["goal_revealed"]), float(min(ev["n_new"], 5)))
            return ig + (_m.log2(K) if ev["subgoal"] else 0.0) + 0.1 * min(ev["n_new"], 5) / 5.0
        c = max(ok, key=lambda c: (sc(c), c["hind_p"])); v = sc(c)
        if (v if not isinstance(v, tuple) else sum(v)) <= 0: continue
        picks.append((key, c, (v, c["hind_p"])))
    else: raise SystemExit(f"unknown rule {a.rule}")
if a.rule == "env_pool":
    # ENVIRONMENT ADJUDICATION of the step (final method, Oct 6): each failure contributes several candidate states (first-step and
    # stall-entry proposals); keep, per failure, the state whose best verified alternative improves most over the action actually
    # taken (margin = env_score(best alternative) - env_score(taken)), then rank failures by the alternative's score.
    def _taken(key):
        e = eps[key[0]]; t = key[1]
        if os.environ.get("EVIDENCE") == "generic":
            from generic_evidence import alfworld_step
            return alfworld_step(e, t, e["steps"][t]["next_obs"])["score"]
        ev = evidence(e, t, e["steps"][t]["action"], e["steps"][t]["next_obs"], parse_goal(e["goal"])["obj"])
        return 2.0 * ev["goal_revealed"] + 2.0 * ev["subgoal"] + min(ev["n_new"], 5) / 5.0 - 2.0 * ev["no_info"]
    best = {}
    for key, cs in by.items():
        ok = [c for c in cs if "env_score" in c and not c["no_info"] and c["env_score"] > 0]
        if not ok: continue
        c = max(ok, key=lambda c: (c["env_score"], c["hind_p"])); m = c["env_score"] - _taken(key)
        if key[0] not in best or (m, c["env_score"]) > best[key[0]][2]: best[key[0]] = (key, c, (m, c["env_score"]))
    picks = sorted([(k, c, (sc[1], sc[0])) for k, c, sc in best.values()], key=lambda x: x[2], reverse=True)[: a.k]
if a.rule in ("env_top", "env_top_margin", "hind_top", "det_top", "fut_top", "combo_top", "kl_top", "ent_top", "two_layer", "fut_veto_env", "env_tie_fut", "generic_top", "env_unit", "env_lex", "env_ig"): picks = sorted(picks, key=lambda x: x[2], reverse=True)[: a.k]
if a.rule == "wide_fut":   # wide env_top (top 137) re-ranked by the future's decisiveness, then top-k
    picks = sorted(picks, key=lambda x: x[2], reverse=True)[: max(a.k, 137)]
    picks = sorted(picks, key=lambda x: (x[1]["decisive_w"], x[2]), reverse=True)[: a.k]
out = Path(a.out) if a.out else SE / "proxy_arms" / f"{a.arm}_{a.rule}"; out.mkdir(parents=True, exist_ok=True)
with open(out / "train.jsonl", "w") as f:
    for key, c, _ in picks:
        T = tg[key]; cmds = T["admissible"]
        f.write(json.dumps({"kind": "kl", "src": str(TRAJ.relative_to(ROOT)), "id": c["id"], "t": c["t"], "cmds": cmds, "target": [1.0 if x == c["cand"] else 0.0 for x in cmds], "cand": c["cand"]}) + "\n")
n = len(picks); won = sum(c["won"] for _, c, _ in picks); vs = sum(c["verified_state"] for _, c, _ in picks)
stats = {"arm": a.arm, "rule": a.rule, "n_targets": n, "rollout_won": won, "state_had_a_winner": vs, "mean_hind_p": sum(c["hind_p"] for _, c, _ in picks) / max(1, n),
         "n_states_total": len(by), "verified_total": len({(r["id"], r["t"]) for r in rows if r["verified_state"]})}
json.dump(stats, open(out / "stats.json", "w"), indent=1)
print(f"=== proxy set {a.arm}/{a.rule}: {n} targets; {won} ({won / max(1, n):.0%}) would pass rollout; {vs} at states where some candidate won; mean hindsight p {stats['mean_hind_p']:.2f}")
