"""Relaxed verification diagnostic. For states where NO candidate won under strict verification, continue greedily
(start student, cap 20 steps) after the original action and after each of the teacher's top-2 alternatives, and
score how far each continuation got toward the goal:
  milestone -- planner-free: ordered goal milestones read off the environment's own feedback text
               (picked up the goal object -> applied the required clean/heat/cool/lamp -> placed it in the goal receptacle;
               two-object tasks count both objects)
  planner   -- ruler only: the PDDL planner's remaining plan length after the continuation (lower = closer)
  rollout   -- GPU: record the continuations and milestone scores
  planner   -- CPU: add the planner distance for each recorded continuation
"""
import argparse, json, re, signal, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

OBJ_WORDS = r"([a-z]+)"


def parse_goal(goal):
    g = goal.lower().strip().rstrip(".")
    lamp = "desklamp" in g or "lamp" in g
    tf = next((w for w in ("clean", "heat", "cool") if w in g or {"clean": "clean", "heat": "hot", "cool": "cool"}[w] in g), None)
    if "hot " in g: tf = "heat"
    two = g.startswith("put two") or g.startswith("find two")
    obj = rec = None
    if lamp:
        m = re.search(r"\b(?:look at|examine)\s+(?:(?:the|a|an|some)\s+)?([a-z]+)", g)
    else:
        m = re.search(r"\b(?:some|a|an|two|the)\s+(?:(?:clean|hot|cool)\s+)?([a-z]+)", g)
    if m: obj = m.group(1)
    m = re.search(r"(?:in|on|into|onto)\s+(?:the\s+)?([a-z]+)$", g) or re.search(r"(?:in|on)\s+([a-z]+)", g.split(" and put ")[-1] if " and put " in g else g)
    if m and not lamp: rec = m.group(1)
    if obj and obj.endswith("s") and two: obj = obj[:-1]
    return {"obj": obj, "rec": rec, "tf": None if lamp else tf, "two": two, "lamp": lamp}


def milestone_score(goal, next_obs_list):
    """Ordered progress: per object, stages = holding (1), transformed/lamp (1 if required), placed (1)."""
    G = parse_goal(goal); obj, rec, tf = G["obj"], G["rec"], G["tf"]
    if not obj: return None, None
    per_obj = 3 if (tf or G["lamp"]) else 2
    need = 2 if G["two"] else 1
    placed = 0; holding = False; transformed = False; lamp_on = False
    for o in next_obs_list:
        o = o.lower()
        if G["lamp"] and o.startswith("you turn on the desklamp"): lamp_on = True
        if re.match(rf"you pick up the {obj}\b", o): holding = True; transformed = False
        elif tf and re.match(rf"you {tf} the {obj}\b", o) and holding: transformed = True
        if G["lamp"] and lamp_on and holding: placed = max(placed, 1); transformed = True
        elif re.match(rf"you move the {obj}\b", o):
            ok = rec is not None and re.search(rf"to the {rec}\b", o) and (transformed or not tf)
            if ok: placed += 1
            holding = False; transformed = False
    placed = min(placed, need)
    cur = 0 if placed >= need else (1 if holding else 0) + (1 if (holding and transformed and (tf or G["lamp"])) else 0)
    return placed * per_obj + cur, need * per_obj


def rollout(ep, t, action, pol, budget, max_steps):
    from error_attr.env.alfworld_env import make_env
    from error_attr.policy.alfworld_agent import InteractionStep
    from alfworld_attr_opd import history_at
    def _alarm(s, f): raise TimeoutError()
    signal.signal(signal.SIGALRM, _alarm); signal.alarm(300)
    env = make_env(ep["game_file"], budget + 1)
    actions = [s["action"] for s in ep["steps"][:t]]; nobs = [s["next_obs"] for s in ep["steps"][:t]]; won = False
    try:
        obs, info = env.reset()
        for a in actions: obs, rw, dn, info = env.step([a])
        obs, rw, dn, info = env.step([action]); actions.append(action); nobs.append(obs[0])
        if dn[0]: return bool(info["won"][0]), actions, nobs
        h = history_at(ep, t); h.steps[-1].action = action
        h.steps.append(InteractionStep(observation=obs[0], admissible_commands=list(info["admissible_commands"][0])))
        k = t + 1; limit = min(budget, t + 1 + max_steps)
        while k < limit:
            a = pol.act(h, temperature=0.0); h.steps[-1].action = a
            obs, rw, dn, info = env.step([a]); actions.append(a); nobs.append(obs[0]); k += 1
            if dn[0]: won = bool(info["won"][0]); break
            h.steps.append(InteractionStep(observation=obs[0], admissible_commands=list(info["admissible_commands"][0])))
        return won, actions, nobs
    except TimeoutError:
        return False, actions, nobs
    finally:
        signal.alarm(0)
        try: env.close()
        except Exception: pass


def cmd_rollout(a):
    import torch
    from alfworld_attr_opd import load_jsonl
    from alfworld_opd_plus import load_policy
    arm = ROOT / "reports/alfworld/opd/student3b_low100" / a.arm
    eps = {e["id"]: e for e in load_jsonl(ROOT / a.traj)}
    tg = {(r["id"], r["t"]): r for r in load_jsonl(arm / a.targets)}
    sk = [r for r in load_jsonl(arm / a.round / "skipped.jsonl") if r.get("reason") != "a_minus_recovers"]
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    done = {(json.loads(l)["id"], json.loads(l)["t"]) for l in open(out)} if out.exists() else set()
    sk = [r for r in sk if (r["id"], r["t"]) not in done]; print(f"{len(sk)} skipped states to roll out", flush=True)
    tok, model, pol = load_policy(a.student, a.adapter)
    with open(out, "a") as f, torch.no_grad():
        for k, r in enumerate(sk):
            e = eps[r["id"]]; t = tg[(r["id"], r["t"])]
            order = sorted(t["admissible"], key=lambda c: -t["teacher_logprobs"][c]); alts = [c for c in order if c != t["a_minus"]][:2]
            res = {}
            for role, act in [("orig", t["a_minus"])] + [(f"alt{j + 1}", c) for j, c in enumerate(alts)]:
                won, acts, nobs = rollout(e, r["t"], act, pol, e["budget"], a.max_continue_steps)
                sc, mx = milestone_score(e["goal"], nobs)
                res[role] = {"action": act, "won": won, "milestone": sc, "milestone_max": mx, "actions": acts}
            f.write(json.dumps({"id": r["id"], "t": r["t"], "goal": e["goal"], "game_file": e["game_file"], "budget": e["budget"], **res}) + "\n"); f.flush()
            if k % 10 == 0: print(f"[{k + 1}/{len(sk)}]", flush=True)
    print("DONE rollout", flush=True)


def cmd_planner(a):
    from error_attr.attribution.oracle_alfworld import progress_trace
    rows = [json.loads(l) for l in open(ROOT / a.inp)]
    with open(ROOT / a.out, "w") as f:
        for k, r in enumerate(rows):
            for role in ("orig", "alt1", "alt2"):
                if role in r:
                    try:
                        pt = progress_trace(r["game_file"], r[role]["actions"], r["budget"]); r[role]["planner_dist_end"] = pt["plan_len"][-1]
                    except Exception as ex:
                        r[role]["planner_dist_end"] = None
            f.write(json.dumps(r) + "\n"); f.flush()
            if k % 20 == 0: print(f"[{k + 1}/{len(rows)}]", flush=True)
    print("DONE planner", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rollout"); r.add_argument("--arm", default="attr_nofb"); r.add_argument("--round", default="round1"); r.add_argument("--targets", default="targets.jsonl")
    r.add_argument("--traj", default="reports/alfworld/student3b_low100_train_b50.jsonl"); r.add_argument("--student", default="Qwen/Qwen2.5-3B-Instruct")
    r.add_argument("--adapter", default=str(ROOT / "reports/alfworld/student3b_low100/final")); r.add_argument("--max_continue_steps", type=int, default=20)
    r.add_argument("--out", default="reports/alfworld/relaxed_verify/attr_nofb_r1_skipped.jsonl")
    p = sub.add_parser("planner"); p.add_argument("--inp", required=True); p.add_argument("--out", required=True)
    a = ap.parse_args(); cmd_rollout(a) if a.cmd == "rollout" else cmd_planner(a)


if __name__ == "__main__":
    main()
