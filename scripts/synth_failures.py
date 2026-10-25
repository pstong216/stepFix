"""Who&When-style hand-crafted failures, built from the student's OWN successful trajectories (no human, no planner):
take a won trajectory, replace one step by a plausible wrong action of a known type, let the current student continue (sampling)
from the corrupted state; keep the attempts that fail.  The injected step is the decisive error by construction -> training data
for a self-trained log-reading error attributor (alfworld_self_attributor.py: --labels <out>.labels.jsonl --trajs <out>).
Injection types: wrong_location (go to another receptacle), wrong_object (take something else), misorder (do the next step now,
e.g. place before cleaning / move before opening), misplace (put the object somewhere else), random (any other admissible action).
    python scripts/synth_failures.py --traj reports/alfworld/student3b_low100_train_b50.jsonl --adapter reports/alfworld/student3b_low100/final --n 500 --out reports/alfworld/self_attr_labels/r0/synth.jsonl
"""
import argparse, json, random, re, signal, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch
from alfworld_attr_opd import history_at, load_jsonl
from alfworld_opd_plus import load_policy
from error_attr.policy.alfworld_agent import InteractionStep


TYPES = ["wrong_location", "wrong_object", "misorder", "misplace", "random"]


def inject(a_t, a_next, adm, rng, lp=None, want=None):
    """Return (type, wrong action) or None.  Candidate sets per type; the injected action is the policy's MOST PLAUSIBLE wrong
    action of that type (highest log-prob among the type's candidates) when lp is given -- i.e. the student's own near-miss, as in
    AgenTracer / Who&When where an LLM writes a realistic error of a requested type -- else a random one.  `want` = requested type
    (types are cycled by the caller for balance); falls back to the other types in order, random last."""
    go = [c for c in adm if c.startswith("go to ") and c != a_t]
    take = [c for c in adm if c.startswith("take ") and c != a_t]
    put = [c for c in adm if (c.startswith("move ") or c.startswith("put ")) and c != a_t]
    others = [c for c in adm if c != a_t and c.split()[0] not in ("look", "inventory", "examine", "help")]
    cands = {"wrong_location": go if a_t.startswith("go to ") else [], "wrong_object": take if (take and (a_t.startswith("take ") or not a_t.startswith("go to "))) else [],
             "misorder": [a_next] if (a_next and a_next in adm and a_next != a_t) else [], "misplace": put if (a_t.startswith("move ") or a_t.startswith("put ")) else [], "random": others}
    order = ([want] if want else []) + [t for t in TYPES if t != want]
    for t in order:
        cs = cands.get(t) or []
        if not cs: continue
        pick = max(cs, key=lambda c: lp.get(c, -1e9)) if lp else rng.choice(cs)
        return (t, pick)
    return None


def corrupt_and_continue(e, t, wrong, pol, budget, temperature):
    from error_attr.env.alfworld_env import make_env
    def _alarm(s, f): raise TimeoutError()
    signal.signal(signal.SIGALRM, _alarm); signal.alarm(300)
    env = make_env(e["game_file"], budget + 1); steps = [dict(s) for s in e["steps"][:t]]
    try:
        obs, info = env.reset()
        for s in e["steps"][:t]:
            obs, rw, dn, info = env.step([s["action"]])
            if dn[0]: return None
        cur = {"obs": e["steps"][t]["obs"], "admissible": list(e["steps"][t]["admissible"]), "action": wrong}
        obs, rw, dn, info = env.step([wrong]); cur["next_obs"] = obs[0]; steps.append(cur)
        if dn[0]: return {"won": bool(info["won"][0]), "steps": steps}
        h = history_at(e, t); h.steps[-1].action = wrong
        h.steps.append(InteractionStep(observation=obs[0], admissible_commands=list(info["admissible_commands"][0])))
        k = t + 1; won = False
        while k < budget:
            act = pol.act(h, temperature=temperature); h.steps[-1].action = act
            st = {"obs": h.steps[-1].observation, "admissible": list(h.steps[-1].admissible_commands), "action": act}
            obs, rw, dn, info = env.step([act]); st["next_obs"] = obs[0]; steps.append(st); k += 1
            if dn[0]: won = bool(info["won"][0]); break
            h.steps.append(InteractionStep(observation=obs[0], admissible_commands=list(info["admissible_commands"][0])))
        return {"won": won, "steps": steps}
    except TimeoutError:
        return None
    finally:
        signal.alarm(0)
        try: env.close()
        except Exception: pass


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--traj", required=True); ap.add_argument("--adapter", required=True); ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n", type=int, default=500); ap.add_argument("--out", required=True); ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--budget", type=int, default=50); ap.add_argument("--min_len", type=int, default=3)
    ap.add_argument("--mode", default="plausible", choices=["plausible", "random"], help="plausible = the policy's most likely wrong action of the requested type; types cycled for balance")
    a = ap.parse_args(); rng = random.Random(a.seed)
    wins = [e for e in load_jsonl(ROOT / a.traj) if e["won"] and len(e["steps"]) >= a.min_len]; rng.shuffle(wins); wins = wins[: a.n]
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True); lab = out.with_suffix(".labels.jsonl")
    done = {json.loads(l)["src_id"] for l in open(out.with_suffix(".attempts.jsonl"))} if out.with_suffix(".attempts.jsonl").exists() else set()
    tok, model, pol = load_policy(a.model, str(ROOT / a.adapter)); model.eval(); t0 = time.time(); kept = 0; n = 0
    with open(out, "a") as f, open(lab, "a") as g, open(out.with_suffix(".attempts.jsonl"), "a") as h, torch.no_grad():
        for e in wins:
            if e["id"] in done: continue
            L = len(e["steps"]); t = rng.randrange(0, L - 1)
            lp = pol.score_admissible_commands_cached(history_at(e, t)) if a.mode == "plausible" else None
            inj = inject(e["steps"][t]["action"], e["steps"][t + 1]["action"] if t + 1 < L else None, list(e["steps"][t]["admissible"]), rng, lp=lp, want=TYPES[n % len(TYPES)])
            if not inj: continue
            typ, wrong = inj; r = corrupt_and_continue(e, t, wrong, pol, a.budget, a.temperature); n += 1
            h.write(json.dumps({"src_id": e["id"], "t": t, "type": typ, "wrong": wrong, "won": None if r is None else r["won"]}) + "\n"); h.flush()
            if r is None or r["won"]: continue
            kept += 1; sid = f"{e['id']}__synth{t}"
            f.write(json.dumps({**{k: e[k] for k in ("game_file", "goal", "initial_obs", "budget", "split") if k in e}, "id": sid, "src_id": e["id"], "steps": r["steps"],
                                "won": False, "n_steps": len(r["steps"]), "inject_t": t, "inject_type": typ, "inject_action": wrong, "orig_action": e["steps"][t]["action"]}) + "\n"); f.flush()
            g.write(json.dumps({"id": sid, "t_star": t, "type": typ}) + "\n"); g.flush()
            if n % 20 == 0: print(f"[{n}] kept failures {kept} {time.time() - t0:.0f}s", flush=True)
    import collections
    rows = load_jsonl(out); print(f"DONE synth: {len(rows)} synthetic failures | by type {dict(collections.Counter(r['inject_type'] for r in rows))} | mean inject_t {sum(r['inject_t'] for r in rows) / max(1, len(rows)):.1f}", flush=True)


if __name__ == "__main__":
    main()
