"""WebShop with the SAME generic recipe as ALFWorld (user decision Oct 5 23:00): generic one-step evidence (scripts/generic_evidence.py,
env='webshop': native score = purchase reward), vocabulary-free step selector, candidates = student samples at the chosen step,
filters {env_top | rollout (all winners) | rollsub (random K winners) | placebo}.  No WebShop-specific parsing.
  pointer  -- selector on the student's own logged evidence: 'late' (late_generic) or 'first' (first_nonpos_generic)
  scan     -- at each chosen state: 6 policy samples -> top-2 distinct candidates != a-, each executed ONE step (generic evidence)
  rollout  -- each scanned candidate: replay prefix, execute it, let the student continue greedily to the budget; winner = reward 1
  filter   -- write train targets for an arm
"""
import argparse, collections, json, os, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("ERROR_ATTR_ROOT", str(ROOT)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from error_attr.env.webshop_env import WebShopEnv
from webshop_agent import render, chat, map_action
from generic_evidence import generic_evidence
from compare_step_selectors import late_generic, first_nonpos_generic
ANALYSIS_Q = ("Before acting: this attempt FAILED. In at most 3 sentences, say why it failed and what the agent should do at the "
              "CURRENT step (do not name a specific action yet).")   # identical to the ALFWorld teacher (teacher_reason_auc.py)


def ev_at(e, t, obs_after, reward_after, done_after):
    hist = [s["obs"] for s in e["steps"][:t]] + [s["next_obs"] for s in e["steps"][:t]]
    return generic_evidence(e["goal"], hist, e["steps"][t]["obs"], obs_after, env="webshop", score_before=0.0,
                            score_after=float(reward_after or 0.0), done=bool(done_after), success=bool(done_after and reward_after >= 1.0))


def taken(e):
    n = len(e["steps"])
    return [ev_at(e, t, s["next_obs"], s["reward"], t == n - 1 and (s["action"] == "click[buy now]"))["score"] for t, s in enumerate(e["steps"])]


def cmd_pointer(a):
    eps = [json.loads(l) for l in open(ROOT / a.traj)]; fails = [e for e in eps if not e["won"]]
    sel = (lambda s: late_generic(s, 2)) if a.rule == "late" else first_nonpos_generic
    rows = [{"id": e["id"], "t_hat": min(sel(taken(e)), len(e["steps"]) - 1)} for e in fails]
    Path(ROOT / a.out).write_text("".join(json.dumps(r) + "\n" for r in rows)); ts = [r["t_hat"] for r in rows]
    print(f"=== webshop pointer ({a.rule}): {len(rows)} failures | mean t {sum(ts) / len(ts):.2f} t0 {sum(t == 0 for t in ts) / len(ts):.2f} | dist {sorted(collections.Counter(ts).items())[:8]}", flush=True)


def load_policy(model_name, adapter):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(model_name); tok.pad_token = tok.pad_token or tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map={"": 0}); m = PeftModel.from_pretrained(m, str(ROOT / adapter)).eval()
    return tok, m


def replay(env, e, t):
    obs, _ = env.reset(e["session"]); r = 0.0; d = False
    for s in e["steps"][:t]: obs, r, d = env.step(s["action"])
    return obs


def cmd_scan(a):
    import torch
    eps = {e["id"]: e for e in map(json.loads, open(ROOT / a.traj))}; ptr = [json.loads(l) for l in open(ROOT / a.pointer)]
    out = ROOT / a.out; done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    tok, model = load_policy(a.model, a.adapter); env = WebShopEnv(worker=f"gscan_{os.getpid()}"); t0 = time.time()
    with open(out, "a") as f:
        for n, p in enumerate(ptr):
            if p["id"] in done: continue
            e = eps[p["id"]]; t = p["t_hat"]; replay(env, e, t); clicks, has_search = env.valid_actions()
            hist = [{"obs": s["obs"], "action": s["action"]} for s in e["steps"][:t]] + [{"obs": e["steps"][t]["obs"], "action": None}]
            # SAME teacher definition as ALFWorld (self_teacher_informed.py): the student with HINDSIGHT (what happened after step t in the
            # failed episode) first writes an ANALYSIS, then proposes actions with the analysis in context.  WebShop's action space is
            # free text, so candidates are sampled (6) instead of scoring an enumerable admissible list.
            fut = e["steps"][t:t + 25]
            hind = ("Hindsight: this attempt FAILED. After this point the agent did the following (episode length %d steps):\n" % len(e["steps"])
                    + "\n".join(f"- step {t + i}: {x['action']} -> {' '.join(x['next_obs'].split())[:160]}" for i, x in enumerate(fut)))
            base = render(e["goal"], hist).replace("Next action:", hind + "\n\n")
            aq = base + ANALYSIS_Q
            ids_a = tok(chat(tok, aq), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            with torch.no_grad(): ga = model.generate(ids_a, max_new_tokens=120, do_sample=False, pad_token_id=tok.pad_token_id)
            analysis = " ".join(tok.decode(ga[0, ids_a.shape[1]:], skip_special_tokens=True).split())
            prompt = base + "Analysis: " + analysis + "\n\nUsing everything above, choose the best action for the CURRENT step.\nNext action:"
            ids_ = tok(chat(tok, prompt), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            with torch.no_grad(): g = model.generate(ids_, max_new_tokens=32, do_sample=True, temperature=1.0, num_return_sequences=6, pad_token_id=tok.pad_token_id)
            cnt = collections.Counter(); a_minus = e["steps"][t]["action"]
            for x in g:
                act, _ = map_action(tok.decode(x[ids_.shape[1]:], skip_special_tokens=True).strip().split("\n")[0], clicks, has_search)
                if act != a_minus: cnt[act] += 1
            cands = []
            for c, k in cnt.most_common(2):
                if cands: replay(env, e, t)
                o, r, d = env.step(c); cands.append({"cand": c, "freq": k / 6, "obs_plus": o[:3000], "reward": r, "done": d, **ev_at(e, t, o, r, d)})
            f.write(json.dumps({"id": e["id"], "t": t, "a_minus": a_minus, "analysis": analysis, "cands": cands}) + "\n"); f.flush()
            if n % 25 == 0: print(f"[{n + 1}/{len(ptr)}] {time.time() - t0:.0f}s", flush=True)
    print("DONE gscan", flush=True)


def cmd_rollout(a):
    import torch
    eps = {e["id"]: e for e in map(json.loads, open(ROOT / a.traj))}; scan = [json.loads(l) for l in open(ROOT / a.scan)]
    out = ROOT / a.out; done = {(json.loads(l)["id"], json.loads(l)["cand"]) for l in open(out)} if out.exists() else set()
    tok, model = load_policy(a.model, a.adapter); env = WebShopEnv(worker=f"groll_{os.getpid()}"); t0 = time.time(); steps_used = 0
    with open(out, "a") as f:
        for n, r in enumerate(scan):
            e = eps[r["id"]]; t = r["t"]
            for c in r["cands"]:
                if (r["id"], c["cand"]) in done: continue
                obs = replay(env, e, t); hist = [{"obs": s["obs"], "action": s["action"]} for s in e["steps"][:t]] + [{"obs": obs, "action": c["cand"]}]
                obs, rew, d = env.step(c["cand"]); k = t + 1; steps_used += 1
                while not d and k < a.budget:
                    hist.append({"obs": obs, "action": None})
                    ids_ = tok(chat(tok, render(e["goal"], hist)), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                    with torch.no_grad(): g = model.generate(ids_, max_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id)
                    clicks, hs = env.valid_actions(); act, _ = map_action(tok.decode(g[0, ids_.shape[1]:], skip_special_tokens=True).strip().split("\n")[0], clicks, hs)
                    hist[-1]["action"] = act; obs, rew, d = env.step(act); k += 1; steps_used += 1
                f.write(json.dumps({"id": r["id"], "t": t, "cand": c["cand"], "won": bool(d and rew >= 1.0), "reward": rew, "env_steps": k - t}) + "\n"); f.flush()
            if n % 25 == 0: print(f"[{n + 1}/{len(scan)}] env_steps {steps_used} {time.time() - t0:.0f}s", flush=True)
    print(f"DONE grollout env_steps {steps_used}", flush=True)


def cmd_filter(a):
    rng = random.Random(a.seed); scan = [json.loads(l) for l in open(ROOT / a.scan)]; picks = []
    won = {(json.loads(l)["id"], json.loads(l)["cand"]): json.loads(l)["won"] for l in open(ROOT / a.roll)} if a.roll and os.path.exists(ROOT / a.roll) else {}
    for r in scan:
        cs = r["cands"]
        if not cs: continue
        if a.rule == "env_top":
            ok = [c for c in cs if not c["no_info"] and c["score"] > 0]
            if ok: c = max(ok, key=lambda c: (c["score"], c["freq"])); picks.append((r["id"], r["t"], c["cand"], c["score"]))
        elif a.rule in ("rollout", "rollsub"):
            w = [c for c in cs if won.get((r["id"], c["cand"]))]
            if w: c = max(w, key=lambda c: c["freq"]); picks.append((r["id"], r["t"], c["cand"], 1.0))
        elif a.rule == "placebo": c = rng.choice(cs); picks.append((r["id"], r["t"], c["cand"], rng.random()))
    if a.rule == "env_top": picks = sorted(picks, key=lambda x: x[3], reverse=True)[: a.k]
    elif a.rule == "rollsub": rng.shuffle(picks); picks = picks[: a.k]
    elif a.rule == "placebo": picks = sorted(picks, key=lambda x: x[3])[: a.k]
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps({"src": a.traj, "id": i, "t": t, "action": c, "rank_score": s}) + "\n" for i, t, c, s in picks))
    print(f"=== webshop generic filter {a.rule}: {len(picks)} targets", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pointer"); p.add_argument("--traj", required=True); p.add_argument("--rule", default="late"); p.add_argument("--out", required=True)
    s = sub.add_parser("scan"); s.add_argument("--traj", required=True); s.add_argument("--pointer", required=True); s.add_argument("--adapter", required=True); s.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); s.add_argument("--out", required=True)
    r = sub.add_parser("rollout"); r.add_argument("--traj", required=True); r.add_argument("--scan", required=True); r.add_argument("--adapter", required=True); r.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); r.add_argument("--budget", type=int, default=15); r.add_argument("--out", required=True)
    f = sub.add_parser("filter"); f.add_argument("--traj", required=True); f.add_argument("--scan", required=True); f.add_argument("--roll", default=None); f.add_argument("--rule", required=True); f.add_argument("--k", type=int, default=100); f.add_argument("--seed", type=int, default=0); f.add_argument("--out", required=True)
    a = ap.parse_args(); {"pointer": cmd_pointer, "scan": cmd_scan, "rollout": cmd_rollout, "filter": cmd_filter}[a.cmd](a)


if __name__ == "__main__":
    main()
