"""TextCraft plug-in checks for the self-evolution recipe (feasibility sketch, Oct 8).
  release -- failed episodes of a collect file -> Who&When 'alfagent' rows for Model A (integer step_number, reasoning None,
             is_final_answer False, no reward / score anywhere) + ids file
  pointer -- Model A predictions -> pointer.jsonl and a summary: relative position of the pointed step, action type / outcome there,
             and agreement with two oracle references computed from the recipes:
               first_waste  = first step whose action does not shorten the oracle plan from that state (L*(s_t+1) >= L*(s_t))
               no_return    = first step after which the task cannot be finished in the remaining budget (t+1 + L*(s_t+1) > budget)
  show    -- print episodes with Model A's pointed step and the references, for manual judgement
  scan    -- at each pointed state: the student itself (hindsight of the failed future + Model A's diagnosis + its own analysis, exactly
             the ALFWorld informed self-teacher prompt) scores the admissible actions -> top-2 candidates != the taken action; each is
             executed ONE step from the saved state and scored by the goal-progress evidence below; the taken action, the oracle action
             and the oracle distance change dL* of every candidate are logged as ground truth
  summary -- does F separate good corrections (dL* <= -1, i.e. on an optimal plan) from bad ones?

Goal-progress evidence F = dPhi - KAPPA * 1[no new info]   (KAPPA = 1)
  components: the goal's requirement tree read off the SHOWN crafting commands (recipe per item = the oracle's choice; a tag ingredient
              such as 'planks' is one component satisfied by any member), with the quantity each one needs (batches rounded up)
  f(c) = min(1, f(parent) + alloc(c) / need(c))    (alloc = inventory count allocated to c -- each held item is allocated once, in
                                                    tree pre-order, a tag drawing on all its members; what the parent already holds
                                                    embodies the same fraction of c, so consumed ingredients stay credited)
  sigma(c) = 2 'confirmed' if f(c) >= 1 (enough for its use, or already crafted into the parent), 'located' if 0 < f(c) < 1
             (partly obtained; graded 2 f(c) by default, or 1 with TC_GRADED=0), 0 'unobserved' otherwise;  Phi = sum_c sigma(c);
             crafting the goal = success (dPhi = WIN = 10)
  no new info: the action failed ('Could not ...'), or 'inventory' returned a listing already seen in the episode
  bottom (irreversible failure): TextCraft has none -- every base item can be fetched again without limit and nothing is ever lost --
             so there is no bottom; destructive moves (crafting an off-tree item from a needed intermediate) show up as dPhi < 0.
"""
import argparse, collections, json, math, os, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("ERROR_ATTR_ROOT", str(ROOT)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from textcraft_env import TextCraftEnv, item_id_to_str, str_to_item_id
KAPPA = 1.0
WIN = 10.0
GRADED = int(os.environ.get("TC_GRADED", "1"))   # 1: 'located' graded by the obtained fraction, sigma = 2 f in (0, 2); 0: sigma in {0, 1, 2}


# ----------------------------------------------------------------------------- evidence
def default_member(env, tag):
    """same choice as the oracle (_concrete with an empty inventory): the shallowest member with a shown recipe (or fetchable)."""
    ms = env.members(tag); ok = [m for m in ms if env._recipe_for(m) is not None or env.fetchable(m)]
    return min(ok or ms, key=lambda m: (env.t.get_min_depth(m), m))


def req_tree(env):
    """[(node_id, parent_id, name, is_tag, need)] for the goal, built from an empty inventory with the oracle's recipe choices."""
    nodes = []
    def build(name, is_tag, need, parent, depth=0):
        nid = len(nodes); nodes.append((nid, parent, name, is_tag, need))
        if depth > 10: return
        conc = default_member(env, name) if is_tag else name
        if env.fetchable(conc): return
        r = env._recipe_for(conc)
        if r is None: return
        B = math.ceil(need / r.output_item.count)
        for x in r.input_items:
            t = x.item_tag.item_id is None; build(x.item_tag.tag if t else x.item_tag.item_id, t, x.count * B, nid, depth + 1)
    build(env.goal, False, 1, None); return nodes


def held(env, inv, name, is_tag):
    return sum(inv.get(m, 0) for m in env.members(name)) if is_tag else inv.get(name, 0)


def phi(env, nodes, inv):
    """held items are ALLOCATED once (an item used by two components, e.g. sticks for the torch and for the rail, is not counted twice):
    nodes in pre-order take min(remaining need, what is left of the item) from a shared pool."""
    pool = dict(inv); f = {}; tot = 0.0; sig = {}
    for nid, par, name, is_tag, need in nodes:   # parents precede children
        base = f[par] if par is not None else 0.0; want = need * (1.0 - base); got = 0
        for m in (env.members(name) if is_tag else [name]):
            take = min(want - got, pool.get(m, 0))
            if take > 0: pool[m] -= take; got += take
        f[nid] = min(1.0, base + got / need)
        s = 2.0 if f[nid] >= 1 else (GRADED * 2 * f[nid] if GRADED else (1.0 if f[nid] > 0 else 0.0)); sig[nid] = s; tot += s
    return tot, sig


def evidence(env, nodes, inv_before, action, obs_after, inv_after, done, seen_inventory):
    p0, _ = phi(env, nodes, inv_before); p1, _ = phi(env, nodes, inv_after)
    failed = obs_after.startswith("Could not") or obs_after.startswith("Wrong")
    nul = failed or (action == "inventory" and obs_after in seen_inventory)
    dphi = WIN if done else p1 - p0
    return {"dphi": dphi, "no_info": bool(nul), "regress": (not done) and p1 < p0, "F": dphi - KAPPA * nul}


def oracle_len(env, inv):
    s = env.get_state(); env.set_state(inv); plan = env.expert_actions(); env.set_state(s)
    return len(plan) if plan is not None else 99


def replay(env, e, t):
    env.reset(e["task"]); seen = set()
    for s in e["steps"][:t]:
        o, _, _ = env.step(s["action"])
        if s["action"] == "inventory": seen.add(o)
    return seen


def one_step(env, e, t, nodes, inv_t, seen, action):
    env.set_state(inv_t); o, r, d = env.step(action); inv1 = env.get_state()
    ev = evidence(env, nodes, inv_t, action, o, inv1, d, seen); L0 = oracle_len(env, inv_t); L1 = 0 if d else oracle_len(env, inv1)
    env.set_state(inv_t)
    return {"action": action, "obs": o, "done": d, **ev, "dL": L1 - L0}


# ----------------------------------------------------------------------------- release / pointer
def cmd_release(a):
    from textcraft_agent import to_release
    eps = [json.loads(l) for l in open(ROOT / a.traj)]; fails = [e for e in eps if not e["won"]][: a.n]
    rows = [to_release(e) for e in fails]
    if a.query_commands:   # variant: the shown crafting commands are part of the task text Model A sees (the user turn is not rendered)
        for r, e in zip(rows, fails): r["task"]["query"] = e["initial_obs"]
    open(ROOT / a.out, "w").writelines(json.dumps(r) + "\n" for r in rows)
    open(ROOT / a.ids, "w").writelines(json.dumps({"id": e["id"]}) + "\n" for e in fails)
    print(f"release: {len(fails)} failed episodes (of {len(eps)})")


def refs(env, e):
    """oracle references for an episode: first_waste and no_return step indices (0-based)."""
    env.reset(e["task"]); inv = env.get_state(); L = [oracle_len(env, inv)]; fw = nr = None
    for t, s in enumerate(e["steps"]):
        env.set_state(inv); env.step(s["action"]); inv = env.get_state(); L.append(oracle_len(env, inv))
        if fw is None and L[t + 1] >= L[t]: fw = t
        if nr is None and t + 1 + L[t + 1] > e["budget"]: nr = t
    return fw, nr, L


def atype(env, e, t):
    s = e["steps"][t]; a, o = s["action"], s["next_obs"]; failed = o.startswith("Could not") or o.startswith("Wrong")
    k = a.split()[0]
    if k == "craft":
        out = str_to_item_id(a[6:].split(" using ")[0].split(" ", 1)[1]); env.reset(e["task"])
        tree = any(r.recipe_str in env.tree_recipes for r in env.shown if r.output_item.item_tag.item_id == out)
        k = "craft(" + ("tree" if tree else "off-tree") + (",failed" if failed else ",ok") + ")"
    elif k == "get":
        k = "get"
    return k


def cmd_pointer(a):
    env = TextCraftEnv(); eps = {e["id"]: e for e in map(json.loads, open(ROOT / a.traj))}; rows = []
    for r in map(json.loads, open(ROOT / a.pred)):
        if r["id"] not in eps: continue
        e = eps[r["id"]]; n = len(e["steps"]); t = r["t_hat"] if r["t_hat"] is not None else 0; t = max(0, min(int(t), n - 1))
        fw, nr, L = refs(env, e)
        rows.append({"id": r["id"], "t_hat": t, "raw_t": r["t_hat"], "n": n, "rel": t / max(1, n - 1), "type": "Model-A mode " + str((r.get("pred") or {}).get("error_mode")),
                     "atype": atype(env, e, t), "first_waste": fw, "no_return": nr, "L": L,
                     "first_fail": next((k for k, st in enumerate(e["steps"]) if st["next_obs"].startswith("Could not")), None)})
    open(ROOT / a.out, "w").writelines(json.dumps(x) + "\n" for x in rows)
    N = len(rows); rel = [x["rel"] for x in rows]
    print(f"=== textcraft Model A pointer: {N} failures; unparsed {sum(x['raw_t'] is None for x in rows)}; mean step {sum(x['t_hat'] for x in rows) / N:.2f} of mean length {sum(x['n'] for x in rows) / N:.2f}")
    print(f"    relative position: mean {sum(rel) / N:.2f}; quartiles " + str(collections.Counter(min(3, int(4 * x)) for x in rel).most_common()) + f"; t=0 share {sum(x['t_hat'] == 0 for x in rows) / N:.2f}")
    print(f"    action type at the pointed step: {dict(collections.Counter(x['atype'] for x in rows).most_common())}")
    print(f"    error mode: {dict(collections.Counter(x['type'] for x in rows).most_common())}")
    for ref in ("first_waste", "no_return"):
        ok = [x for x in rows if x[ref] is not None]
        ex = sum(x["t_hat"] == x[ref] for x in ok); w1 = sum(abs(x["t_hat"] - x[ref]) <= 1 for x in ok); early = sum(x["t_hat"] < x[ref] for x in ok)
        print(f"    vs oracle {ref}: exact {ex}/{len(ok)} = {ex / max(1, len(ok)):.2f}, within 1 {w1 / max(1, len(ok)):.2f}, Model A earlier {early / max(1, len(ok)):.2f}, later {(len(ok) - ex - early) / max(1, len(ok)):.2f}")
    # baselines with the same references
    for name, f in (("Model A", lambda x: x["t_hat"]), ("first step", lambda x: 0), ("last step", lambda x: x["n"] - 1), ("first failed action", lambda x: x["first_fail"] if x["first_fail"] is not None else x["n"] - 1)):
        ok = [x for x in rows if x["first_waste"] is not None]; ok2 = [x for x in rows if x["no_return"] is not None]
        print(f"    '{name}': exact vs first_waste {sum(f(x) == x['first_waste'] for x in ok) / max(1, len(ok)):.2f}, vs no_return {sum(f(x) == x['no_return'] for x in ok2) / max(1, len(ok2)):.2f}")


def cmd_show(a):
    env = TextCraftEnv(); eps = {e["id"]: e for e in map(json.loads, open(ROOT / a.traj))}; ptr = [json.loads(l) for l in open(ROOT / a.pointer)]
    random.Random(a.seed).shuffle(ptr)
    for p in ptr[: a.n]:
        e = eps[p["id"]]; print(f"\n######## {p['id']}  goal: {e['goal']} (depth {e['depth']}, oracle {e['expert_len']} steps, budget {e['budget']}) | Model A t={p['t_hat']} {p['type']} | first_waste {p['first_waste']} no_return {p['no_return']}")
        env.reset(e["task"]); print("   oracle plan: " + " | ".join(env.expert_actions()))
        for t, s in enumerate(e["steps"]):
            mark = "<== A" if t == p["t_hat"] else ""
            print(f"   {t:2d} L*={p['L'][t]:2d} {s['action'][:70]:70s} -> {s['next_obs'][:60]:60s} {mark}")


# ----------------------------------------------------------------------------- scan with the informed self-teacher
ANALYSIS_Q = ("Before acting: this attempt FAILED. In at most 3 sentences, say why it failed and what the agent should do at the "
              "CURRENT step (do not name a specific action yet).")


def cmd_scan(a):
    import torch
    from textcraft_agent import load_policy, history_at
    env = TextCraftEnv(); eps = {e["id"]: e for e in map(json.loads, open(ROOT / a.traj))}; ptr = [json.loads(l) for l in open(ROOT / a.pointer)][: a.n]
    out = ROOT / a.out; done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    tok, model, pol = load_policy(a.model, a.adapter); t0 = time.time()
    with open(out, "a") as f, torch.no_grad():
        for k, p in enumerate(ptr):
            if p["id"] in done: continue
            e = eps[p["id"]]; t = p["t_hat"]; h = history_at(e, t)
            fut = e["steps"][t:t + 25]
            hind = ("Hindsight: this attempt FAILED. After this point the agent did the following "
                    f"(episode length {len(e['steps'])} steps, budget exhausted):\n" + "\n".join(f"- step {t + i}: {s['action']} -> {s['next_obs'][:160]}" for i, s in enumerate(fut))
                    + "\nUsing everything above, choose the best action for the CURRENT step.\n")
            diag = f"Diagnosis (from an error attributor): the first decisive error was at step {t}, type: {p['type']}."
            base = h.render_prompt()
            p1 = base.replace("Admissible actions:", hind + diag + "\nAdmissible actions:", 1)
            user = p1.replace("Next action:", ANALYSIS_Q)
            msg = tok.apply_chat_template([{"role": "system", "content": "You are a crafting agent reviewing a failed attempt."}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
            x = tok(msg, return_tensors="pt", add_special_tokens=False).to(model.device)
            g = model.generate(**x, max_new_tokens=96, do_sample=False, pad_token_id=tok.pad_token_id)
            analysis = " ".join(tok.decode(g[0, x.input_ids.shape[1]:], skip_special_tokens=True).split())
            p2 = base.replace("Admissible actions:", hind + diag + "\nAnalysis: " + analysis + "\nAdmissible actions:", 1)

            class _H:
                steps = h.steps
                def render_prompt(self, *_): return p2
            lp = pol.score_admissible_commands_cached(_H()); a_minus = e["steps"][t]["action"]
            cands = [c for c in sorted(lp, key=lambda c: -lp[c]) if c != a_minus][:2]
            seen = replay(env, e, t); inv_t = env.get_state(); nodes = req_tree(env)
            orac = env.expert_actions(); orac = orac[0] if orac else None
            row = {"id": e["id"], "t": t, "a_minus": a_minus, "analysis": analysis, "inv_t": inv_t,
                   "taken": one_step(env, e, t, nodes, inv_t, seen, a_minus), "oracle": one_step(env, e, t, nodes, inv_t, seen, orac) if orac else None,
                   "cands": [one_step(env, e, t, nodes, inv_t, seen, c) for c in cands], "plain_top": max(lp, key=lp.get)}
            f.write(json.dumps(row) + "\n"); f.flush()
            if k % 10 == 0: print(f"[{k + 1}/{len(ptr)}] {time.time() - t0:.0f}s", flush=True)
    print("DONE scan", flush=True)


def auc(pos, neg):
    if not pos or not neg: return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def cmd_summary(a):
    rows = [json.loads(l) for l in open(ROOT / a.scan)]; C = [c for r in rows for c in r["cands"]]
    good = [c for c in C if c["dL"] <= -1 or c["done"]]; bad = [c for c in C if not (c["dL"] <= -1 or c["done"])]
    print(f"=== textcraft evidence on {len(rows)} pointed failure states, {len(C)} teacher candidates: {len(good)} good (shorten the oracle plan), {len(bad)} not")
    print(f"    F mean good {sum(c['F'] for c in good) / max(1, len(good)):.2f} vs not {sum(c['F'] for c in bad) / max(1, len(bad)):.2f}; AUC(F: good vs not) {auc([c['F'] for c in good], [c['F'] for c in bad]):.3f}")
    keep = [max(r["cands"], key=lambda c: c["F"]) for r in rows if r["cands"] and max(c["F"] for c in r["cands"]) > 0]
    print(f"    states with a positive-F candidate {len(keep)}/{len(rows)}; of the kept (best-F) candidates {sum(c['dL'] <= -1 or c['done'] for c in keep)}/{len(keep)} shorten the oracle plan")
    print(f"    taken action at the pointed step: good {sum(r['taken']['dL'] <= -1 for r in rows)}/{len(rows)}, mean F {sum(r['taken']['F'] for r in rows) / len(rows):.2f};"
          f" oracle action mean F {sum(r['oracle']['F'] for r in rows if r['oracle']) / max(1, sum(1 for r in rows if r['oracle'])):.2f}")
    print(f"    teacher top-1 == oracle action: {sum(1 for r in rows if r['cands'] and r['oracle'] and r['cands'][0]['action'] == r['oracle']['action'])}/{len(rows)};"
          f" oracle action among the 2 candidates: {sum(1 for r in rows if r['oracle'] and any(c['action'] == r['oracle']['action'] for c in r['cands']))}/{len(rows)}")
    cf = collections.Counter((c["dL"] <= -1 or c["done"], round(c["F"], 1)) for c in C); print("    (good?, F) counts:", sorted(cf.items()))
    for r in rows[: a.n_examples]:
        print(f"\n  {r['id']} t={r['t']} inv={r['inv_t']}\n    taken  {r['taken']['action']:60s} F={r['taken']['F']:+.1f} dL={r['taken']['dL']:+d} -> {r['taken']['obs'][:50]}")
        if r["oracle"]: print(f"    oracle {r['oracle']['action']:60s} F={r['oracle']['F']:+.1f} dL={r['oracle']['dL']:+d}")
        for c in r["cands"]: print(f"    cand   {c['action']:60s} F={c['F']:+.1f} dL={c['dL']:+d} -> {c['obs'][:50]}")


def cmd_selftest(a):
    """sanity: along oracle demos every step should have F > 0 (dPhi > 0) and no step regress."""
    env = TextCraftEnv(); eps = [json.loads(l) for l in open(ROOT / a.traj)][: a.n]; Fs = []; nonpos = 0
    for e in eps:
        env.reset(e["task"]); nodes = req_tree(env); seen = set()
        for s in e["steps"]:
            inv0 = env.get_state(); o, r, d = env.step(s["action"]); ev = evidence(env, nodes, inv0, s["action"], o, env.get_state(), d, seen); Fs.append(ev["F"]); nonpos += ev["F"] <= 0
    print(f"oracle steps: {len(Fs)}, F<=0 on {nonpos} ({nonpos / len(Fs):.3f}), mean F {sum(Fs) / len(Fs):.2f}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("release"); s.add_argument("--traj", required=True); s.add_argument("--out", required=True); s.add_argument("--ids", required=True); s.add_argument("--n", type=int, default=50); s.add_argument("--query_commands", action="store_true")
    s = sub.add_parser("pointer"); s.add_argument("--traj", required=True); s.add_argument("--pred", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("show"); s.add_argument("--traj", required=True); s.add_argument("--pointer", required=True); s.add_argument("--n", type=int, default=15); s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("scan"); s.add_argument("--traj", required=True); s.add_argument("--pointer", required=True); s.add_argument("--out", required=True); s.add_argument("--adapter", required=True)
    s.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); s.add_argument("--n", type=int, default=50)
    s = sub.add_parser("summary"); s.add_argument("--scan", required=True); s.add_argument("--n_examples", type=int, default=6)
    s = sub.add_parser("selftest"); s.add_argument("--traj", default="reports/textcraft/gold_train.jsonl"); s.add_argument("--n", type=int, default=378)
    a = ap.parse_args(); {"release": cmd_release, "pointer": cmd_pointer, "show": cmd_show, "scan": cmd_scan, "summary": cmd_summary, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
