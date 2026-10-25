"""WebShop with the FINAL recipe (user-approved Oct 7 22:10): frozen 3B Model A picks the step, the hindsight self-teacher (with Model A's
diagnosis) samples candidates, graded one-step goal-progress evidence decides, K = 20% of failures + 100 replay.
  release -- failed sessions -> Who&When 'alfagent' release rows (instruction as the user turn, one action entry per step) for Model A
             (scripts/alfworld_attribute.py --variant no_observations drops the page text, as on ALFWorld)
  pointer -- Model A predictions -> pointer.jsonl {id, t_hat (clipped to the episode), type}
  scan    -- at each pointer state: hindsight + diagnosis + own analysis -> 6 samples -> top-2 distinct candidates != a-, each executed ONE
             step on the server, scored by ev_goal (below)
  filter  -- per state the best candidate with score > 0 and not no_info; states ranked by (score, freq); top K
Goal-progress evidence for WebShop (ev_goal): conditions = category/attribute phrases, required option values, price cap.
  loc  = a new results/product page whose new words overlap the instruction (a candidate product located)
  sat  = (a) opening a product page that offers every required option, is within the price cap and shows >= 1 attribute phrase, graded by
         the fraction of conditions verified on the page; (b) selecting a required option; (c) buying with reward 1
  vio  = buying with reward < 1 (irreversible)
  opening a product page that FAILS a requirement is reversible -> no progress: loc = sat = 0 (user decision Oct 7)
  nul  = page unchanged, a page already seen, or nothing new and no progress
  score = 2*loc + 2*sat*(1 + frac) + min(n_new,5)/5 - 2*nul - 4*vio
"""
import argparse, collections, json, os, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("ERROR_ATTR_ROOT", str(ROOT)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from webshop_evidence import toks, body, parse_goal, product_match, goal_tokens, known_tokens
NAV = ("next >", "< prev", "back to search", "buy now", "description", "features", "reviews", "attributes")


def ev_goal(e, t, action, obs_plus, reward, done):
    known = known_tokens(e, t); seen = [body(s["obs"]).strip() for s in e["steps"][:t]] + [body(s["next_obs"]).strip() for s in e["steps"][:t]] + [body(e["steps"][t]["obs"]).strip()]
    gw = goal_tokens(e); b = body(obs_plus); bt = toks(b); new = bt - known
    is_product = action.startswith("click[b0") or action.startswith("click[b1"); opt = action[6:-1] if action.startswith("click[") else ""
    loc = sat = vio = 0; frac = 0.0
    if done:
        if reward >= 1.0: sat, frac = 1, 1.0
        else: vio = 1
    elif is_product:
        opt_ok, price_ok, n_attr, n_attr_all = product_match(e["goal"], obs_plus); attrs, opts, cap = parse_goal(e["goal"])
        if opt_ok and price_ok and n_attr >= 1:
            sat = 1; frac = (n_attr + len(opts) + 1) / max(1, n_attr_all + len(opts) + 1)
        # else: reversible non-matching product -> no progress (loc = sat = 0)
    elif opt and opt not in NAV and opt in e["goal"].lower():
        sat = 1
    else:
        loc = int(len(new & gw) > 0)
    nul = int(b.strip() == seen[-1] or b.strip() in seen[:-1] or (len(new) == 0 and not sat))
    score = 2.0 * loc + 2.0 * sat * (1.0 + frac) + min(len(new), 5) / 5.0 - 2.0 * nul - 4.0 * vio
    return {"loc": loc, "sat": sat, "frac": round(frac, 3), "vio": vio, "no_info": bool(nul), "n_new": len(new), "score": score}


def cmd_release(a):
    eps = [json.loads(l) for l in open(ROOT / a.traj)][: a.first]; n = 0
    with open(ROOT / a.out, "w") as f, open(ROOT / a.ids, "w") as g:
        for e in eps:
            if e["won"]: continue
            traj = [{"kind": "user", "content": e["goal"].replace("Instruction:", "").strip() + "\n\n" + body(e["steps"][0]["obs"])[:2000]}]
            for k, s in enumerate(e["steps"]):
                traj.append({"kind": "action", "step_number": str(k + 1), "reasoning": "None", "action": s["action"], "observation": body(s["next_obs"])[:1500],
                             "is_final_answer": str(k == len(e["steps"]) - 1)})
            f.write(json.dumps({"id": e["id"], "framework": "alfagent", "benchmark": "webshop", "modality": "text", "task": {"query": e["goal"].replace("Instruction:", "").strip(), "answer": ""},
                                "trajectory": traj, "ground_truth": {}, "extras": {"agents": [{"id": "agent", "role": "agent"}]}}) + "\n")
            g.write(json.dumps({"id": e["id"]}) + "\n"); n += 1
    print(f"release: {n} failed sessions among the first {a.first}")


def cmd_pointer(a):
    eps = {e["id"]: e for e in map(json.loads, open(ROOT / a.traj))}; rows = []
    for r in map(json.loads, open(ROOT / a.pred)):
        if r["id"] not in eps: continue
        t = r["t_hat"] if r["t_hat"] is not None else 0; t = max(0, min(int(t), len(eps[r["id"]]["steps"]) - 1))
        rows.append({"id": r["id"], "t_hat": t, "type": "Model-A mode " + str((r.get("pred") or {}).get("error_mode"))})
    open(ROOT / a.out, "w").writelines(json.dumps(x) + "\n" for x in rows)
    ts = [x["t_hat"] for x in rows]; print(f"=== webshop Model A pointer: {len(rows)} states, mean t {sum(ts) / len(ts):.2f}, t0 {sum(t == 0 for t in ts) / len(ts):.2f}, dist {sorted(collections.Counter(ts).items())[:10]}")


def cmd_scan(a):
    import torch
    from error_attr.env.webshop_env import WebShopEnv
    from webshop_agent import render, chat, map_action
    from webshop_generic import load_policy, replay, ANALYSIS_Q
    eps = {e["id"]: e for e in map(json.loads, open(ROOT / a.traj))}; ptr = [json.loads(l) for l in open(ROOT / a.pointer)]
    if a.shard: ptr = ptr[a.shard[0]::a.shard[1]]
    out = ROOT / a.out; done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    tok, model = load_policy(a.model, a.adapter); env = WebShopEnv(worker=f"fscan_{os.getpid()}"); t0 = time.time()
    with open(out, "a") as f:
        for n, p in enumerate(ptr):
            if p["id"] in done: continue
            e = eps[p["id"]]; t = p["t_hat"]; replay(env, e, t); clicks, has_search = env.valid_actions()
            hist = [{"obs": s["obs"], "action": s["action"]} for s in e["steps"][:t]] + [{"obs": e["steps"][t]["obs"], "action": None}]
            fut = e["steps"][t:t + 25]
            diag = f"Diagnosis (from an error attributor): the first decisive error was at step {t}, type: {p['type']}."
            hind = ("Hindsight: this attempt FAILED. After this point the agent did the following (episode length %d steps):\n" % len(e["steps"])
                    + "\n".join(f"- step {t + i}: {x['action']} -> {' '.join(x['next_obs'].split())[:160]}" for i, x in enumerate(fut)))
            base = render(e["goal"], hist).replace("Next action:", diag + "\n" + hind + "\n\n")
            ids_a = tok(chat(tok, base + ANALYSIS_Q), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
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
                o, r, d = env.step(c); cands.append({"cand": c, "freq": k / 6, "obs_plus": o[:3000], "reward": r, "done": d, **ev_goal(e, t, c, o, r, d)})
            st = e["steps"][t]
            taken = ev_goal(e, t, a_minus, st["next_obs"], st["reward"], t == len(e["steps"]) - 1 and a_minus == "click[buy now]")
            f.write(json.dumps({"id": e["id"], "t": t, "a_minus": a_minus, "taken": taken, "analysis": analysis, "cands": cands}) + "\n"); f.flush()
            if n % 25 == 0: print(f"[{n + 1}/{len(ptr)}] {time.time() - t0:.0f}s", flush=True)
    print("DONE fscan", flush=True)


def cmd_filter(a):
    rows = {}
    for p in sorted(Path(ROOT / a.scan_dir).glob("scan_s*.jsonl")):
        for l in open(p): r = json.loads(l); rows[r["id"]] = r
    nf = sum(1 for _ in open(ROOT / a.ids)); K = max(1, round(a.frac * nf)); picks = []; kinds = collections.Counter()
    for r in rows.values():
        ok = [c for c in r["cands"] if c["score"] > 0 and not c["no_info"]]
        if not ok: continue
        c = max(ok, key=lambda c: (c["score"], c["freq"])); picks.append((r["id"], r["t"], c["cand"], c["score"], c["freq"], c))
    picks.sort(key=lambda x: (x[3], x[4]), reverse=True); keep = picks[:K]
    for _, _, cand, _, _, c in keep:
        kind = "buy" if cand == "click[buy now]" else "search" if cand.startswith("search[") else "product" if cand.startswith("click[b0") or cand.startswith("click[b1") else "nav" if cand[6:-1] in NAV else "option"
        kinds[(kind, "sat" if c["sat"] else "loc" if c["loc"] else "novelty")] += 1
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for i, t, cand, s, fr, c in keep: f.write(json.dumps({"src": a.traj, "id": i, "t": t, "action": cand, "rank_score": s}) + "\n")
    print(f"=== webshop final filter: {len(rows)} scanned states, {len(picks)} with a positive candidate, K = {a.frac} x {nf} = {K}, kept {len(keep)} | kinds {dict(kinds)}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("release"); s.add_argument("--traj", required=True); s.add_argument("--out", required=True); s.add_argument("--ids", required=True); s.add_argument("--first", type=int, default=1000)
    s = sub.add_parser("pointer"); s.add_argument("--traj", required=True); s.add_argument("--pred", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("scan"); s.add_argument("--traj", required=True); s.add_argument("--pointer", required=True); s.add_argument("--out", required=True)
    s.add_argument("--adapter", required=True); s.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); s.add_argument("--shard", type=int, nargs=2, default=None)
    s = sub.add_parser("filter"); s.add_argument("--scan_dir", required=True); s.add_argument("--ids", required=True); s.add_argument("--traj", required=True); s.add_argument("--out", required=True); s.add_argument("--frac", type=float, default=0.2)
    a = ap.parse_args(); {"release": cmd_release, "pointer": cmd_pointer, "scan": cmd_scan, "filter": cmd_filter}[a.cmd](a)


if __name__ == "__main__":
    main()
