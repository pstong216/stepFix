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
from webshop_evidence import toks, parse_goal, product_match, goal_tokens
import re


def body(obs):
    """page text without the instruction banner; handles both 'Instruction: [SEP] goal [SEP] ...' and the first search page
    'WebShop [SEP] Instruction: [SEP] goal [SEP] Search' (v2 fix, Oct 7: the old body() kept the banner on the first page)."""
    parts = obs.split("[SEP]")
    for i in range(min(2, len(parts))):
        if parts[i].strip().lower().startswith("instruction"): return "[SEP]".join(parts[i + 2:])
    return obs


def known_tokens(e, t):
    k = set()
    for s in e["steps"][:t]: k |= toks(body(s["obs"])) | toks(body(s["next_obs"]))
    return k | toks(body(e["steps"][t]["obs"]))


def results_loc(goal, page):
    """graded 'located': for every product listed on a results page, the fraction of the instruction's attribute phrases whose content
    words appear in its title (averaged over phrases) plus whether its price is under the cap; the best product's value in [0, 1]."""
    attrs, opts, cap = parse_goal(goal); segs = [x.strip() for x in body(page).split("[SEP]")]; best = 0.0
    for i, sg in enumerate(segs[:-1]):
        if not re.fullmatch(r"[bB]0[0-9A-Za-z]{8}", sg): continue
        title = toks(segs[i + 1]); price = None
        if i + 2 < len(segs):
            m = re.search(r"\$([0-9.]+)", segs[i + 2]); price = float(m.group(1)) if m else None
        match = sum(len(toks(a) & title) / max(1, len(toks(a))) for a in attrs if toks(a))
        price_ok = 1.0 if (cap is None or (price is not None and price <= cap)) else 0.0
        best = max(best, (match + price_ok) / (len([a for a in attrs if toks(a)]) + 1))
    return best
NAV = ("next >", "< prev", "back to search", "buy now", "description", "features", "reviews", "attributes")


def ev_goal(e, t, action, obs_plus, reward, done):
    known = known_tokens(e, t); seen = [body(s["obs"]).strip() for s in e["steps"][:t]] + [body(s["next_obs"]).strip() for s in e["steps"][:t]] + [body(e["steps"][t]["obs"]).strip()]
    gw = goal_tokens(e); b = body(obs_plus); bt = toks(b); new = bt - known
    is_product = action.startswith("click[b0") or action.startswith("click[b1"); opt = action[6:-1] if action.startswith("click[") else ""
    loc = 0.0; sat = vio = 0; frac = 0.0
    if done:
        if reward >= 1.0: sat, frac = 1, 1.0
        else: vio = 1
    elif is_product:
        opt_ok, price_ok, n_attr, n_attr_all = product_match(e["goal"], obs_plus); attrs, opts, cap = parse_goal(e["goal"])
        if opt_ok and price_ok and n_attr >= 1:
            sat = 1; frac = (n_attr + len(opts) + 1) / max(1, n_attr_all + len(opts) + 1)
        # else: reversible non-matching product -> no progress (loc = sat = 0)
    elif opt and opt not in NAV and opt in e["goal"].lower():
        sat = 1   # selecting a required option: the page text barely changes, so this must not count as 'no change' (v2 fix)
    else:
        loc = results_loc(e["goal"], obs_plus)   # graded in [0, 1]; 0 if the page lists no product
    nul = 0 if sat else int(b.strip() == seen[-1] or b.strip() in seen[:-1] or (len(new) == 0 and loc == 0))
    score = 2.0 * loc + 2.0 * sat * (1.0 + frac) + min(len(new), 5) / 5.0 - 2.0 * nul - 4.0 * vio
    return {"loc": round(loc, 3), "sat": sat, "frac": round(frac, 3), "vio": vio, "no_info": bool(nul), "n_new": len(new), "score": score}


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
    out = ROOT / a.out; done = {(json.loads(l)["id"], json.loads(l)["t"]) for l in open(out)} if out.exists() else set()
    tok, model = load_policy(a.model, a.adapter); env = WebShopEnv(worker=f"fscan_{os.getpid()}"); t0 = time.time()
    with open(out, "a") as f:
        for n, p in enumerate(ptr):
            if (p["id"], p["t_hat"]) in done: continue
            e = eps[p["id"]]; t = p["t_hat"]; replay(env, e, t); clicks, has_search = env.valid_actions()
            hist = [{"obs": s["obs"], "action": s["action"]} for s in e["steps"][:t]] + [{"obs": e["steps"][t]["obs"], "action": None}]
            fut = e["steps"][t:t + 25]
            diag = "" if a.no_diag else f"Diagnosis (from an error attributor): the first decisive error was at step {t}, type: {p['type']}."
            hind = ("Hindsight: this attempt FAILED. After this point the agent did the following (episode length %d steps):\n" % len(e["steps"])
                    + "\n".join(f"- step {t + i}: {x['action']} -> {' '.join(x['next_obs'].split())[:160]}" for i, x in enumerate(fut)))
            base = render(e["goal"], hist).replace("Next action:", (diag + "\n" if diag else "") + hind + "\n\n")
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


def cmd_pointer_adj(a):
    """ENVIRONMENT ADJUDICATION proposals (same rule as ALFWorld's STEP_POOL): the taken action's own goal-progress evidence at every
    step is read from the log; propose the first-step rule (first non-positive step, stepped back) and the stall-entry rule (start of
    the first 2-step non-positive run, stepped back); both states are scanned and filter_pool keeps the one with the larger verified
    improvement over the taken action."""
    from compare_step_selectors import late_generic, first_nonpos_generic
    ids = [json.loads(l)["id"] for l in open(ROOT / a.ids)]; eps = {e["id"]: e for e in map(json.loads, open(ROOT / a.traj))}; rows = []
    for i in ids:
        e = eps[i]; n = len(e["steps"])
        sc = [ev_goal(e, t, s["action"], s["next_obs"], s["reward"], t == n - 1 and s["action"] == "click[buy now]")["score"] for t, s in enumerate(e["steps"])]
        for t in sorted({first_nonpos_generic(sc), late_generic(sc, 2)}): rows.append({"id": i, "t_hat": max(0, min(t, n - 1)), "type": None})
    open(ROOT / a.out, "w").writelines(json.dumps(x) + "\n" for x in rows)
    ts = [x["t_hat"] for x in rows]; print(f"=== webshop adjudication proposals: {len(rows)} states for {len(ids)} failures, mean t {sum(ts) / len(ts):.2f}, t0 {sum(t == 0 for t in ts) / len(ts):.2f}, dist {sorted(collections.Counter(ts).items())[:10]}")


def cmd_filter_pool(a):
    """per failure: among its scanned states keep the one whose best candidate (score > 0, not no_info) beats the taken action by the
    largest margin (margin > 0); rank failures by the kept candidate's score (tie: sample frequency); top K = frac x failures."""
    rows = collections.defaultdict(list)
    for p in sorted(Path(ROOT / a.scan_dir).glob(a.glob)):
        for l in open(p): r = json.loads(l); rows[r["id"]].append(r)
    nf = sum(1 for _ in open(ROOT / a.ids)); K = max(1, round(a.frac * nf)); picks = []; kinds = collections.Counter()
    for i, rs in rows.items():
        best = None
        for r in rs:
            ok = [c for c in r["cands"] if c["score"] > 0 and not c["no_info"]]
            if not ok: continue
            c = max(ok, key=lambda c: (c["score"], c["freq"])); m = c["score"] - r["taken"]["score"]
            if m > 0 and (best is None or m > best[0]): best = (m, r, c)
        if best: picks.append((best[1]["id"], best[1]["t"], best[2]["cand"], best[2]["score"], best[2]["freq"], best[2], best[0]))
    picks.sort(key=lambda x: (x[3], x[4]), reverse=True); keep = picks[:K]
    for _, _, cand, _, _, c, _ in keep:
        kind = "buy" if cand == "click[buy now]" else "search" if cand.startswith("search[") else "product" if cand.startswith("click[b0") or cand.startswith("click[b1") else "nav" if cand[6:-1] in NAV else "option"
        kinds[(kind, "sat" if c["sat"] else "loc" if c["loc"] > 0 else "novelty")] += 1
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for i, t, cand, s, fr, c, m in keep: f.write(json.dumps({"src": a.traj, "id": i, "t": t, "action": cand, "rank_score": s, "margin": m}) + "\n")
    ts = [x[1] for x in keep]
    print(f"=== webshop final filter_pool: {len(rows)} failures scanned, {len(picks)} with a verified improvement, K = {a.frac} x {nf} = {K}, kept {len(keep)} (mean t {sum(ts) / max(1, len(ts)):.2f}) | kinds {dict(kinds)}")


def cmd_replay(a):
    """self-replay as in the final recipe: the shortest winning sessions, 2 random steps each, n rows (default 100)."""
    rng = random.Random(a.seed); wins = sorted([e for e in map(json.loads, open(ROOT / a.traj)) if e["won"]], key=lambda e: len(e["steps"])); rows = []
    for e in wins:
        for t in rng.sample(range(len(e["steps"])), min(2, len(e["steps"]))): rows.append({"src": a.traj, "id": e["id"], "t": t, "action": e["steps"][t]["action"], "rank_score": 0.0})
        if len(rows) >= a.n: break
    open(ROOT / a.out, "w").writelines(json.dumps(r) + "\n" for r in rows[: a.n]); print(f"=== webshop replay: {min(a.n, len(rows))} steps from the shortest of {len(wins)} winning sessions")


def summarize_step(action, obs, reward, done, prev_title):
    """compact, readable description of what a WebShop step showed (for Model A's input; structured release v2, Oct 8).
    Uses only what the agent itself saw; on the purchase page only 'Purchased ... options ...' is read (the page also prints the hidden
    target, which is never used)."""
    segs = [x.strip() for x in body(obs).split("[SEP]")]; title = prev_title
    if action == "click[buy now]" or "Thank you for shopping" in obs:
        opts = ""
        for i, sg in enumerate(segs):
            if sg == "options" and i + 1 < len(segs): opts = segs[i + 1]; break
        return f"Purchased '{(prev_title or '?')[:90]}' with options {opts}. Episode ended with reward {reward:.2f} (1.00 = fully correct).", title
    m = [i for i, sg in enumerate(segs) if re.fullmatch(r"[bB]0[0-9A-Za-z]{8}", sg)]
    page = next((sg for sg in segs if sg.startswith("Page ")), None)
    if m and page:
        items = []
        for i in m[:5]:
            t = segs[i + 1][:70] if i + 1 < len(segs) else "?"; pr = segs[i + 2] if i + 2 < len(segs) else ""
            items.append(f"[{segs[i].upper()}] {t} ({pr})")
        return f"Search results, {page}. Top items: " + "; ".join(items), title
    pi = next((i for i, sg in enumerate(segs) if sg.startswith("Price:")), None)
    if pi is not None:
        title = segs[pi - 1] if pi > 0 else "?"
        start = segs.index("< Prev") + 1 if "< Prev" in segs else 0
        opts = [x for x in segs[start:pi - 1] if x and x not in ("Description", "Features", "Reviews", "Attributes", "Buy Now")]
        sel = f" Selected option '{action[6:-1]}'." if action.startswith("click[") and not action.startswith("click[b0") and not action.startswith("click[b1") else ""
        return f"Product page: '{title[:90]}', {segs[pi]}. Options listed: {', '.join(opts[:30])}.{sel}", title
    return " ".join(body(obs).split())[:300], title


def cmd_release_struct(a):
    eps = [json.loads(l) for l in open(ROOT / a.traj)][: a.first]; n = 0
    with open(ROOT / a.out, "w") as f:
        for e in eps:
            if e["won"]: continue
            traj = [{"kind": "user", "content": "Shopping instruction: " + e["goal"].replace("Instruction:", "").strip()}]; title = None
            for k, st in enumerate(e["steps"]):
                last = k == len(e["steps"]) - 1; summ, title = summarize_step(st["action"], st["next_obs"], st.get("reward") or 0.0, last, title)
                traj.append({"kind": "action", "step_number": str(k + 1), "reasoning": "None", "action": st["action"], "observation": summ, "is_final_answer": str(last)})
            f.write(json.dumps({"id": e["id"], "framework": "alfagent", "benchmark": "webshop", "modality": "text", "task": {"query": e["goal"].replace("Instruction:", "").strip(), "answer": ""},
                                "trajectory": traj, "ground_truth": {}, "extras": {"agents": [{"id": "agent", "role": "agent"}]}}) + "\n"); n += 1
    print(f"structured release: {n} failed sessions")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("release"); s.add_argument("--traj", required=True); s.add_argument("--out", required=True); s.add_argument("--ids", required=True); s.add_argument("--first", type=int, default=1000)
    s = sub.add_parser("pointer"); s.add_argument("--traj", required=True); s.add_argument("--pred", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("scan"); s.add_argument("--traj", required=True); s.add_argument("--pointer", required=True); s.add_argument("--out", required=True)
    s.add_argument("--adapter", required=True); s.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); s.add_argument("--shard", type=int, nargs=2, default=None); s.add_argument("--no_diag", action="store_true")
    s = sub.add_parser("release_struct"); s.add_argument("--traj", required=True); s.add_argument("--out", required=True); s.add_argument("--first", type=int, default=1000)
    s = sub.add_parser("pointer_adj"); s.add_argument("--traj", required=True); s.add_argument("--ids", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("filter_pool"); s.add_argument("--scan_dir", required=True); s.add_argument("--glob", default="scan_s*.jsonl"); s.add_argument("--ids", required=True); s.add_argument("--traj", required=True); s.add_argument("--out", required=True); s.add_argument("--frac", type=float, default=0.2)
    s = sub.add_parser("replay"); s.add_argument("--traj", required=True); s.add_argument("--out", required=True); s.add_argument("--n", type=int, default=100); s.add_argument("--seed", type=int, default=1)
    s = sub.add_parser("filter"); s.add_argument("--scan_dir", required=True); s.add_argument("--ids", required=True); s.add_argument("--traj", required=True); s.add_argument("--out", required=True); s.add_argument("--frac", type=float, default=0.2)
    a = ap.parse_args(); {"release": cmd_release, "pointer": cmd_pointer, "scan": cmd_scan, "filter": cmd_filter, "pointer_adj": cmd_pointer_adj, "release_struct": cmd_release_struct, "filter_pool": cmd_filter_pool, "replay": cmd_replay}[a.cmd](a)


if __name__ == "__main__":
    main()
