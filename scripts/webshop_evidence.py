"""WebShop instantiation of the environment-evidence pipeline (mirrors sciworld_evidence.py):
  scan    -- for each failure state (id, t): sample n candidate actions from the policy, replay the episode prefix on the server,
             execute each candidate ONE step, record its one-step evidence (+ the taken action's evidence from the log)
  label   -- env-evidence attribution: first step whose best alternative beats the taken action by >= margin
  filter  -- env_top | conf_top | placebo | all   (same interface as ScienceWorld)
  train   -- one-hot continued LoRA on (state, action) targets from the student adapter
One-step evidence (generic + the two WebShop terminal facts): new tokens on the new page (information), overlap of the new tokens
with the instruction (relevance), progress = a purchase with reward 1 or opening a product page that matches the instruction,
regress = a purchase that ends the episode with reward < 1 (irreversible wrong buy), no_info = page unchanged.
"""
import argparse, collections, difflib, json, os, random, re, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("ERROR_ATTR_ROOT", str(ROOT)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from error_attr.env.webshop_env import WebShopEnv
from webshop_agent import render, chat, map_action, SYSTEM
STOP = set("a an the of to in on at is are was be you your it its this that with and or for from into onto as by see also called there here now not no yes find me lower than dollars price".split())


def toks(s): return {w for w in re.findall(r"[a-z][a-z\-]+", s.lower()) if w not in STOP and len(w) > 2}


def body(obs):
    """page text without the instruction banner (every WebShop page repeats 'Instruction: [SEP] <goal> [SEP]' at the top; counting
    the goal words there as 'known' killed the relevance term in the first run, Oct 5 19:50)."""
    parts = obs.split("[SEP]")
    return "[SEP]".join(parts[2:]) if len(parts) > 2 and parts[0].strip().lower().startswith("instruction") else obs


def goal_tokens(e):
    return toks(e["goal"].replace("Instruction:", ""))


def known_tokens(e, t):
    k = set()
    for s in e["steps"][:t]: k |= toks(body(s["obs"])) | toks(body(s["next_obs"]))
    return k | toks(body(e["steps"][t]["obs"]))


def parse_goal(goal):
    """instruction -> (attribute phrases, required option values, price cap); all three are stated verbatim in the instruction."""
    g = goal.replace("Instruction:", "").strip().lower()
    m = re.search(r"price lower than ([0-9.]+)", g); cap = float(m.group(1)) if m else None
    opts = [v.strip() for _, v in re.findall(r"(?:with|and) ([a-z ]+?): ([^,]+?)(?:,| and | with |$)", g)]
    head = re.split(r" with [a-z ]+?: ", g)[0]; head = re.sub(r"^find me ", "", head)
    attrs = [a.strip() for a in re.split(r" with |, ", head) if len(a.strip()) > 2]
    return attrs, opts, cap


def product_match(goal, page):
    """verifiable facts on a product page: required options offered, price within cap, attribute phrases present."""
    attrs, opts, cap = parse_goal(goal); segs = [x.strip().lower() for x in page.split("[SEP]")]; text = " ".join(segs)
    opt_ok = all(o in segs for o in opts) if opts else True
    pm = re.search(r"price: \$([0-9.]+)", text); price_ok = (float(pm.group(1)) <= cap) if (pm and cap) else True
    n_attr = sum(a in text for a in attrs)
    return opt_ok, price_ok, n_attr, len(attrs)


def evidence(e, t, action, obs_plus, reward, done, known):
    gw = goal_tokens(e); bt = toks(body(obs_plus)); new = bt - known; overlap = len(new & gw); match = len(bt & gw)
    unchanged = body(obs_plus).strip() == body(e["steps"][t]["obs"]).strip()
    is_product = action.startswith("click[b0") or action.startswith("click[b1")
    opt = action[6:-1] if action.startswith("click[") else ""
    opt_ok = price_ok = True; n_attr = 0
    if is_product: opt_ok, price_ok, n_attr, _ = product_match(e["goal"], obs_plus)
    # progress = a verifiable commitment: bought with reward 1; opened a product that offers every required option, is within the
    # price cap and carries at least one attribute phrase of the instruction; or selected an option value the instruction asks for
    progress = (done and reward >= 1.0) or (is_product and opt_ok and price_ok and n_attr >= 1) or (bool(opt) and not is_product and opt not in ("next >", "< prev", "back to search", "buy now", "description", "features", "reviews") and opt in e["goal"].lower())
    # regress = an irreversible wrong commitment: bought with reward < 1, or opened a product that fails a stated requirement
    regress = bool(done and reward < 1.0) or (is_product and not (opt_ok and price_ok))
    no_info = unchanged or (len(new) == 0 and not progress)
    return {"valid": not unchanged, "n_new": len(new), "overlap": overlap, "match": match, "n_attr": n_attr, "opt_ok": opt_ok, "price_ok": price_ok,
            "progress": bool(progress), "regress": bool(regress), "no_info": bool(no_info),
            "score": 2.0 * (overlap > 0) + 2.0 * progress - 4.0 * regress + min(len(new), 5) / 5.0 - 2.0 * no_info}


def load_scan(d):
    scan = {}
    for f_ in sorted(Path(d).glob("scan*.jsonl")):
        for l in open(f_):
            try: r = json.loads(l)
            except Exception: continue
            scan.setdefault((r["id"], r["t"]), r)
    return scan


def load_policy(model_name, adapter):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(model_name); tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map={"": 0})
    if adapter: model = PeftModel.from_pretrained(model, str(ROOT / adapter))
    return tok, model.eval()


def cmd_scan(a):
    import torch
    eps = {e["id"]: e for e in map(json.loads, open(ROOT / a.traj))}
    ids = [json.loads(l)["id"] for l in open(ROOT / a.ids)] if a.ids else [i for i, e in eps.items() if not e["won"]]
    out = ROOT / a.out_dir / (f"scan_s{a.shard[0]}.jsonl" if a.shard else "scan.jsonl"); out.parent.mkdir(parents=True, exist_ok=True)
    done = set(load_scan(ROOT / a.out_dir))
    todo = [(i, t) for i in ids for t in range(min(a.k, len(eps[i]["steps"]))) if (i, t) not in done]
    if a.shard: todo = todo[a.shard[0]::a.shard[1]]
    print(f"{len(todo)} (id, t) states", flush=True)
    tok, model = load_policy(a.model, a.adapter); env = WebShopEnv(worker=f"scan{a.shard[0] if a.shard else 0}_{os.getpid()}"); t0 = time.time()
    with open(out, "a") as f:
        for n, (i, t) in enumerate(todo):
            e = eps[i]
            def replay():
                obs, _ = env.reset(e["session"])
                for s in e["steps"][:t]: obs, r, dn = env.step(s["action"])
                return obs
            replay(); clicks, has_search = env.valid_actions()
            hist = [{"obs": s["obs"], "action": s["action"]} for s in e["steps"][:t]] + [{"obs": e["steps"][t]["obs"], "action": None}]
            ids_ = tok(chat(tok, render(e["goal"], hist, a.history)), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            with torch.no_grad():
                g = model.generate(ids_, max_new_tokens=32, do_sample=True, temperature=1.0, top_p=1.0, num_return_sequences=a.n_samples, pad_token_id=tok.pad_token_id)
            raws = [tok.decode(x[ids_.shape[1]:], skip_special_tokens=True).strip().split("\n")[0] for x in g]
            a_minus = e["steps"][t]["action"]; cnt = collections.Counter()
            for r in raws:
                act, _ = map_action(r, clicks, has_search)
                if act != a_minus: cnt[act] += 1
            cands = [c for c, _ in cnt.most_common(2)]; known = known_tokens(e, t); rows = []
            for c in cands:
                if rows: replay()
                o, r, dn = env.step(c); ev = evidence(e, t, c, o, r, dn, known)
                rows.append({"cand": c, "freq": cnt[c] / a.n_samples, "obs_plus": o[:3000], "reward": r, "done": dn, **ev})
            st = e["steps"][t]; taken_ev = evidence(e, t, a_minus, st["next_obs"], st["reward"], t == len(e["steps"]) - 1 and st["reward"] > 0, known)
            f.write(json.dumps({"id": i, "t": t, "a_minus": a_minus, "taken": taken_ev, "cands": rows}) + "\n"); f.flush()
            if n % 25 == 0: print(f"[{n + 1}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
    print("DONE scan", flush=True)


def cmd_rescore(a):
    eps = {e["id"]: e for e in map(json.loads, open(ROOT / a.traj))}; rows = list(load_scan(ROOT / a.out_dir).values()); n = 0
    arch = ROOT / a.out_dir / "archive_scan_v1"; arch.mkdir(exist_ok=True)
    for f_ in sorted((ROOT / a.out_dir).glob("scan*.jsonl")): f_.rename(arch / f_.name)
    with open(ROOT / a.out_dir / "scan_v2.jsonl", "w") as f:
        for r in rows:
            e = eps[r["id"]]; t = r["t"]; known = known_tokens(e, t)
            for c in r["cands"]: c.update(evidence(e, t, c["cand"], c["obs_plus"], c["reward"], c["done"], known))
            st = e["steps"][t]; r["taken"] = evidence(e, t, r["a_minus"], st["next_obs"], st["reward"], t == len(e["steps"]) - 1 and st["reward"] > 0, known)
            f.write(json.dumps(r) + "\n"); n += 1
    print(f"DONE rescore {n} states", flush=True)


def cmd_label(a):
    rows = list(load_scan(ROOT / a.out_dir).values()); per = {}
    for r in rows:
        if not r["cands"]: continue
        best = max(r["cands"], key=lambda c: c["score"]); per.setdefault(r["id"], {})[r["t"]] = {"margin": best["score"] - r["taken"]["score"], "best_cand": best["cand"], "a_minus": r["a_minus"]}
    n_thr = 0
    with open(ROOT / a.out_dir / "labels.jsonl", "w") as f, open(ROOT / a.out_dir / "steps.jsonl", "w") as g:
        for i, d in per.items():
            ts = sorted(d); hit = [t for t in ts if d[t]["margin"] >= a.margin]; t_star = hit[0] if hit else max(ts, key=lambda t: d[t]["margin"]); n_thr += bool(hit)
            f.write(json.dumps({"id": i, "t_star": t_star, "reached_threshold": bool(hit), "margin": d[t_star]["margin"], "best_cand": d[t_star]["best_cand"], "a_minus": d[t_star]["a_minus"]}) + "\n")
            g.write(json.dumps({"id": i, "steps": [t_star], "mode": None}) + "\n")
    import statistics as st
    ls = [json.loads(l)["t_star"] for l in open(ROOT / a.out_dir / "labels.jsonl")]
    print(f"=== webshop env labels: {len(ls)} failures | threshold reached {n_thr} ({n_thr / max(1, len(ls)):.0%}) | t* mean {st.mean(ls):.1f} median {st.median(ls):.0f} | t0 {sum(t == 0 for t in ls) / len(ls):.2f}", flush=True)


def cmd_filter(a):
    rng = random.Random(a.seed); scan = load_scan(ROOT / a.out_dir)
    steps = [(json.loads(l)["id"], json.loads(l)["steps"][0]) for l in open(ROOT / a.steps)]; picks = []
    for i, t in steps:
        r = scan.get((i, t))
        if not r or not r["cands"]: continue
        if a.rule == "placebo": c = rng.choice(r["cands"]); picks.append((i, t, c["cand"], rng.random()))
        elif a.rule == "conf_top": c = max(r["cands"], key=lambda c: c["freq"]); picks.append((i, t, c["cand"], c["freq"]))
        elif a.rule == "all": c = max(r["cands"], key=lambda c: c["freq"]); picks.append((i, t, c["cand"], 1.0))
        else:
            ok = [c for c in r["cands"] if not c["no_info"] and c["score"] > 0]
            if not ok: continue
            c = max(ok, key=lambda c: (c["score"], c["freq"])); picks.append((i, t, c["cand"], c["score"]))
    if a.rule != "all": picks = sorted(picks, key=lambda x: x[3], reverse=True)[: a.k]
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for i, t, c, s in picks: f.write(json.dumps({"src": a.traj, "id": i, "t": t, "action": c, "rank_score": s}) + "\n")
    print(f"=== webshop filter {a.rule}: {len(picks)} targets", flush=True)


def cmd_train(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    rows = [json.loads(l) for l in open(ROOT / a.data)]; srcs = {}
    for r in rows:
        if r["src"] not in srcs: srcs[r["src"]] = {e["id"]: e for e in map(json.loads, open(ROOT / r["src"]))}
    tok = AutoTokenizer.from_pretrained(a.model); tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}); base.gradient_checkpointing_enable(); base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, str(ROOT / a.adapter), is_trainable=True); model.train()
    params = [p for p in model.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    rng = random.Random(a.seed); torch.manual_seed(a.seed); items = []
    for r in rows:
        e = srcs[r["src"]][r["id"]]; t = r["t"]
        hist = [{"obs": s["obs"], "action": s["action"]} for s in e["steps"][:t]] + [{"obs": e["steps"][t]["obs"], "action": None}]
        items.append((render(e["goal"], hist, a.history), r["action"]))
    rng.shuffle(items); n = 0; run = 0.0
    for ep in range(a.epochs):
        for u, act in items:
            p = tok(chat(tok, u), add_special_tokens=False).input_ids; tt = tok(act, add_special_tokens=False).input_ids + [tok.eos_token_id]
            ids = torch.tensor([(p + tt)[-a.max_len:]], device=model.device); lab = torch.tensor([([-100] * len(p) + tt)[-a.max_len:]], device=model.device)
            loss = model(input_ids=ids, labels=lab).loss; (loss / a.batch).backward(); run += loss.item(); n += 1
            if n % a.batch == 0: torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
            if n % 20 == 0: print(f"[{n}/{len(items) * a.epochs}] loss {run / 20:.3f}", flush=True); run = 0.0
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True); model.save_pretrained(str(out / "adapter")); tok.save_pretrained(str(out / "adapter"))
    print(f"DONE train {len(items)} examples", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan"); s.add_argument("--traj", required=True); s.add_argument("--ids", default=None); s.add_argument("--out_dir", required=True); s.add_argument("--k", type=int, default=15)
    s.add_argument("--n_samples", type=int, default=6); s.add_argument("--history", type=int, default=6); s.add_argument("--shard", type=int, nargs=2, default=None); s.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); s.add_argument("--adapter", default=None)
    l = sub.add_parser("label"); l.add_argument("--out_dir", required=True); l.add_argument("--margin", type=float, default=2.0)
    f = sub.add_parser("filter"); f.add_argument("--out_dir", required=True); f.add_argument("--steps", required=True); f.add_argument("--traj", required=True); f.add_argument("--rule", default="env_top"); f.add_argument("--k", type=int, default=86); f.add_argument("--seed", type=int, default=0); f.add_argument("--out", required=True)
    t = sub.add_parser("train"); t.add_argument("--data", required=True); t.add_argument("--adapter", required=True); t.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); t.add_argument("--out", required=True)
    t.add_argument("--lr", type=float, default=1e-5); t.add_argument("--epochs", type=int, default=1); t.add_argument("--batch", type=int, default=8); t.add_argument("--seed", type=int, default=0); t.add_argument("--history", type=int, default=6); t.add_argument("--max_len", type=int, default=3072)
    r = sub.add_parser("rescore"); r.add_argument("--traj", required=True); r.add_argument("--out_dir", required=True)
    a = ap.parse_args(); {"scan": cmd_scan, "rescore": cmd_rescore, "label": cmd_label, "filter": cmd_filter, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
