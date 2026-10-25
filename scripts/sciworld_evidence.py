"""ScienceWorld instantiation of the method (plan doc: 统一框架 / 主表 3).  Same recipe as ALFWorld, generic evidence only:
  scan   -- GPU+env: for each failed trajectory and each of its first --k steps, sample --n_samples actions from the policy (T=1),
            map them to valid actions (difflib), drop the taken action a-, keep the 2 most frequent distinct candidates, replay the
            prefix and execute each candidate ONE step; record the new observation, the environment's score delta, generic evidence.
            Resumable per (id, t).  -> <out_dir>/scan.jsonl
  label  -- CPU: env-evidence attribution: t* = first step where (best alternative evidence) - (taken action evidence) >= --margin;
            type from the verb difference.  -> labels.jsonl, steps.jsonl
  filter -- CPU: at t*, pick the candidate by --rule (env_top | conf_top | placebo | all), rank states, top --k -> train.jsonl
  train  -- GPU: LoRA continued from --adapter on (prefix -> action) pairs, 1 epoch.  -> <out>/adapter
Generic evidence from the new observation o+ (no task parsing):
  novelty   = number of content tokens in o+ never seen before (cap 5)
  overlap   = some new token also appears in the task description
  progress  = the environment's own score increased (potential-based)
  valid     = o+ is not an "unknown/ambiguous action" message
  no_info   = not valid, or (no new token and no progress and o+ == current observation)
  score     = 2*overlap + 2*progress + min(novelty,5)/5 - 2*no_info
"""
import argparse, collections, difflib, json, os, random, re, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("ERROR_ATTR_ROOT", str(ROOT)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from sciworld_agent import render, chat, SYSTEM  # noqa

STOP = set("a an the of to in on at is are was be you your it its this that with and or for from into onto as by see also called there here now not no yes".split())
BAD = ("no known action", "ambiguous request", "i'm not sure", "that's not", "unknown")


def toks(s):
    return {w for w in re.findall(r"[a-z][a-z\-]+", s.lower()) if w not in STOP and len(w) > 2}


def known_tokens(e, t):
    k = toks(e.get("initial_obs", ""))
    for s in e["steps"][:t]: k |= toks(s["obs"]) | toks(s["next_obs"])
    k |= toks(e["steps"][t]["obs"])
    return k


def evidence(e, t, obs_plus, score_before, score_after, known):
    ol = obs_plus.lower(); valid = not any(ol.startswith(b) for b in BAD); new = (toks(obs_plus) - known) if valid else set()
    gw = toks(e["goal"]); overlap = bool(new & gw); progress = score_after > score_before; regress = score_after < score_before
    no_info = (not valid) or (len(new) == 0 and not progress and obs_plus.strip() == e["steps"][t]["obs"].strip())
    # regress: the environment's own score DROPPED (ScienceWorld: focusing on a wrong object ends the task at -100) -- the
    # potential-based progress term with a negative sign; it is what makes the catastrophic step attributable.
    return {"valid": valid, "n_new": len(new), "overlap": overlap, "progress": progress, "regress": regress, "no_info": no_info,
            "score": 2.0 * overlap + 2.0 * progress - 4.0 * regress + min(len(new), 5) / 5.0 - 2.0 * no_info}


def load_scan(d):
    """Union of scan*.jsonl (one file per shard), de-duplicated by (id, t); corrupt lines skipped."""
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
    m = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map={"": 0})
    if adapter: m = PeftModel.from_pretrained(m, str(ROOT / adapter))
    m.eval(); return tok, m


def cmd_scan(a):
    import torch
    from error_attr.env.scienceworld_env import make, reset, step, valid_actions
    eps = {e["id"]: e for e in map(json.loads, open(ROOT / a.traj))}
    ids = [json.loads(l)["id"] for l in open(ROOT / a.ids)] if a.ids else [i for i, e in eps.items() if not e["won"]]
    out = ROOT / a.out_dir / (f"scan_s{a.shard[0]}.jsonl" if a.shard else "scan.jsonl"); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    for f_ in sorted((ROOT / a.out_dir).glob("scan*.jsonl")):
        for l in open(f_):
            try: r = json.loads(l); done.add((r["id"], r["t"]))
            except Exception: pass
    todo = [(i, t) for i in ids for t in range(min(a.k, len(eps[i]["steps"]))) if (i, t) not in done]
    if a.shard: todo = todo[a.shard[0]::a.shard[1]]
    print(f"{len(todo)} (id, t) states", flush=True)
    tok, model = load_policy(a.model, a.adapter); env = make(200); t0 = time.time()
    with open(out, "a") as f:
        for n, (i, t) in enumerate(todo):
            e = eps[i]
            def replay():
                obs, info = reset(env, e["task"], e["variation"]); sc = 0
                for s in e["steps"][:t]: obs, sc, dn, info = step(env, s["action"])
                return obs, sc
            obs, sc0 = replay(); va = valid_actions(env)
            hist = [{"obs": s["obs"], "action": s["action"]} for s in e["steps"][:t]] + [{"obs": e["steps"][t]["obs"], "action": None}]
            ids_ = tok(chat(tok, render(e["goal"], hist, a.history)), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            with torch.no_grad():
                g = model.generate(ids_, max_new_tokens=24, do_sample=True, temperature=1.0, top_p=1.0, num_return_sequences=a.n_samples, pad_token_id=tok.pad_token_id)
            raws = [tok.decode(x[ids_.shape[1]:], skip_special_tokens=True).strip().split("\n")[0] for x in g]
            a_minus = e["steps"][t]["action"]; cnt = collections.Counter()
            for r in raws:
                act = r if r in va else (difflib.get_close_matches(r, va, n=1, cutoff=0.0) or [r])[0]
                if act != a_minus: cnt[act] += 1
            cands = [c for c, _ in cnt.most_common(2)]
            known = known_tokens(e, t); rows = []
            for c in cands:
                if rows: obs, sc0 = replay()
                o, sc1, dn, info = step(env, c)
                ev = evidence(e, t, o, sc0, sc1, known)
                rows.append({"cand": c, "freq": cnt[c] / a.n_samples, "obs_plus": o[:400], "score_before": sc0, "score_after": sc1, "done": dn, **ev})
            taken_ev = evidence(e, t, e["steps"][t]["next_obs"], sc0, e["steps"][t]["score"], known)
            f.write(json.dumps({"id": i, "t": t, "a_minus": a_minus, "taken": taken_ev, "cands": rows}) + "\n"); f.flush()
            if n % 25 == 0: print(f"[{n + 1}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
    print("DONE scan", flush=True)


def verb(a): return a.split()[0] if a else ""


def cmd_label(a):
    rows = list(load_scan(ROOT / a.out_dir).values()); per = {}
    for r in rows:
        if not r["cands"]: continue
        best = max(r["cands"], key=lambda c: c["score"]); per.setdefault(r["id"], {})[r["t"]] = {"taken": r["taken"]["score"], "best": best["score"], "best_cand": best["cand"], "a_minus": r["a_minus"], "margin": best["score"] - r["taken"]["score"]}
    n_thr = 0
    with open(ROOT / a.out_dir / "labels.jsonl", "w") as f, open(ROOT / a.out_dir / "steps.jsonl", "w") as g:
        for i, d in per.items():
            ts = sorted(d); hit = [t for t in ts if d[t]["margin"] >= a.margin]; t_star = hit[0] if hit else max(ts, key=lambda t: d[t]["margin"]); n_thr += bool(hit)
            am, bc = d[t_star]["a_minus"], d[t_star]["best_cand"]
            typ = "skip" if verb(am) in ("look", "examine", "wait", "inventory", "task") else ("wrong_target" if verb(am) == verb(bc) else "wrong_op")
            f.write(json.dumps({"id": i, "t_star": t_star, "type": typ, "reached_threshold": bool(hit), "margin": d[t_star]["margin"], "best_cand": bc}) + "\n")
            g.write(json.dumps({"id": i, "steps": [t_star], "mode": None}) + "\n")
    import statistics as st
    ls = [json.loads(l)["t_star"] for l in open(ROOT / a.out_dir / "labels.jsonl")]
    print(f"=== sciworld env labels: {len(ls)} failures | threshold reached {n_thr} ({n_thr / max(1, len(ls)):.0%}) | t* mean {st.mean(ls):.1f} median {st.median(ls):.0f} | t0 {sum(t == 0 for t in ls) / len(ls):.2f} t<=2 {sum(t <= 2 for t in ls) / len(ls):.2f} t>=5 {sum(t >= 5 for t in ls) / len(ls):.2f}", flush=True)


def cmd_filter(a):
    rng = random.Random(a.seed)
    scan = load_scan(ROOT / a.out_dir)
    steps = [(json.loads(l)["id"], json.loads(l)["steps"][0]) for l in open(ROOT / a.steps)]
    picks = []
    for i, t in steps:
        r = scan.get((i, t))
        if not r or not r["cands"]: continue
        if a.rule == "placebo": c = rng.choice(r["cands"]); picks.append((i, t, c["cand"], rng.random()))
        elif a.rule == "conf_top": c = max(r["cands"], key=lambda c: c["freq"]); picks.append((i, t, c["cand"], c["freq"]))
        elif a.rule == "all": c = max(r["cands"], key=lambda c: c["freq"]); picks.append((i, t, c["cand"], 1.0))
        else:   # env_top: veto no-info, best evidence, rank states by it
            ok = [c for c in r["cands"] if not c["no_info"] and c["score"] > 0]
            if not ok: continue
            c = max(ok, key=lambda c: (c["score"], c["freq"])); picks.append((i, t, c["cand"], c["score"]))
    if a.rule != "all": picks = sorted(picks, key=lambda x: x[3], reverse=True)[: a.k]
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for i, t, c, s in picks: f.write(json.dumps({"src": a.traj, "id": i, "t": t, "action": c, "rank_score": s}) + "\n")
    print(f"=== sciworld filter {a.rule}: {len(picks)} targets", flush=True)


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
    s = sub.add_parser("scan"); s.add_argument("--traj", required=True); s.add_argument("--ids", default=None); s.add_argument("--out_dir", required=True); s.add_argument("--k", type=int, default=8)
    s.add_argument("--n_samples", type=int, default=6); s.add_argument("--history", type=int, default=10); s.add_argument("--shard", type=int, nargs=2, default=None); s.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); s.add_argument("--adapter", required=True)
    l = sub.add_parser("label"); l.add_argument("--out_dir", required=True); l.add_argument("--margin", type=float, default=2.0)
    f = sub.add_parser("filter"); f.add_argument("--out_dir", required=True); f.add_argument("--steps", required=True); f.add_argument("--traj", required=True); f.add_argument("--rule", default="env_top"); f.add_argument("--k", type=int, default=50); f.add_argument("--seed", type=int, default=0); f.add_argument("--out", required=True)
    t = sub.add_parser("train"); t.add_argument("--data", required=True); t.add_argument("--adapter", required=True); t.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); t.add_argument("--out", required=True)
    t.add_argument("--lr", type=float, default=1e-5); t.add_argument("--epochs", type=int, default=1); t.add_argument("--batch", type=int, default=8); t.add_argument("--seed", type=int, default=0); t.add_argument("--history", type=int, default=10); t.add_argument("--max_len", type=int, default=3072)
    a = ap.parse_args(); {"scan": cmd_scan, "label": cmd_label, "filter": cmd_filter, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
