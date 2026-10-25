"""HINT-SD ONLINE baseline on ScienceWorld (Oct 11 2026; port of scripts/hintsd_online.py, the ALFWorld online run in the paper).
Algorithm, losses, optimizer, EMA, analyzer prompt/parser and state handling are IMPORTED from scripts/hintsd_online.py (identical code);
only the environment / policy parts are ScienceWorld-specific:

  every optimizer step   B=2 tasks x 4 rollouts (8 episodes) with the CURRENT student = the generative ScienceWorld policy of
                         scripts/sciworld_agent.py collect (same render/chat prompt, history 10, max_new_tokens 24, T=1.0, top_p 1.0, the
                         same HF generate call / generation_config, first line of the output, --snap safe, budget 50, env step limit 55);
                         the 8 episodes are stepped in parallel (8 env worker processes, one ScienceWorld JVM each) and the policy is batched
  failed rollouts only   EMA teacher writes <= 3 (step, feedback) pairs with the official multi-step prompt (wording "ALFWorld household-task"
                         -> "ScienceWorld science-experiment"), T=0.6, top_p 0.95, <= 512 tokens, official parser
  teacher / loss / EMA   as ALFWorld: teacher prompt = student prompt + "Consider the following feedback to solve this step:\\n\\n{hint}" turn;
                         token-level reverse KL(student || stop-grad teacher), student top-100 + tail, on the student's sampled action tokens
                         (the raw generated first line, before any --snap safe mapping); lr 5e-6, warmup 5%, linear decay, AdamW, clip 1.0,
                         EMA 0.999; no environment verification, no replay
Budget: the 1,483 ScienceWorld train variations our method collects each round (collect --split train --per_task 75 --seed 1, same
seeded per-task shuffle), shuffled once with --seed 0 (as ALFWorld's game list; the 1,483 list is grouped by task); the first
ceil(4449 / 4) = 1,113 variations x 4 rollouts, truncated to 4,449 episodes (= our 3 rounds x 1,483).  Checkpoints EXACTLY at 1,483 / 2,966
/ 4,449 cumulative episodes (ckpt1..3/adapter): the optimizer step that would cross a checkpoint is cut at it (ALFWorld logic would save at
the first step boundary >= the threshold, i.e. 1,488 / 2,968 / 4,449 episodes with 8 episodes per step).
Resumable: state/ (student + teacher adapters, AdamW + scheduler, counters) every --save_every steps and at every checkpoint.

    python scripts/hintsd_online_sciworld.py --out reports/sciworld/stepfix/hsdon_train
"""
import argparse, json, math, os, random, sys, time
import os as _os_umask; _os_umask.umask(0o027)   # Oct 12: some srun steps inherit umask 0117 (dirs created without x); force 0027
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("ERROR_ATTR_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import hintsd_online as H   # light: heavy imports live inside its functions

ANALYZER_SYSTEM = H.ANALYZER_SYSTEM.replace("ALFWorld household-task", "ScienceWorld science-experiment")
ANALYZER_USER = H.ANALYZER_USER.replace("ALFWorld attempt", "ScienceWorld attempt")
assert "ALFWorld" not in ANALYZER_SYSTEM + ANALYZER_USER


# =============================================================== env workers (one ScienceWorld JVM per process, reused across episodes)
def _env_worker(conn, root, budget):
    sys.path.insert(0, os.path.join(root, "src")); os.environ.setdefault("ERROR_ATTR_ROOT", root)
    import difflib
    from error_attr.env.scienceworld_env import make, reset, step, valid_actions
    env = make(budget + 5)   # = sciworld_agent.py collect: make(max(budget) + 5)
    while True:
        try:
            msg = conn.recv()
        except (EOFError, KeyboardInterrupt):
            break
        try:
            if msg[0] == "reset":
                obs, info = reset(env, msg[1], msg[2]); conn.send(("ok", obs, info["taskDesc"]))
            elif msg[0] == "act":   # verbatim --snap safe logic of sciworld_agent.py collect
                raw = msg[1]; va = valid_actions(env); act = raw; mapped = False
                if raw not in va:
                    vb = raw.split(" ")[0] if raw else ""
                    pool = [x for x in va if x.split(" ")[0] == vb and not (x.startswith("focus on") and not raw.startswith("focus on"))]
                    m = difflib.get_close_matches(raw, pool, n=1, cutoff=0.85); act = m[0] if m else raw; mapped = act != raw
                nxt, score, dn, info = step(env, act)
                conn.send(("ok", act, mapped, nxt, score, bool(dn)))
            elif msg[0] == "variations":   # the seeded train job list of collect --per_task P --seed S (same code path)
                rng = random.Random(msg[2]); jobs = []
                for task in env.get_task_names():
                    env.load(task, 0); vs = list(env.get_variations_train()); rng.shuffle(vs); jobs += [(task, v) for v in vs[: msg[1]]]
                conn.send(("ok", jobs))
            elif msg[0] == "close":
                break
        except Exception as e:
            try: conn.send(("err", repr(e)))
            except Exception: break
    try: env.close()
    except Exception: pass


class EnvPool(H.EnvPool):
    def _start(self, i):
        a, b = self.ctx.Pipe()
        p = self.ctx.Process(target=_env_worker, args=(b, str(ROOT), self.budget)); p.start(); b.close()
        self.w[i] = (p, a); self.uses[i] = 0


def sw_render(goal, steps_upto, obs_t, history=10):
    from sciworld_agent import render
    return render(goal, [{"obs": s["obs"], "action": s["action"]} for s in steps_upto] + [{"obs": obs_t, "action": None}], history)


_NL = {}
def _newline_stop(tok, plen):
    """Per-row stop once the row has generated a token containing a newline. Only the first output line is used as the action
    (collect: .split("\n")[0]), and later tokens never influence earlier ones, so this is output-identical to collect; it only stops
    the batch from running to max_new_tokens for rows that keep writing after the first line (batched decoding waits for the slowest row)."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    if "ids" not in _NL:
        ids = [i for i in range(len(tok)) if "\n" in tok.decode([i])]
        _NL["ids"] = torch.tensor(ids)

    class NL(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            gen = input_ids[:, plen:]
            return torch.isin(gen, _NL["ids"].to(gen.device)).any(dim=1)
    return StoppingCriteriaList([NL()])


def rollout(pool, model, tok, jobs, budget, temperature, step, history):
    """len(jobs) episodes in parallel, the sciworld_agent.py collect episode loop with a batched policy."""
    import torch
    from sciworld_agent import chat
    eps = [{"id": f"hsdon_s{step:04d}_{t}_{v}_r{k}", "task": t, "variation": v, "split": "train", "rollout": k, "task_idx": ti, "steps": [],
            "won": False, "final_score": 0, "done": False} for ti, t, v, k in jobs]
    pool.recycle_if_needed()
    rep = pool.request({i: ("reset", e["task"], e["variation"]) for i, e in enumerate(eps)}); obs = {}
    for i, e in enumerate(eps):
        pool.uses[i] += 1
        if rep[i][0] != "ok": e.update(done=True, error=f"reset: {rep[i][1]}"); continue
        obs[i] = rep[i][1]; e["goal"] = rep[i][2]; e["initial_obs"] = rep[i][1]
    gen_s = env_s = 0.0; dev = next(model.parameters()).device
    for k in range(budget):
        act = [i for i, e in enumerate(eps) if not e["done"]]
        if not act: break
        g0 = time.time()
        prompts = [chat(tok, sw_render(eps[i]["goal"], eps[i]["steps"], obs[i], history)) for i in act]
        old = tok.padding_side; tok.padding_side = "left"
        try:
            x = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        finally:
            tok.padding_side = old
        with torch.no_grad():   # same generate kwargs as collect (generation_config top_k / repetition_penalty inherited identically)
            g = model.generate(**x, max_new_tokens=24, do_sample=temperature > 0, temperature=max(temperature, 1e-5), top_p=1.0,
                               pad_token_id=tok.pad_token_id, use_cache=True, stopping_criteria=_newline_stop(tok, x.input_ids.shape[1]))
        raws = {i: tok.decode(row, skip_special_tokens=True).strip().split("\n")[0] for i, row in zip(act, g[:, x.input_ids.shape[1]:].tolist())}
        g1 = time.time(); gen_s += g1 - g0
        rep = pool.request({i: ("act", raws[i]) for i in act}); env_s += time.time() - g1
        for i in act:
            e = eps[i]
            if rep[i][0] != "ok": e.update(done=True, error=f"step {k}: {rep[i][1]}"); continue
            _, a_exec, mapped, nxt, score, dn = rep[i]
            e["steps"].append({"obs": obs[i], "action": a_exec, "raw": raws[i], "mapped": mapped, "next_obs": nxt, "score": score, "done": dn})
            obs[i] = nxt; e["final_score"] = score
            if dn: e["done"] = True
    for e in eps:
        e["won"] = e["final_score"] >= 100; e["n_steps"] = len(e["steps"]); e["budget"] = budget; e["temperature"] = temperature; e.pop("done", None)
    return eps, {"sec_policy": gen_s, "sec_env": env_s}


def render_for_detection(e, sys_prompt, first_user, max_chars=12000):
    """ScienceWorld version of render_episode_for_detection (appworld_feedback.py:10-61): step reward = the env's score increase / 100,
    final reward = max(final score, 0) / 100 (ScienceWorld's own 0-100 score), terminated = env done flag."""
    initial = f"SYSTEM:\n{sys_prompt.strip()}\n\nUSER:\n{first_user.strip()}"
    blocks = []; prev = 0
    for k, s in enumerate(e["steps"]):
        last = k == len(e["steps"]) - 1; sc = s["score"] if isinstance(s["score"], (int, float)) else prev
        blocks.append("\n".join([f"Step {k + 1}", f"Assistant action:\n{s['action']}", f"Environment output:\n{str(s['next_obs']).strip()}",
                                 f"Step reward: {(sc - prev) / 100.0}", f"Terminated: {bool(s.get('done', False))}"]))
        prev = sc
    traj = "\n\n".join(blocks)
    if max_chars > 0 and len(traj) > max_chars: traj = "[earlier trajectory omitted]\n" + traj[-max_chars:]
    parts = [f"Task id: {e['task']}_var{e['variation']}", "Dataset: scienceworld", f"Final reward: {max(e['final_score'], 0) / 100.0}",
             f"Success: {e['won']}", f"Terminated: {bool(e['steps'] and e['steps'][-1].get('done', False))}", f"Initial task prompt:\n{initial}", f"Attempt trajectory:\n{traj}"]
    return "\n\n".join(parts)


def chunks(total, per_step, ckpt_at):
    """[(lo, hi)] episode ranges of the optimizer steps: per_step episodes each, cut exactly at every checkpoint threshold."""
    out, lo, cuts = [], 0, sorted(set([c for c in ckpt_at if 0 < c < total] + [total]))
    for c in cuts:
        while lo < c:
            hi = min(lo + per_step, c); out.append((lo, hi)); lo = hi
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="reports/sciworld/student10/final", help="start = R0 (kept training)")
    ap.add_argument("--per_task", type=int, default=75); ap.add_argument("--var_seed", type=int, default=1, help="= collect --seed of our rounds")
    ap.add_argument("--total_eps", type=int, default=4449); ap.add_argument("--ckpt_at", default="1483,2966", help="ckpt1, ckpt2; ckpt3 = --total_eps")
    ap.add_argument("--B", type=int, default=2); ap.add_argument("--rollout_n", type=int, default=4); ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--history", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-6); ap.add_argument("--warmup_frac", type=float, default=0.05); ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=0.0); ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--topk", type=int, default=100); ap.add_argument("--no_tail", action="store_true")
    ap.add_argument("--fb_max_steps", type=int, default=3); ap.add_argument("--fb_temperature", type=float, default=0.6); ap.add_argument("--fb_top_p", type=float, default=0.95)
    ap.add_argument("--fb_max_tokens", type=int, default=512); ap.add_argument("--fb_max_chars", type=int, default=12000)
    ap.add_argument("--save_every", type=int, default=5); ap.add_argument("--max_steps", type=int, default=0, help="stop after this many optimizer steps (smoke test)")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--gen_bs", type=int, default=16)
    ap.add_argument("--env_recycle", type=int, default=100); ap.add_argument("--env_timeout", type=int, default=300)
    ap.add_argument("--reserve_gb", type=float, default=8.0, help="pre-reserve this much GPU memory after loading (shared GPUs)")
    ap.add_argument("--jvm_xmx", default="1g", help="max heap of each ScienceWorld worker JVM (host-RAM OOM fix, Oct 11)")
    ap.add_argument("--smoke_budget", type=int, default=0, help="smoke test only: override the episode budget")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    from peft import PeftModel
    from sciworld_agent import SYSTEM, chat

    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    adapter = a.adapter if Path(a.adapter).is_absolute() else str(ROOT / a.adapter)
    budget = a.smoke_budget or a.budget
    # model first (shared GPUs: shortest window between the free-memory check and the allocation), then the env workers
    tok = AutoTokenizer.from_pretrained(a.model); tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    base.gradient_checkpointing_enable(); base.enable_input_require_grads(); base.config.use_cache = False
    model = PeftModel.from_pretrained(base, adapter, adapter_name="student", is_trainable=True)
    model.load_adapter(adapter, adapter_name="teacher", is_trainable=False)
    model.set_adapter("student")
    pairs = H.student_teacher_pairs(model); params = [s for s, _ in pairs]
    for s, t in pairs: s.requires_grad_(True); t.requires_grad_(False)
    print(f"student params {sum(p.numel() for p in params) / 1e6:.1f}M ({params[0].dtype}), teacher {pairs[0][1].dtype}; "
          f"gen config top_k={model.generation_config.top_k} rep={model.generation_config.repetition_penalty}", flush=True)
    # shared, oversubscribed GPUs: create the cuBLAS handle now and pre-reserve working memory in the caching allocator (freed tensor stays
    # reserved for this process), so later activation / generation peaks do not lose a race with other jobs' processes
    _w = torch.ones(256, 256, device='cuda', dtype=torch.bfloat16); (_w @ _w).sum().item(); del _w
    if a.reserve_gb > 0: _r = torch.empty(int(a.reserve_gb * 2**30), dtype=torch.uint8, device='cuda'); del _r
    print(f'reserved {torch.cuda.memory_reserved() / 2**30:.1f} GB', flush=True)
    # Oct 11 21:45 fix: host-RAM OOM kills (job cgroups are 64G, shared). The ScienceWorld JVMs default to a heap of 1/4 of host RAM and
    # grew to 3-4 GB RSS each (8 workers ~ 30 GB). Cap every worker JVM (inherited env); tested: 50-step episodes run at ~0.5 GB RSS.
    os.environ["JAVA_TOOL_OPTIONS"] = f"-Xmx{a.jvm_xmx} -XX:+UseSerialGC -XX:MaxMetaspaceSize=256m -XX:ReservedCodeCacheSize=64m"
    print(f"JAVA_TOOL_OPTIONS={os.environ['JAVA_TOOL_OPTIONS']}", flush=True)
    pool = EnvPool(a.B * a.rollout_n, budget, a.env_recycle, a.env_timeout)
    try:
        vf = out / "variations.json"
        if vf.exists(): allv = json.loads(vf.read_text())
        else:
            r = pool.request({0: ("variations", a.per_task, a.var_seed)}); assert r[0][0] == "ok", r[0]
            allv = [list(x) for x in r[0][1]]; vf.write_text(json.dumps(allv))
        tasks = list(allv); random.Random(a.seed).shuffle(tasks)
        n_tasks = math.ceil(a.total_eps / a.rollout_n); tasks = tasks[:n_tasks]
        flat = [(ti, tasks[ti][0], tasks[ti][1], k) for ti in range(n_tasks) for k in range(a.rollout_n)][: a.total_eps]
        ckpt_at = [int(x) for x in a.ckpt_at.split(",") if x]; ranges = chunks(len(flat), a.B * a.rollout_n, ckpt_at); total_steps = len(ranges)
        print(f"HINT-SD online ScienceWorld: {len(allv)} train variations (per_task {a.per_task}, seed {a.var_seed}) -> {n_tasks} used x {a.rollout_n} rollouts "
              f"= {len(flat)} episodes, {a.B * a.rollout_n} episodes/step -> {total_steps} optimizer steps; ckpts at {ckpt_at} + {len(flat)}; out {out}", flush=True)

        opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=a.weight_decay)
        sched = get_linear_schedule_with_warmup(opt, math.ceil(a.warmup_frac * total_steps), total_steps)

        meta = {"step": 0, "cum_episodes": 0, "ckpts": [], "sec_total": 0.0, "gen_tokens": 0, "targets": 0, "target_tokens": 0, "episodes_won": 0,
                "failed": 0, "analyzed": 0, "env_errors": 0, "nonfinite_rows": 0, "empty_steps": 0, "args": vars(a)}
        st = H.find_state(out)
        if st is not None:
            meta = json.loads((st / "meta.json").read_text())
            H.load_adapter_into(model, "student", st / "student"); H.load_adapter_into(model, "teacher", st / "teacher")
            sd = torch.load(st / "optim.pt", map_location="cpu", weights_only=False); opt.load_state_dict(sd["opt"]); sched.load_state_dict(sd["sched"])
            model.set_adapter("student")
            for s, t in pairs: s.requires_grad_(True); t.requires_grad_(False)
            for f in ("train_log.jsonl", "rollouts.jsonl", "analysis.jsonl", "targets.jsonl"): H.truncate_jsonl(out / f, meta["step"])
            cs = H.checksum(pairs)
            print(f"RESUMED from {st} at step {meta['step']} ({meta['cum_episodes']} episodes): |student| {cs[0]} |teacher| {cs[1]} max|s-t| {cs[2]:.3e}", flush=True)
        if meta["step"] >= total_steps and (out / f"ckpt{len(ckpt_at) + 1}" / "adapter" / "adapter_config.json").exists():
            print("DONE hintsd_online (already complete)", flush=True); return

        flog, frol, fana, ftgt = (open(out / f, "a") for f in ("train_log.jsonl", "rollouts.jsonl", "analysis.jsonl", "targets.jsonl"))
        steps_this_run = 0
        while meta["step"] < total_steps:
            step = meta["step"] + 1; lo, hi = ranges[step - 1]; assert lo == meta["cum_episodes"], (lo, meta["cum_episodes"])
            jobs = flat[lo:hi]
            random.seed(a.seed * 1000003 + step); torch.manual_seed(a.seed * 1000003 + step); torch.cuda.manual_seed_all(a.seed * 1000003 + step)
            t0 = time.time()
            # ---------------- 1. rollouts with the current student
            model.eval(); model.set_adapter("student")
            with torch.no_grad():
                eps, rst = rollout(pool, model, tok, jobs, budget, a.temperature, step, a.history)
            t1 = time.time()
            for e in eps: H.jdump(frol, {"step": step, **e})
            # ---------------- 2. hindsight analysis of the failed rollouts by the EMA teacher
            failed = [i for i, e in enumerate(eps) if not e["won"] and e["steps"] and "error" not in e]
            prompts = [tok.apply_chat_template([{"role": "system", "content": ANALYZER_SYSTEM.format(k=a.fb_max_steps)},
                                                {"role": "user", "content": ANALYZER_USER.format(traj=render_for_detection(
                                                    eps[i], SYSTEM, sw_render(eps[i]["goal"], [], eps[i]["steps"][0]["obs"], a.history), a.fb_max_chars))}],
                                               tokenize=False, add_generation_prompt=True) for i in failed]
            model.set_adapter("teacher")
            with torch.no_grad():
                raws, n_gen = H.generate_batch(model, tok, prompts, a.fb_max_tokens, a.fb_temperature, a.fb_top_p, a.gen_bs) if prompts else ([], 0)
            model.set_adapter("student")
            for s, t in pairs: s.requires_grad_(True); t.requires_grad_(False)
            detections = {}
            for i, raw in zip(failed, raws):
                det = H.parse_multi_step_failure_json(raw, len(eps[i]["steps"]), a.fb_max_steps); detections[i] = det
                H.jdump(fana, {"step": step, "id": eps[i]["id"], "n_steps": len(eps[i]["steps"]), "parsed": det is not None,
                               "targets": [{"t": t, "feedback": fb} for t, fb in (det or [])], "raw": raw})
            t2 = time.time()
            # ---------------- 3. targets: student prompt at t, the student's sampled (raw) action tokens, teacher prompt = + feedback turn
            rows_by_task = {}
            for i, det in detections.items():
                e = eps[i]
                for t, fb in det or []:
                    u = sw_render(e["goal"], e["steps"][:t], e["steps"][t]["obs"], a.history); sp = chat(tok, u)
                    tp = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": u},
                                                  {"role": "user", "content": H.FB_TEMPLATE.format(hint=fb)}], tokenize=False, add_generation_prompt=True)
                    action = e["steps"][t]["raw"]; comp = tok(action, add_special_tokens=False).input_ids
                    if comp: rows_by_task.setdefault(e["task_idx"], []).append({"id": e["id"], "t": t, "feedback": fb, "action": action, "sp": sp, "tp": tp, "comp": comp})
            # ---------------- 4. loss: per task token-mean reverse KL, / #tasks in the step (grad accumulation), as ALFWorld
            model.train(); n_rows = n_tok = 0; loss_sum = 0.0; kl_full_w = 0.0; tgt_logs = []
            opt.zero_grad(set_to_none=True)
            n_acc = len({j[0] for j in jobs})
            for ti, rows in rows_by_task.items():
                valid = sum(len(r["comp"]) for r in rows)
                for r in rows:
                    sp_ids = tok(r["sp"], add_special_tokens=False).input_ids; tp_ids = tok(r["tp"], add_special_tokens=False).input_ids
                    model.set_adapter("teacher")
                    with torch.no_grad(): t_lp = H.completion_log_probs(model, tp_ids, r["comp"])
                    model.set_adapter("student")
                    for s, t in pairs: s.requires_grad_(True); t.requires_grad_(False)
                    s_lp = H.completion_log_probs(model, sp_ids, r["comp"])
                    mask = torch.ones(s_lp.shape[:2], device=s_lp.device)
                    row_loss = H.compute_distillation_loss(s_lp, t_lp, mask, a.topk, not a.no_tail)
                    w = len(r["comp"]) / valid
                    if not torch.isfinite(row_loss):
                        meta["nonfinite_rows"] += 1; print(f"SKIP non-finite row {r['id']} t={r['t']}", flush=True); continue
                    (row_loss * w / n_acc).backward()
                    with torch.no_grad():
                        kl_full = (s_lp.exp() * (s_lp - t_lp)).sum(-1).mean().item()
                        first_ok = int(t_lp[0, 0].argmax().item() == r["comp"][0])
                        cidx = torch.tensor(r["comp"], device=s_lp.device)[:, None]
                    loss_sum += row_loss.item() * len(r["comp"]); kl_full_w += kl_full * len(r["comp"]); n_rows += 1; n_tok += len(r["comp"])
                    tgt_logs.append({"step": step, "id": r["id"], "t": r["t"], "feedback": r["feedback"], "action": r["action"], "n_tok": len(r["comp"]),
                                     "loss": round(row_loss.item(), 5), "kl_full": round(kl_full, 5), "teacher_top1_is_action_tok0": first_ok,
                                     "teacher_lp_action": round(t_lp[0].gather(1, cidx).sum().item(), 4), "student_lp_action": round(s_lp[0].gather(1, cidx).sum().item(), 4)})
                    del s_lp, t_lp, row_loss
            gnorm = float("nan"); lr_used = sched.get_last_lr()[0]
            if n_rows:
                gnorm = float(torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm)); opt.step()
            else:
                meta["empty_steps"] += 1
            sched.step(); opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                for s, t in pairs: t.lerp_(s.detach().to(t.dtype), 1.0 - a.ema_decay)
            t3 = time.time()
            for x in tgt_logs: H.jdump(ftgt, x)
            # ---------------- 5. bookkeeping, checkpoints, state
            n_won = sum(e["won"] for e in eps); n_err = sum("error" in e for e in eps); n_parsed = sum(d is not None for d in detections.values())
            meta.update(step=step, cum_episodes=hi, sec_total=meta["sec_total"] + (t3 - t0))
            for k, v in (("gen_tokens", n_gen), ("targets", n_rows), ("target_tokens", n_tok), ("episodes_won", n_won), ("failed", len(failed)),
                         ("analyzed", n_parsed), ("env_errors", n_err)):
                meta[k] = meta.get(k, 0) + v
            rec = {"step": step, "of": total_steps, "cum_episodes": hi, "episodes": len(eps), "success_rate": round(n_won / len(eps), 4),
                   "mean_score": round(sum(max(e["final_score"], 0) for e in eps) / len(eps), 1), "mean_steps": round(sum(e["n_steps"] for e in eps) / len(eps), 1),
                   "n_failed": len(failed), "n_analyzed": n_parsed, "n_targets": n_rows, "n_target_tokens": n_tok,
                   "loss": round(loss_sum / n_tok, 6) if n_tok else None, "kl_full": round(kl_full_w / n_tok, 6) if n_tok else None,
                   "lr": lr_used, "grad_norm": round(gnorm, 5) if n_rows else None, "env_errors": n_err, "gen_tokens": n_gen,
                   "sec": round(t3 - t0, 1), "sec_rollout": round(t1 - t0, 1), "sec_rollout_policy": round(rst["sec_policy"], 1), "sec_rollout_env": round(rst["sec_env"], 1),
                   "sec_analyze": round(t2 - t1, 1), "sec_loss": round(t3 - t2, 1), "env_restarts": pool.restarts,
                   "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            H.jdump(flog, rec); print(json.dumps(rec), flush=True)
            saved_ckpt = False
            for c, thr in enumerate(ckpt_at + [len(flat)], 1):
                if hi >= thr and c not in meta["ckpts"]:
                    H.save_adapter(model, tok, "student", out / f"ckpt{c}" / "adapter"); meta["ckpts"].append(c); saved_ckpt = True
                    print(f"CKPT{c} saved at step {step}, {hi} episodes", flush=True)
            steps_this_run += 1
            stop = a.max_steps and steps_this_run >= a.max_steps
            if step % a.save_every == 0 or step == total_steps or stop or saved_ckpt:
                H.save_state(out, model, tok, opt, sched, meta)
                cs = H.checksum(pairs); print(f"STATE saved at step {step}: |student| {cs[0]} |teacher| {cs[1]} max|s-t| {cs[2]:.3e}", flush=True)
            if stop:
                print(f"STOP after {steps_this_run} steps (--max_steps)", flush=True); break
    finally:
        pool.close()
    if meta["step"] >= total_steps:
        summ = {"gpu_hours": round(meta["sec_total"] / 3600, 2), "episodes": meta["cum_episodes"], "optimizer_steps": meta["step"],
                "rollout_success_rate": round(meta["episodes_won"] / max(1, meta["cum_episodes"]), 4), "failed": meta["failed"], "analyzed": meta["analyzed"],
                "analyzer_generated_tokens": meta["gen_tokens"], "backprop_target_steps": meta["targets"], "backprop_target_tokens": meta["target_tokens"],
                "nonfinite_rows": meta["nonfinite_rows"], "empty_steps": meta["empty_steps"], "env_errors": meta["env_errors"]}
        (out / "summary.json").write_text(json.dumps(summ, indent=1))
        print("SUMMARY " + json.dumps(summ), flush=True); print("DONE hintsd_online", flush=True)


if __name__ == "__main__":
    main()
