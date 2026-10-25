"""Step-localised, environment-verified Attribution-Guided OPD for the ALFWorld student (Research Plan Sec. 10 + 14).

Stages (run sequentially on one GPU; each loads only what it needs):
  choose   : pick the supervised step(s) per failed trajectory for an arm
             attr   -> Model A's t_hat (from alfworld_attribute.py output)
             oracle -> progress oracle's first harmful action (t_first_regress; computed here if missing)
             random -> uniform random step
             last   -> last step
             uniform-> every step (standard OPD; capped by --max_teacher_calls)
  teacher  : privileged teacher (hindsight hint) scores the admissible commands at each chosen state -> q_T
  distill  : (optional) verify the teacher's top-k alternatives by replaying the prefix and letting the student
             finish; target = verified successes (else teacher q_T if --fallback_unverified); then minimise
             KL(target || q_S) over the admissible set at the chosen step(s). Saves a LoRA adapter.
Rollouts / eval reuse alfworld_collect.py (with --adapter).
"""
import argparse, json, math, os, random, signal, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("ERROR_ATTR_ROOT", str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from error_attr.policy.alfworld_agent import ALFWorldHistory, InteractionStep, LLMPolicy, SYSTEM_PROMPT

MODE_NAMES = {"R.2": "a reasoning error", "R.4": "a misunderstanding of the task", "A.3": "premature termination / giving up on the goal",
              "A.1": "a wrong or incomplete action sequence", "R.1": "acting on something not actually observed", "A.2": "a malformed action"}


def load_jsonl(p): return [json.loads(l) for l in open(p)] if p and Path(p).exists() else []


def history_at(ep, t, history=10):
    """The policy's history object at step t (before acting), rebuilt from the logged trajectory."""
    h = ALFWorldHistory(goal=ep["goal"])
    for i, st in enumerate(ep["steps"][: t + 1]):
        h.steps.append(InteractionStep(observation=st["obs"], admissible_commands=list(st["admissible"]), action=st["action"] if i < t else None))
    return h


# ------------------------------------------------------------------ choose
def cmd_choose(a):
    eps = {e["id"]: e for e in load_jsonl(a.traj) if not e["won"]}
    rng = random.Random(a.seed)
    attr = {r["id"]: r for r in load_jsonl(a.attr)} if a.attr else {}
    prog = {r["id"]: r for r in load_jsonl(a.progress)} if a.progress else {}
    tstar = {r["id"]: r for r in load_jsonl(a.tstar)} if a.tstar else {}
    out = []; n_calls = 0
    ids = list(eps)
    if a.ids_from:  # restrict every arm to the same failure set (e.g. the oracle-labelled ones)
        keep = {r["id"] for r in load_jsonl(a.ids_from) if "error" not in r}; ids = [i for i in ids if i in keep]
    ids.sort(); rng.shuffle(ids)
    for i in ids:
        e = eps[i]; n = len(e["steps"])
        if a.arm == "attr":
            r = attr.get(i); t = r.get("t_hat") if r else None
            if t is None: continue
            steps = [min(max(0, t), n - 1)]; mode = (r.get("pred") or {}).get("error_mode")
        elif a.arm == "oracle":
            r = prog.get(i)
            if r is None:
                if not a.compute_missing: continue
                from error_attr.attribution.oracle_alfworld import progress_trace
                r = progress_trace(e["game_file"], [s["action"] for s in e["steps"]], e["budget"]); prog[i] = r
            t = r.get("t_first_regress")
            if t is None: t = r.get("t_first_waste")
            if t is None: continue
            steps = [min(t, n - 1)]; mode = None
        elif a.arm == "tstar":   # point-of-no-return oracle: the last action after which the game is still winnable within budget, +1
            r = tstar.get(i); t = r.get("t_star") if r else None
            if t is None: continue
            steps = [min(t, n - 1)]; mode = None
        elif a.arm == "random":
            steps = [rng.randrange(n)]; mode = None
        elif a.arm == "random_matched":   # depth-matched random control: resample from a real pointer's own empirical step distribution (not uniform over the episode), so the comparison isolates "which position" from "how early/late positions typically are"
            if not hasattr(cmd_choose, "_match_pool"):
                pool = [r["t_first_regress"] for r in prog.values() if r.get("t_first_regress") is not None]
                if not pool: pool = [r.get("t_hat") for r in attr.values() if r.get("t_hat") is not None]
                cmd_choose._match_pool = pool
            t = rng.choice(cmd_choose._match_pool)
            steps = [min(max(0, t), n - 1)]; mode = None
        elif a.arm == "last":
            steps = [n - 1]; mode = None
        elif a.arm == "const":   # control: the same fixed step for every failure (no attribution at all; e.g. step 1 = the prior "early error")
            steps = [min(a.const_step, n - 1)]; mode = None
        elif a.arm == "uniform":
            steps = list(range(n)); mode = None
        else:
            raise ValueError(a.arm)
        if a.offset and a.arm in ("attr", "oracle", "tstar", "random", "const"):  # control: deliberately mis-locate the pointer by a fixed offset
            steps = [min(max(0, steps[0] + a.offset), n - 1)]
        if a.window and a.arm in ("attr", "oracle", "random"):  # supervise a +-window of steps around the chosen one
            c = steps[0]; steps = [t for t in range(c - a.window, c + a.window + 1) if 0 <= t < n]
        if a.max_teacher_calls and n_calls + len(steps) > a.max_teacher_calls: break
        n_calls += len(steps)
        out.append({"id": i, "steps": steps, "mode": mode})
        if a.max_failures and len(out) >= a.max_failures: break
    with open(a.out, "w") as f:
        for r in out: f.write(json.dumps(r) + "\n")
    print(f"arm {a.arm}: {len(out)} failures, {n_calls} supervised states -> {a.out}", flush=True)


# ------------------------------------------------------------------ teacher
def privileged_history(ep, t, mode, hint=True):
    h = history_at(ep, t)
    if hint:
        a_minus = ep["steps"][t]["action"]
        m = MODE_NAMES.get(mode, "a mistake") if mode else "a mistake"
        h.goal = (h.goal + f"\n\nHindsight note: a previous attempt reached exactly this state, chose '{a_minus}', and then failed the task; "
                  f"that choice was diagnosed as {m}. Choose a better admissible action that makes progress on the goal.")
    return h


def cmd_teacher(a):
    eps = {e["id"]: e for e in load_jsonl(a.traj)}
    chosen = load_jsonl(a.steps)
    done = {(r["id"], r["t"]) for r in load_jsonl(a.out)}
    tok = AutoTokenizer.from_pretrained(a.teacher); tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.teacher, dtype=torch.bfloat16, device_map={"": 0}).eval()
    if a.teacher_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.teacher_adapter).merge_and_unload().eval()
    pol = LLMPolicy(model, tok)
    t0 = time.time(); n = 0
    with open(a.out, "a") as f:
        for r in chosen:
            e = eps[r["id"]]
            for t in r["steps"]:
                if (r["id"], t) in done: continue
                h = privileged_history(e, t, r.get("mode"), hint=not a.no_hint)
                lp = pol.score_admissible_commands_cached(h)
                f.write(json.dumps({"id": r["id"], "t": t, "a_minus": e["steps"][t]["action"], "admissible": list(lp), "teacher_logprobs": lp, "mode": r.get("mode")}) + "\n"); f.flush()
                n += 1
                if n % 50 == 0: print(f"[{n}] {(time.time() - t0) / n:.2f}s/state", flush=True)
    print(f"DONE teacher: {n} states", flush=True)


# ------------------------------------------------------------------ distill
def student_logprobs_grad(policy, tok, h, commands, batch_size=8):
    """Differentiable log p_S(cmd | h) for each admissible command (full prompt+cmd forward per batch)."""
    prompt = policy._build_chat_prompt(h)
    p_ids = tok(prompt, add_special_tokens=False).input_ids; plen = len(p_ids)
    pad = tok.pad_token_id or tok.eos_token_id; dev = policy.device
    outs = []
    for s in range(0, len(commands), batch_size):
        chunk = commands[s: s + batch_size]; ids = [tok(c, add_special_tokens=False).input_ids for c in chunk]
        m = max(len(x) for x in ids); L = plen + m
        inp = torch.full((len(chunk), L), pad, dtype=torch.long); att = torch.zeros((len(chunk), L), dtype=torch.long)
        for i, x in enumerate(ids):
            seq = p_ids + x; inp[i, : len(seq)] = torch.tensor(seq); att[i, : len(seq)] = 1
        logits = policy.model(input_ids=inp.to(dev), attention_mask=att.to(dev)).logits[:, plen - 1: -1].float()
        lsm = torch.log_softmax(logits, -1)
        for i, x in enumerate(ids):
            tg = torch.tensor(x, device=dev)
            outs.append(lsm[i, : len(x)].gather(1, tg.unsqueeze(1)).sum())
    return torch.stack(outs)


def verify_alternatives(ep, t, alts, policy, budget, continue_temperature=0.0, max_continue_steps=0):
    """Replay prefix actions[:t], take each alternative, let the student finish. Returns {alt: won}, env steps used."""
    from error_attr.env.alfworld_env import make_env
    res = {}; steps_used = 0
    def _alarm(signum, frame): raise TimeoutError("verify timeout")
    signal.signal(signal.SIGALRM, _alarm)
    env = make_env(ep["game_file"], budget + 1)
    for alt in alts:
        signal.alarm(300)
        try:
            obs, info = env.reset(); done = False
            for act in [s["action"] for s in ep["steps"][:t]]:
                obs, rw, dn, info = env.step([act]); steps_used += 1
                if dn[0]: done = True; break
            if done: res[alt] = bool(info["won"][0]); continue
            obs, rw, dn, info = env.step([alt]); steps_used += 1
            if dn[0]: res[alt] = bool(info["won"][0]); continue
            h = history_at(ep, t); h.steps[-1].action = alt
            h.steps.append(InteractionStep(observation=obs[0], admissible_commands=list(info["admissible_commands"][0])))
            k = t + 1; won = False; limit = min(budget, t + 1 + max_continue_steps) if max_continue_steps else budget
            while k < limit:
                act = policy.act(h, temperature=continue_temperature); h.steps[-1].action = act
                obs, rw, dn, info = env.step([act]); steps_used += 1; k += 1
                if dn[0]: won = bool(info["won"][0]); break
                h.steps.append(InteractionStep(observation=obs[0], admissible_commands=list(info["admissible_commands"][0])))
            res[alt] = won
        except TimeoutError:
            res[alt] = False
        finally:
            signal.alarm(0)
    try: env.close()
    except Exception: pass
    return res, steps_used


def cmd_distill(a):
    from peft import PeftModel, LoraConfig, get_peft_model
    eps = {e["id"]: e for e in load_jsonl(a.traj)}
    targets = load_jsonl(a.targets)
    rng = random.Random(a.seed); rng.shuffle(targets)
    out_dir = Path(a.out)
    done = {(r["id"], r["t"]) for r in load_jsonl(out_dir / "examples.jsonl")} | {(r["id"], r["t"]) for r in load_jsonl(out_dir / "skipped.jsonl")}
    todo = [t for t in targets if (t["id"], t["t"]) not in done]
    if a.max_examples_per_run: todo = todo[: a.max_examples_per_run]
    print(f"targets {len(targets)}, done {len(done)}, this run {len(todo)}", flush=True)
    if not todo:
        print("DONE distill (nothing left)", flush=True); return
    if (out_dir / "adapter" / "adapter_config.json").exists():
        a.adapter = str(out_dir / "adapter"); print(f"continuing from {a.adapter}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.student); tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(a.student, dtype=torch.bfloat16, device_map={"": 0})
    base.gradient_checkpointing_enable(); base.enable_input_require_grads()
    if a.adapter:
        model = PeftModel.from_pretrained(base, a.adapter, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    pol = LLMPolicy(model, tok)
    params = [p for p in model.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    out_dir.mkdir(parents=True, exist_ok=True); log = open(out_dir / "distill_log.jsonl", "a"); exlog = open(out_dir / "examples.jsonl", "a"); sklog = open(out_dir / "skipped.jsonl", "a")
    stats = {"examples": 0, "verified_used": 0, "unverified_used": 0, "skipped_no_success": 0, "skipped_recoverable": 0, "env_steps": 0, "teacher_calls": len(targets)}
    t0 = time.time(); model.train(); acc_loss = 0.0; n_in_batch = 0
    for k, tg in enumerate(todo):
        e = eps[tg["id"]]; t = tg["t"]; cmds = tg["admissible"]
        tl = torch.tensor([tg["teacher_logprobs"][c] for c in cmds]); qT = torch.softmax(tl / a.teacher_temp, 0)
        target = None; kind = None
        if a.verify_k > 0:
            order = sorted(cmds, key=lambda c: -tg["teacher_logprobs"][c]); alts = [c for c in order if c != tg["a_minus"]][: a.verify_k]
            to_check = alts + ([tg["a_minus"]] if a.contrastive and tg["a_minus"] in cmds and tg["a_minus"] not in alts else [])
            model.eval()
            with torch.no_grad():
                res, used = verify_alternatives(e, t, to_check, pol, e["budget"], a.continue_temperature, a.max_continue_steps)
            model.train(); stats["env_steps"] += used
            good = [c for c in alts if res.get(c)]
            if a.contrastive and res.get(tg["a_minus"]):   # the student's own action also wins under the same continuation: no correction to teach
                stats["skipped_recoverable"] += 1; sklog.write(json.dumps({"id": tg["id"], "t": t, "reason": "a_minus_recovers"}) + "\n"); sklog.flush(); continue
            if good:
                target = torch.zeros(len(cmds)); w = torch.tensor([math.exp(tg["teacher_logprobs"][c]) for c in good]); w = w / w.sum()
                for c, wi in zip(good, w): target[cmds.index(c)] = wi
                kind = "verified"
            elif a.fallback_unverified:
                target = qT.clone(); kind = "unverified"
            else:
                stats["skipped_no_success"] += 1; sklog.write(json.dumps({"id": tg["id"], "t": t}) + "\n"); sklog.flush(); continue
        else:
            target = qT.clone(); kind = "unverified"
        if a.exclude_a_minus and tg["a_minus"] in cmds:
            target[cmds.index(tg["a_minus"])] = 0.0; target = target / target.sum().clamp_min(1e-8)
        exlog.write(json.dumps({"id": tg["id"], "t": t, "kind": kind, "a_minus": tg["a_minus"], "n_admissible": len(cmds),
                                "alternatives": (res if a.verify_k > 0 else None), "target_top": cmds[int(target.argmax())]}) + "\n"); exlog.flush()
        h = history_at(e, t)
        lpS = student_logprobs_grad(pol, tok, h, cmds); logqS = torch.log_softmax(lpS, 0)
        loss = -(target.to(logqS.device) * logqS).sum() + (target.to(logqS.device) * torch.log(target.to(logqS.device).clamp_min(1e-8))).sum()  # KL(target || qS)
        (loss / a.batch).backward(); acc_loss += loss.item(); n_in_batch += 1
        stats["examples"] += 1; stats[f"{kind}_used"] += 1
        if n_in_batch == a.batch:
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
            rec = {"k": k + 1, "loss": acc_loss / n_in_batch, "elapsed_s": round(time.time() - t0), **stats}; print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            acc_loss = 0.0; n_in_batch = 0
    if n_in_batch: torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
    model.save_pretrained(str(out_dir / "adapter")); tok.save_pretrained(str(out_dir / "adapter"))
    stats["elapsed_s"] = round(time.time() - t0); stats["remaining"] = len(targets) - len(done) - len(todo)
    log.write(json.dumps({"final": stats}) + "\n"); log.close(); exlog.close(); sklog.close()
    print(("DONE distill" if stats["remaining"] <= 0 else "CHUNK distill"), json.dumps(stats), flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("choose"); c.add_argument("--traj", required=True); c.add_argument("--arm", required=True, choices=["attr", "oracle", "tstar", "random", "random_matched", "last", "uniform", "const"]); c.add_argument("--const_step", type=int, default=1); c.add_argument("--offset", type=int, default=0, help="shift the chosen step by this many actions (imprecision control)")
    c.add_argument("--attr", default=None); c.add_argument("--progress", default=None); c.add_argument("--tstar", default=None); c.add_argument("--out", required=True)
    c.add_argument("--max_failures", type=int, default=0); c.add_argument("--max_teacher_calls", type=int, default=0); c.add_argument("--seed", type=int, default=0)
    c.add_argument("--ids_from", default=None, help="restrict to failure ids present in this jsonl (same set for every arm)"); c.add_argument("--compute_missing", action="store_true")
    c.add_argument("--window", type=int, default=0, help="also supervise the steps within +-window of the chosen step (attr/oracle/random)")
    t = sub.add_parser("teacher"); t.add_argument("--traj", required=True); t.add_argument("--steps", required=True); t.add_argument("--teacher", default="Qwen/Qwen2.5-14B-Instruct")
    t.add_argument("--out", required=True); t.add_argument("--no_hint", action="store_true"); t.add_argument("--teacher_adapter", default=None)
    d = sub.add_parser("distill"); d.add_argument("--traj", required=True); d.add_argument("--targets", required=True); d.add_argument("--student", default="Qwen/Qwen2.5-3B-Instruct")
    d.add_argument("--adapter", default=None); d.add_argument("--out", required=True); d.add_argument("--lr", type=float, default=1e-5); d.add_argument("--batch", type=int, default=8)
    d.add_argument("--verify_k", type=int, default=3); d.add_argument("--continue_temperature", type=float, default=0.0); d.add_argument("--fallback_unverified", action="store_true")
    d.add_argument("--exclude_a_minus", action="store_true"); d.add_argument("--teacher_temp", type=float, default=1.0); d.add_argument("--seed", type=int, default=0)
    d.add_argument("--contrastive", action="store_true", help="also continue after the student's original action; skip the state if it wins too (only causal corrections become targets)")
    d.add_argument("--max_continue_steps", type=int, default=20, help="cap on the student's continuation after an alternative (0 = to budget)")
    d.add_argument("--max_examples_per_run", type=int, default=0, help="process at most this many targets, save the adapter, exit (host-RAM grows per env; run in chunks and resume)")
    a = ap.parse_args()
    {"choose": cmd_choose, "teacher": cmd_teacher, "distill": cmd_distill}[a.cmd](a)


if __name__ == "__main__":
    main()
