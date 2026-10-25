"""TextCraft PLAN-THEN-ACT pipeline (user-approved Oct 9).  Root cause targeted: the student copies recipe counts instead of multiplying them
by the number of crafts it needs, picks wrong recipe paths and loses track of its inventory, because it never reasons -- it only scores
admissible commands.
  STUDENT  at t=0 it GENERATES a plan (total quantity of every base ingredient to get + the crafts in order); the plan stays in the prompt at
           every later step; actions are still chosen by scoring the admissible commands, conditioned on task + plan + history.
           Plan format (rendered from the oracle solver for the SFT demos):
             Need: 4 bamboo, 1 acacia logs. Then: craft 2 stick (2x 'craft 1 stick using 2 bamboo'), craft 4 acacia planks (1x 'craft 4 acacia
             planks using 1 acacia logs'), craft 1 wooden shovel (1x 'craft 1 wooden shovel using 2 stick, 1 acacia planks').
  demos    the same 100 TRAIN demos as student100 (first 100 of gold_train.jsonl, seed 0) + their oracle plan text
  sft      token-level SFT of the plan (t=0) + the actions (same admissible-scoring SFT as student100), same hyper-parameters as student100
  collect  plan (greedy unless --plan_temperature) then act (greedy or T); writes the episode with its plan (+ the oracle plan, for diagnostics
           only -- never shown to any model) and, for failures, a Who&When release row (see release)
  release  failed episodes -> Model A rows: query = the COMPLETE task statement (all crafting commands + goal) + HINT; the plan is the
           reasoning of step 1 (the agent's thought before its first action); integer step_number, is_final_answer False, no score
  variant  Model A agreement with the oracle reference first_waste (first action that does not shorten the oracle's remaining plan) for the
           no_observations and full variants, on TRAINING failures -> picks the better one
  teacher  full-view hindsight teacher (whole failed trajectory, plan, Model A diagnosis) that first ANALYSES (total requirements, what was
           missing / wrong, best action now; 256 tokens) and then scores the admissible commands
  replay_plan  adds the plan-generation target (t = -1 rows) for every replay row at t = 0
  train    alfworld_opd_plus.cmd_train unchanged, with the plan student's prompts (t = -1 rows = plan generation SFT)
"""
import argparse, collections, glob, json, math, os, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("ERROR_ATTR_ROOT", str(ROOT)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from textcraft_env import TextCraftEnv, splits, budget_for, tree, item_id_to_str

PLAN_SYSTEM = ("You are a crafting agent in a text-based Minecraft environment. You will be given the available crafting commands and a goal "
               "item. When asked for a plan, write the plan. Otherwise you will be given your plan and the history of your actions and their "
               "results: choose the single best next action from the list of admissible actions and respond with the exact text of one "
               "admissible action and nothing else.")
PLAN_Q = ("Before acting, write a plan: the TOTAL quantity of every base ingredient you must get (recipe counts multiplied by the number of "
          "crafts needed), then the crafts in order.")
HINT = ("Hint: in crafting tasks the decisive error is usually the earliest action that made success unreachable -- often a 'get' that fetched "
        "fewer items than the recipe chain requires (recipe counts must be multiplied by the number of crafts needed), a skipped ingredient, or a "
        "wrong recipe path (possibly already in the agent's plan, shown in step 1). A later failed craft ('Could not find enough items') only "
        "reveals the shortage and is usually not the decisive error.")
ANALYSIS_Q_PLAN = ("Before acting: this attempt FAILED. First list the TOTAL requirements of the whole task: every base ingredient with the total "
                   "quantity needed (recipe counts multiplied by the number of crafts needed) and the crafts in order. Then say what was missing or "
                   "wrong in this attempt (wrong counts, missed ingredients, wrong recipe path). Finally say what the best action at the CURRENT "
                   "step is.")
GEN_SYSTEM = "You are a crafting agent reviewing a failed attempt."


def L(p): return [json.loads(l) for l in open(p)] if Path(p).exists() else []


# ----------------------------------------------------------------------------- plan text
def plan_from_actions(actions):
    need = {}; crafts = []
    for a in actions:
        if a.startswith("get "):
            _, n, item = a.split(" ", 2); need[item] = need.get(item, 0) + int(n)
        elif a.startswith("craft "):
            if crafts and crafts[-1][0] == a: crafts[-1][1] += 1
            else: crafts.append([a, 1])
    parts = []
    for cmd, n in crafts:
        out = cmd[6:].split(" using ")[0]; c, name = out.split(" ", 1)
        parts.append(f"craft {n * int(c)} {name} ({n}x '{cmd}')")
    return "Need: " + (", ".join(f"{n} {it}" for it, n in need.items()) or "nothing") + ". Then: " + ", ".join(parts) + "."


def parse_need(plan):
    """{item: total} from a plan's 'Need:' clause (None if unparsable)."""
    try:
        s = plan.split("Need:", 1)[1].split(". Then:", 1)[0].strip()
        if s == "nothing": return {}
        out = {}
        for x in s.split(", "):
            n, it = x.strip().split(" ", 1); out[it] = int(n)
        return out
    except Exception:
        return None


# ----------------------------------------------------------------------------- prompts / histories
def render_act(obs0, plan, past, adm, history=10):
    lines = [obs0, "", f"Plan: {plan}", ""]
    if len(past) > history: lines.append(f"(... {len(past) - history} earlier steps omitted)")
    for act, obs in past[-history:]: lines += [f"Action: {act}", f"Observation: {obs}"]
    lines.append("Admissible actions:"); lines += [f"- {a}" for a in adm]; lines += ["", "Next action:"]
    return "\n".join(lines)


def render_plan_prompt(obs0): return obs0 + "\n\n" + PLAN_Q + "\nPlan:"


class _Step:
    def __init__(self, adm): self.admissible_commands = adm


class PlanHistory:
    def __init__(self, obs0, plan, past, adm, history=10):
        self.obs0, self.plan, self.past, self.adm, self.H = obs0, plan, list(past), list(adm), history; self.steps = [_Step(self.adm)]
    def render_prompt(self, max_history_steps=None): return render_act(self.obs0, self.plan, self.past, self.adm, self.H)


class PlanGenHistory:
    """prompt for the plan-generation target (trainer rows with t = -1)."""
    def __init__(self, obs0): self.obs0 = obs0; self.steps = [_Step([])]
    def render_prompt(self, max_history_steps=None): return render_plan_prompt(self.obs0)


def history_at(e, t, history=10):
    if t == -1: return PlanGenHistory(e["initial_obs"])
    return PlanHistory(e["initial_obs"], e["plan"], [(s["action"], s["next_obs"]) for s in e["steps"][:t]], e["steps"][t]["admissible"], history)


class FullHindsightPlan:
    """full-view teacher: task + the student's plan + the FULL prefix + EVERY remaining step with its full observation + the outcome."""
    def __init__(self, e, t):
        self._h = PlanHistory(e["initial_obs"], e["plan"], [(s_["action"], s_["next_obs"]) for s_ in e["steps"][:t]], e["steps"][t]["admissible"], history=10 ** 9)
        self.steps = self._h.steps
        self._block = "\n".join(f"- step {t + i}: {s_['action']} -> {s_['next_obs']}" for i, s_ in enumerate(e["steps"][t:]))
        self._n = len(e["steps"]); self._b = e["budget"]
    def render_prompt(self, max_history_steps=None):
        p = self._h.render_prompt()
        note = ("Hindsight: this attempt FAILED (the goal item was never crafted). After this point the agent did the following "
                f"(episode length {self._n} steps, step budget {self._b}, budget exhausted):\n{self._block}\n"
                "Using everything above, choose the best action for the CURRENT step.\n")
        return p.replace("Admissible actions:", note + "Admissible actions:", 1)


def policy_class():
    from error_attr.policy.alfworld_agent import LLMPolicy

    class PlanPolicy(LLMPolicy):
        def _build_chat_prompt(self, history):
            return self.tokenizer.apply_chat_template([{"role": "system", "content": PLAN_SYSTEM}, {"role": "user", "content": history.render_prompt()}],
                                                      tokenize=False, add_generation_prompt=True)
    return PlanPolicy


def load_policy(student, adapter, trainable=False, merge=False):
    import alfworld_opd_plus as O
    orig = getattr(O, "_orig_load_policy", O.load_policy)
    if merge and not trainable:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(student); tok.pad_token = tok.pad_token or tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(student, dtype=torch.bfloat16, device_map={"": 0}).eval()
        if adapter: model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
    else:
        tok, model, _ = orig(student, adapter, trainable)
    return tok, model, policy_class()(model, tok)


def generate(tok, model, system, user, max_new_tokens, temperature=0.0):
    import torch
    msg = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    x = tok(msg, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        o = model.generate(**x, max_new_tokens=max_new_tokens, do_sample=temperature > 0, temperature=max(temperature, 1e-5), top_p=1.0, pad_token_id=tok.pad_token_id)
    return " ".join(tok.decode(o[0, x.input_ids.shape[1]:], skip_special_tokens=True).split())


# ----------------------------------------------------------------------------- demos / SFT
def cmd_demos(a):
    eps = [e for e in L(ROOT / a.src) if e["won"]][: a.n]; out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for e in eps:
            e = dict(e); e["plan"] = plan_from_actions([s["action"] for s in e["steps"]]); f.write(json.dumps(e) + "\n")
    print(f"{len(eps)} plan demos -> {a.out}; example plan: {json.loads(open(out).readline())['plan']}")


def cmd_sft(a):
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    rng = random.Random(a.seed); eps = [e for e in L(ROOT / a.traj) if e["won"]][: a.n_demos]
    pairs = []
    for e in eps:
        pairs.append((render_plan_prompt(e["initial_obs"]), e["plan"]))
        for t, s in enumerate(e["steps"]): pairs.append((history_at(e, t, a.history).render_prompt(), s["action"]))
    rng.shuffle(pairs); print(f"{len(eps)} demos, {len(pairs)} pairs ({len(eps)} plan + {len(pairs) - len(eps)} action)", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model); tok.pad_token = tok.pad_token or tok.eos_token
    chat = lambda u: tok.apply_chat_template([{"role": "system", "content": PLAN_SYSTEM}, {"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)

    class DS(Dataset):
        def __init__(self):
            self.items = []
            for u, act in pairs:
                p = tok(chat(u), add_special_tokens=False).input_ids; t = tok(act, add_special_tokens=False).input_ids + [tok.eos_token_id]
                self.items.append(((p + t)[-a.max_len:], ([-100] * len(p) + t)[-a.max_len:]))
            print(f"max prompt+target length {max(len(x[0]) for x in self.items)} tokens", flush=True)
        def __len__(self): return len(self.items)
        def __getitem__(self, i): return {"input_ids": self.items[i][0], "labels": self.items[i][1]}

    def collate(b):
        m = max(len(x["input_ids"]) for x in b); pad = tok.pad_token_id
        return {"input_ids": torch.tensor([x["input_ids"] + [pad] * (m - len(x["input_ids"])) for x in b]),
                "labels": torch.tensor([x["labels"] + [-100] * (m - len(x["labels"])) for x in b]),
                "attention_mask": torch.tensor([[1] * len(x["input_ids"]) + [0] * (m - len(x["input_ids"])) for x in b])}
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    args = TrainingArguments(output_dir=str(ROOT / a.out), per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.grad_accum, num_train_epochs=a.epochs,
                             learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=20, logging_steps=10, save_strategy="no", bf16=True, report_to=[],
                             remove_unused_columns=False, seed=a.seed)
    Trainer(model=model, args=args, train_dataset=DS(), data_collator=collate).train()
    model.save_pretrained(str(ROOT / a.out / "final")); tok.save_pretrained(str(ROOT / a.out / "final")); print("SAVED", flush=True)


# ----------------------------------------------------------------------------- collect
def to_release(ep, hint=True):
    traj = [{"kind": "user", "content": ep["initial_obs"]}]
    for k, s in enumerate(ep["steps"]):
        traj.append({"kind": "action", "step_number": k + 1, "reasoning": ("Plan: " + ep["plan"]) if k == 0 else None, "action": s["action"],
                     "observation": s["next_obs"], "is_final_answer": False})
    q = ep["initial_obs"] + ("\n\n" + HINT if hint else "")
    return {"id": ep["id"], "framework": "alfagent", "benchmark": "textcraft", "modality": "text", "task": {"query": q, "answer": ""},
            "trajectory": traj, "ground_truth": {}, "extras": {"agents": [{"id": "agent", "role": "agent"}]}}


def cmd_collect(a):
    import torch
    torch.manual_seed(a.seed); random.seed(a.seed)
    tasks = splits()[a.split][a.offset:]; tasks = tasks[: a.n] if a.n else tasks           # shard BEFORE the resume filter
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True); done = set()
    if out.exists():   # tolerate a truncated last line
        keep = []
        for l in open(out):
            try: r = json.loads(l)
            except Exception: continue
            if r["task"] not in done: done.add(r["task"]); keep.append(json.dumps(r) + "\n")
        open(out, "w").writelines(keep)
    todo = [t for t in tasks if t not in done]; print(f"split {a.split}: {len(tasks)} tasks, {len(done)} done, {len(todo)} to do", flush=True)
    tok, model, pol = load_policy(a.model, a.adapter, merge=True); env = TextCraftEnv()
    rel = open(str(out).replace(".jsonl", ".release.jsonl"), "a"); t0 = time.time(); n = wins = 0
    with open(out, "a") as f:
        for tid in todo:
            obs0 = env.reset(tid); gold = env.expert_actions(); B = a.budget or budget_for(len(gold))
            plan = generate(tok, model, PLAN_SYSTEM, render_plan_prompt(obs0), a.plan_tokens, a.plan_temperature)
            past = []; steps = []; won = False; obs = obs0
            for k in range(B):
                adm = env.admissible(); lp = pol.score_admissible_commands_cached(PlanHistory(obs0, plan, past, adm, a.history)); cmds = list(lp)
                if a.temperature <= 0: act = max(cmds, key=lambda c: lp[c])
                else:
                    p = torch.softmax(torch.tensor([lp[c] for c in cmds]) / a.temperature, 0); act = cmds[torch.multinomial(p, 1).item()]
                nxt, r, d = env.step(act); z = torch.logsumexp(torch.tensor(list(lp.values())), 0).item()
                steps.append({"obs": obs, "admissible": adm, "action": act, "next_obs": nxt, "reward": r, "p_action": round(math.exp(lp[act] - z), 4)})
                past.append((act, nxt)); obs = nxt
                if d: won = True; break
            ep = {"id": f"tc_{a.tag}_{a.split}_{tid}", "task": tid, "split": a.split, "goal": item_id_to_str(env.goal), "depth": tree().get_min_depth(env.goal),
                  "initial_obs": obs0, "plan": plan, "plan_oracle": plan_from_actions(gold), "steps": steps, "won": won, "n_steps": len(steps), "budget": B,
                  "expert_len": len(gold), "temperature": a.temperature, "plan_temperature": a.plan_temperature, "model": a.adapter or a.model}
            f.write(json.dumps(ep) + "\n"); f.flush()
            if not won: rel.write(json.dumps(to_release(ep)) + "\n"); rel.flush()
            n += 1; wins += won
            if n % 20 == 0: print(f"[{n}/{len(todo)}] success {wins}/{n} = {wins / n:.3f} {(time.time() - t0) / n:.1f}s/task", flush=True)
    print(f"DONE collect {wins}/{n}", flush=True)


def cmd_plan_report(a):
    """plan quality vs the oracle plan (diagnostic only): Need-list exact match, per-item count errors, success by plan correctness."""
    for path in a.files:
        E = L(ROOT / path); ok = []; under = over = miss = extra = unp = 0
        for e in E:
            p, g = parse_need(e["plan"]), parse_need(e["plan_oracle"])
            if p is None: unp += 1; ok.append(False); continue
            ok.append(p == g)
            for it, n in g.items():
                if it not in p: miss += 1
                elif p[it] < n: under += 1
                elif p[it] > n: over += 1
            extra += sum(1 for it in p if it not in g)
        W = [e["won"] for e in E]; A = [w for w, o in zip(W, ok) if o]; B = [w for w, o in zip(W, ok) if not o]
        print(f"== {path}: plan Need-list exact {sum(ok)}/{len(E)} = {sum(ok) / max(1, len(E)):.3f}; unparsable {unp}; item errors: under-count {under}, over-count {over}, "
              f"missing item {miss}, extra item {extra} | success with exact plan {sum(A)}/{len(A)}, with wrong plan {sum(B)}/{len(B)}")


# ----------------------------------------------------------------------------- Model A release / variant choice
def cmd_fails(a):
    eps = L(ROOT / a.traj); first = eps[: a.first]; fails = [e for e in first if not e["won"]]
    D = ROOT / a.out_dir; D.mkdir(parents=True, exist_ok=True)
    open(D / "ids_fail.jsonl", "w").writelines(json.dumps({"id": e["id"]}) + "\n" for e in fails)
    K = max(1, round(a.k_frac * len(fails)))
    json.dump({"n_first": len(first), "n_fail_first": len(fails), "k_frac": a.k_frac, "K": K, "n_total": len(eps), "n_won_total": sum(e["won"] for e in eps)},
              open(D / "K.json", "w"), indent=1)
    rows = [to_release(e) for e in fails]
    for i in range(a.n_shards): open(D / f"release_s{i}.jsonl", "w").writelines(json.dumps(r) + "\n" for r in rows[i::a.n_shards])
    print(f"fails: {len(eps)} episodes ({sum(e['won'] for e in eps)} won); first {len(first)}: {len(fails)} failures -> K = round({a.k_frac} x {len(fails)}) = {K}; "
          f"release (task statement + hint as query, plan as step-1 reasoning) in {a.n_shards} shard files")


def cmd_variant(a):
    """agreement of each Model A variant's step with the oracle reference first_waste on the TRAINING failures; picks the better variant and
    copies its predictions to <out_dir>/modelA_pred_s*.jsonl (the file name textcraft_round.py pointer reads)."""
    import shutil
    from textcraft_evidence import refs
    from textcraft_round import merge
    D = ROOT / a.out_dir; eps = {e["id"]: e for e in L(ROOT / a.traj)}; ids = [r["id"] for r in L(D / "ids_fail.jsonl")]
    env = TextCraftEnv(); fw = {}
    for i in ids: fw[i] = refs(env, eps[i])[0]
    res = {}
    for v in a.variants:
        pred = merge(sorted(glob.glob(str(D / f"modelA_{v}" / "modelA_pred_s*.jsonl"))), lambda r: r["id"])
        miss = [i for i in ids if i not in pred]
        if miss: raise SystemExit(f"variant {v}: {len(miss)} predictions missing")
        th = {i: max(0, min(int(pred[i]["t_hat"]) if pred[i]["t_hat"] is not None else 0, len(eps[i]["steps"]) - 1)) for i in ids}
        ok = [i for i in ids if fw[i] is not None]
        ex = sum(th[i] == fw[i] for i in ok); w1 = sum(abs(th[i] - fw[i]) <= 1 for i in ok)
        ff = lambda e: next((k for k, s in enumerate(e["steps"]) if s["next_obs"].startswith("Could not")), None)
        res[v] = {"exact": ex / max(1, len(ok)), "within1": w1 / max(1, len(ok)), "n": len(ok), "t0": sum(th[i] == 0 for i in ids) / len(ids),
                  "eq_first_failed": sum(th[i] == ff(eps[i]) for i in ids) / len(ids), "earlier": sum(th[i] < fw[i] for i in ok) / max(1, len(ok))}
        print(f"variant {v}: exact agreement with first_waste {ex}/{len(ok)} = {res[v]['exact']:.3f}, within 1 {res[v]['within1']:.3f}, earlier {res[v]['earlier']:.2f}; "
              f"t0 share {res[v]['t0']:.2f}; equals the first failed action {res[v]['eq_first_failed']:.2f}")
    best = max(res, key=lambda v: (res[v]["exact"], res[v]["within1"]))
    for f_ in glob.glob(str(D / "modelA_pred_s*.jsonl")): os.remove(f_)
    for f_ in sorted(glob.glob(str(D / f"modelA_{best}" / "modelA_pred_s*.jsonl"))): shutil.copy(f_, D / Path(f_).name)
    json.dump({"choice": best, "results": res}, open(D / "modelA_variant.json", "w"), indent=1)
    print(f"CHOSEN Model A variant: {best}")


# ----------------------------------------------------------------------------- analysis-first full-view teacher
def cmd_teacher(a):
    import torch
    from teacher_reason_auc import NoteHistory
    from textcraft_round import TYPE_WORDS
    D = ROOT / a.out_dir; eps = {e["id"]: e for e in L(ROOT / a.traj)}
    steps = [(r["id"], r["steps"][0]) for r in L(D / "pointer.jsonl")]; diag = {r["id"]: r for r in L(D / "diag_modelA.raw.jsonl")}
    steps = steps[a.shard[0]::a.shard[1]]                                     # shard BEFORE the resume filter
    out = D / f"targets_s{a.shard[0]}.jsonl"; done = set()
    if out.exists():
        keep = []; own = set(steps)
        for l in open(out):
            try: r = json.loads(l)
            except Exception: continue
            k = (r["id"], r["t"])
            if k in own and k not in done: done.add(k); keep.append(json.dumps(r) + "\n")
        open(out, "w").writelines(keep)
    todo = [s for s in steps if s not in done]; print(f"shard {a.shard}: {len(steps)} states, {len(done)} done, {len(todo)} to do", flush=True)
    tok, model, pol = load_policy(a.model, str(ROOT / a.adapter) if not os.path.isabs(a.adapter) else a.adapter); model.eval()
    with open(out, "a") as f, torch.no_grad():
        for k, (i, t) in enumerate(todo):
            e = eps[i]; lp_plain = pol.score_admissible_commands_cached(history_at(e, t))
            hh = FullHindsightPlan(e, t); parts = []; d = diag.get(i)
            if d:
                typ = d.get("type"); parts.append(f"Diagnosis (an error attributor trained on this agent's own trajectories): the first decisive error was at step {d['t_hat']}"
                                                  + (f", type: {typ} ({TYPE_WORDS.get(typ, typ)})" if typ else "") + ".")
            note = NoteHistory(hh, "\n".join(parts) if parts else "Note: this attempt FAILED.", hind=hh)
            an = generate(tok, model, GEN_SYSTEM, note.render_prompt().replace("Next action:", ANALYSIS_Q_PLAN), a.analysis_tokens)
            parts.append("Analysis: " + an)
            lp = pol.score_admissible_commands_cached(NoteHistory(hh, "\n".join(parts), hind=hh))
            f.write(json.dumps({"id": i, "t": t, "a_minus": e["steps"][t]["action"], "admissible": list(lp), "teacher_logprobs": lp, "plain_logprobs": lp_plain,
                                "mode": "informed_plan", "analysis": an, "diag": d}) + "\n"); f.flush()
            if k % 20 == 0: print(f"[{k + 1}/{len(todo)}]", flush=True)
    print("DONE informed_teacher", flush=True)


# ----------------------------------------------------------------------------- replay plan rows / trainer
def cmd_replay_plan(a):
    rows = L(ROOT / a.replay); eps = {e["id"]: e for e in L(ROOT / a.traj)}; out = []
    for r in rows:
        out.append(r)
        if r["t"] == 0: out.append({"kind": "sft", "src": r["src"], "id": r["id"], "t": -1, "action": eps[r["id"]]["plan"]})
    open(ROOT / a.out, "w").writelines(json.dumps(r) + "\n" for r in out)
    print(f"replay with plan targets: {len(rows)} action rows + {len(out) - len(rows)} plan-generation rows (t=0 rows) -> {a.out}")


def cmd_train(a):
    import functools
    import alfworld_opd_plus as O
    O._orig_load_policy = O.load_policy
    O.load_policy = lambda student, adapter, trainable=False: load_policy(student, adapter, trainable)
    O.history_at = lambda ep, t: history_at(ep, t)
    O.student_logprobs_grad = functools.partial(O.student_logprobs_grad, batch_size=int(os.environ.get("TC_LP_CHUNK", "8")))
    O.cmd_train(a)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("demos"); s.add_argument("--src", default="reports/textcraft/gold_train.jsonl"); s.add_argument("--n", type=int, default=100); s.add_argument("--out", required=True)
    s = sub.add_parser("sft"); s.add_argument("--traj", required=True); s.add_argument("--n_demos", type=int, default=100); s.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    s.add_argument("--out", required=True); s.add_argument("--epochs", type=float, default=3); s.add_argument("--lr", type=float, default=1e-4); s.add_argument("--bs", type=int, default=4)
    s.add_argument("--grad_accum", type=int, default=4); s.add_argument("--history", type=int, default=10); s.add_argument("--max_len", type=int, default=3072); s.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("collect"); c.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); c.add_argument("--adapter", required=True); c.add_argument("--split", default="test810")
    c.add_argument("--offset", type=int, default=0); c.add_argument("--n", type=int, default=0); c.add_argument("--budget", type=int, default=0); c.add_argument("--temperature", type=float, default=0.0)
    c.add_argument("--plan_temperature", type=float, default=0.0); c.add_argument("--plan_tokens", type=int, default=320); c.add_argument("--history", type=int, default=10)
    c.add_argument("--seed", type=int, default=0); c.add_argument("--out", required=True); c.add_argument("--tag", default="plan")
    s = sub.add_parser("plan_report"); s.add_argument("files", nargs="+")
    s = sub.add_parser("fails"); s.add_argument("--traj", required=True); s.add_argument("--out_dir", required=True); s.add_argument("--first", type=int, default=1000)
    s.add_argument("--k_frac", type=float, default=0.2); s.add_argument("--n_shards", type=int, default=2)
    s = sub.add_parser("variant"); s.add_argument("--traj", required=True); s.add_argument("--out_dir", required=True); s.add_argument("--variants", nargs="+", default=["no_observations", "full"])
    s = sub.add_parser("teacher"); s.add_argument("--traj", required=True); s.add_argument("--out_dir", required=True); s.add_argument("--adapter", required=True)
    s.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); s.add_argument("--shard", type=int, nargs=2, default=[0, 1]); s.add_argument("--analysis_tokens", type=int, default=256)
    s = sub.add_parser("replay_plan"); s.add_argument("--replay", required=True); s.add_argument("--traj", required=True); s.add_argument("--out", required=True)
    t = sub.add_parser("train"); t.add_argument("--data", nargs="+", required=True); t.add_argument("--student", default="Qwen/Qwen2.5-3B-Instruct"); t.add_argument("--adapter", required=True)
    t.add_argument("--out", required=True); t.add_argument("--lr", type=float, default=1e-5); t.add_argument("--batch", type=int, default=8)
    t.add_argument("--epochs", type=int, default=1); t.add_argument("--beta", type=float, default=0.1); t.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"demos": cmd_demos, "sft": cmd_sft, "collect": cmd_collect, "plan_report": cmd_plan_report, "fails": cmd_fails, "variant": cmd_variant, "teacher": cmd_teacher,
     "replay_plan": cmd_replay_plan, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
