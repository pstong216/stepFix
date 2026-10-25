"""TextCraft agent (mirrors scripts/webshop_agent.py; policy = the ALFWorld admissible-scoring policy: every admissible action is scored by
its log-probability under the model and the argmax (greedy) or a softmax sample at temperature T is taken).
  demos   -- oracle solver (scripts/textcraft_env.py expert_actions: uses only the shown crafting commands) on train tasks
             -> reports/textcraft/gold_train.jsonl (one task per train goal, distractor seed 0, pool order)
  sft     -- LoRA SFT on the first --n_demos demos (Qwen2.5-3B-Instruct, LoRA r32/alpha64/dropout .05 on all projections, lr 1e-4 cosine,
             warmup 20, effective batch 16, 3 epochs = webshop_agent.py sft; prompt = admissible-list prompt as in alfworld_sft.py)
  collect -- roll out a policy on a split; per-task budget = 2 x oracle plan length (min 6) unless --budget; writes the shared schema
             {id, task, split, goal, depth, initial_obs, steps:[{obs, admissible, action, next_obs, reward}], won, n_steps, budget, ...}
             and, for failures, a Who&When 'alfagent' release row (*.release.jsonl; integer step_number, reasoning None, no score).
  report  -- success / steps / failure-type breakdown of one or more collect files.
"""
import argparse, collections, json, os, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("ERROR_ATTR_ROOT", str(ROOT)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from textcraft_env import TextCraftEnv, splits, budget_for, parse_task, tree, item_id_to_str, str_to_item_id

SYSTEM = ("You are a crafting agent in a text-based Minecraft environment. You will be given the available crafting commands, a goal item, "
          "and the history of your actions and their results. Choose the single best next action from the list of admissible actions. "
          "Respond with the exact text of one admissible action and nothing else.")


def render(obs0, past, adm, history=10):
    """obs0 = 'Crafting commands: ... Goal: craft X.' (always shown); past = [(action, observation)]; the last `history` steps are shown."""
    lines = [obs0, ""]
    if len(past) > history: lines.append(f"(... {len(past) - history} earlier steps omitted)")
    for act, obs in past[-history:]:
        lines += [f"Action: {act}", f"Observation: {obs}"]
    lines.append("Admissible actions:")
    lines += [f"- {a}" for a in adm]
    lines += ["", "Next action:"]
    return "\n".join(lines)


class _Step:
    def __init__(self, adm): self.admissible_commands = adm


class TCHistory:
    """duck-types error_attr.policy.alfworld_agent.ALFWorldHistory for LLMPolicy (steps[-1].admissible_commands + render_prompt)."""
    def __init__(self, obs0, past, adm, history=10):
        self.obs0, self.past, self.adm, self.H = obs0, list(past), list(adm), history; self.steps = [_Step(self.adm)]
    def render_prompt(self, max_history_steps=None): return render(self.obs0, self.past, self.adm, self.H)


def history_at(e, t, history=10):
    return TCHistory(e["initial_obs"], [(s["action"], s["next_obs"]) for s in e["steps"][:t]], e["steps"][t]["admissible"], history)


def load_policy(model_name, adapter):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from error_attr.policy.alfworld_agent import LLMPolicy

    class TCPolicy(LLMPolicy):
        def _build_chat_prompt(self, history):
            return self.tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": history.render_prompt()}],
                                                      tokenize=False, add_generation_prompt=True)
    tok = AutoTokenizer.from_pretrained(model_name); tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map={"": 0}).eval()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
    return tok, model, TCPolicy(model, tok)


def cmd_demos(a):
    env = TextCraftEnv(); out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in splits()["train"] if t.endswith("_s0")][: a.n]; n = 0
    with open(out, "w") as f:
        for tid in tasks:
            obs0 = env.reset(tid); plan = env.expert_actions(); steps = []; won = False; obs = obs0
            for act in plan:
                adm = env.admissible(); assert act in adm, (tid, act)
                nxt, r, d = env.step(act); steps.append({"obs": obs, "admissible": adm, "action": act, "next_obs": nxt, "reward": r}); obs = nxt
                if d: won = True; break
            f.write(json.dumps({"id": f"tc_oracle_{tid}", "task": tid, "split": "train", "goal": item_id_to_str(env.goal), "depth": tree().get_min_depth(env.goal),
                                "initial_obs": obs0, "steps": steps, "won": won, "n_steps": len(steps), "model": "oracle"}) + "\n"); n += won
    print(f"DONE demos {n}/{len(tasks)} solved", flush=True)


def cmd_sft(a):
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    rng = random.Random(a.seed); eps = [e for e in map(json.loads, open(ROOT / a.traj)) if e["won"]][: a.n_demos]
    pairs = [(history_at(e, t, a.history).render_prompt(), s["action"]) for e in eps for t, s in enumerate(e["steps"])]
    rng.shuffle(pairs); print(f"{len(eps)} demos, {len(pairs)} pairs", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model); tok.pad_token = tok.pad_token or tok.eos_token
    chat = lambda u: tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)

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


def to_release(ep):
    """Who&When 'alfagent' row (as alfworld_collect.to_release / webshop_final.cmd_release): the task text as the user turn, one action entry
    per step with the environment's reply; no reward / success field anywhere."""
    traj = [{"kind": "user", "content": ep["initial_obs"]}]
    for k, s in enumerate(ep["steps"]):
        traj.append({"kind": "action", "step_number": k + 1, "reasoning": None, "action": s["action"], "observation": s["next_obs"], "is_final_answer": False})
    return {"id": ep["id"], "framework": "alfagent", "benchmark": "textcraft", "modality": "text", "task": {"query": "craft " + ep["goal"], "answer": ""},
            "trajectory": traj, "ground_truth": {}, "extras": {"agents": [{"id": "agent", "role": "agent"}]}}


def cmd_collect(a):
    import torch
    torch.manual_seed(a.seed); random.seed(a.seed)
    tasks = splits()[a.split][a.offset:]; tasks = tasks[: a.n] if a.n else tasks
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():   # Oct 9: tolerate a truncated last line (a step killed mid-write, e.g. by the dedicated-GPU restart): keep only parsable rows
        keep = []
        for l in open(out):
            try: r = json.loads(l)
            except Exception: continue
            if r["task"] not in done: done.add(r["task"]); keep.append(json.dumps(r) + "\n")
        open(out, "w").writelines(keep)
    todo = [t for t in tasks if t not in done]
    print(f"split {a.split}: {len(tasks)} tasks, {len(done)} done, {len(todo)} to do", flush=True)
    tok, model, pol = load_policy(a.model, a.adapter); env = TextCraftEnv()
    rel = open(str(out).replace(".jsonl", ".release.jsonl"), "a"); t0 = time.time(); n = wins = 0
    with open(out, "a") as f:
        for tid in todo:
            obs0 = env.reset(tid); L = len(env.expert_actions()); B = a.budget or budget_for(L); past = []; steps = []; won = False; obs = obs0
            for k in range(B):
                adm = env.admissible(); lp = pol.score_admissible_commands_cached(TCHistory(obs0, past, adm, a.history)); cmds = list(lp)
                if a.temperature <= 0: act = max(cmds, key=lambda c: lp[c])
                else:
                    p = torch.softmax(torch.tensor([lp[c] for c in cmds]) / a.temperature, 0); act = cmds[torch.multinomial(p, 1).item()]
                nxt, r, d = env.step(act)
                z = torch.logsumexp(torch.tensor(list(lp.values())), 0).item()
                steps.append({"obs": obs, "admissible": adm, "action": act, "next_obs": nxt, "reward": r, "p_action": round(float(torch.exp(torch.tensor(lp[act] - z))), 4),
                              **({"logprobs": lp} if a.save_logprobs else {})})
                past.append((act, nxt)); obs = nxt
                if d: won = True; break
            ep = {"id": f"tc_{a.tag}_{a.split}_{tid}", "task": tid, "split": a.split, "goal": item_id_to_str(env.goal), "depth": tree().get_min_depth(env.goal),
                  "initial_obs": obs0, "steps": steps, "won": won, "n_steps": len(steps), "budget": B, "expert_len": L, "temperature": a.temperature,
                  "model": a.adapter or a.model}
            f.write(json.dumps(ep) + "\n"); f.flush()
            if not won: rel.write(json.dumps(to_release(ep)) + "\n"); rel.flush()
            n += 1; wins += won
            if n % 20 == 0: print(f"[{n}/{len(todo)}] success {wins}/{n} = {wins / n:.3f} {(time.time() - t0) / n:.1f}s/task", flush=True)
    print(f"DONE collect {wins}/{n}", flush=True)


# ----------------------------------------------------------------------------- failure analysis
def tree_items(env):
    """items that appear in a recipe of the goal's recipe tree (outputs and ingredients, tags expanded to members)."""
    items = set()
    for r in env.shown:
        if r.recipe_str not in env.tree_recipes: continue
        for x in list(r.input_items) + [r.output_item]:
            nm = x.item_tag.item_id or x.item_tag.tag; items.add(nm)
            if x.item_tag.item_id is None: items.update(env.members(x.item_tag.tag))
    return items


def classify(e, env):
    """multi-label failure types + a primary type (the first problem in time order).  A failed in-tree craft is 'missing_ingredient:
    skipped' if a lacking ingredient was never obtained before in the episode, else 'missing_ingredient:count' (obtained, too few left)."""
    env.reset(e["task"]); ti = tree_items(env); ever = set()
    labels = []; first = None; fails = 0; reps = collections.Counter()
    def mark(lbl, k):
        nonlocal first
        labels.append(lbl)
        if first is None: first = (k, lbl)
    for k, s in enumerate(e["steps"]):
        a, o = s["action"], s["next_obs"]; ok = not o.startswith("Could not") and not o.startswith("Wrong")
        reps[(a, o)] += 1; inv = env.get_state(); env.step(a); ever |= set(env.inventory)
        if a.startswith("get "):
            nm = "minecraft:" + a.split(" ", 2)[2].replace(" ", "_")
            if nm not in ti: mark("wrong_item", k)
        elif a.startswith("craft "):
            out_name = a[6:].split(" using ")[0].split(" ", 1)[1]; ings = [x.split(" ", 1)[1] for x in a.split(" using ")[1].split(", ")]
            rs = [r for r in env.shown if r.output_item.item_tag.item_id == str_to_item_id(out_name)]
            if not any(r.recipe_str in env.tree_recipes for r in rs): mark("wrong_recipe", k)
            elif not ok:
                if any(env.t.is_tag(str_to_item_id(x)) for x in ings): mark("tag_literal", k)
                else:
                    cnt = {str_to_item_id(x.split(" ", 1)[1]): int(x.split(" ", 1)[0]) for x in a.split(" using ")[1].split(", ")}
                    lack = [i for i, c in cnt.items() if inv.get(i, 0) < c]
                    mark("missing_ingredient:count" if any(i in ever for i in lack) else "missing_ingredient:skipped", k)
        if not ok: fails += 1
    if any(v >= 3 and (o.startswith("Could not") or o.startswith("Wrong")) for (a, o), v in reps.items()): labels.append("invalid_loop")   # same failing action >= 3x
    if not first: first = (len(e["steps"]) - 1, "budget_exhausted_no_error")
    return sorted(set(labels)), first[1], first[0], fails


def cmd_report(a):
    env = TextCraftEnv()
    for path in a.files:
        eps = [json.loads(l) for l in open(ROOT / path)]; won = [e for e in eps if e["won"]]; fail = [e for e in eps if not e["won"]]
        print(f"== {path}: {len(won)}/{len(eps)} = {len(won) / max(1, len(eps)):.3f} success; mean steps {sum(e['n_steps'] for e in eps) / max(1, len(eps)):.2f} "
              f"(won {sum(e['n_steps'] for e in won) / max(1, len(won)):.2f} vs oracle {sum(e['expert_len'] for e in won) / max(1, len(won)):.2f}); mean budget {sum(e['budget'] for e in eps) / max(1, len(eps)):.1f}")
        byd = collections.defaultdict(list)
        for e in eps: byd[e["depth"]].append(e["won"])
        print("   by depth: " + ", ".join(f"d{d} {sum(v)}/{len(v)}" for d, v in sorted(byd.items())))
        multi = collections.Counter(); prim = collections.Counter(); fr = []
        for e in fail:
            labels, p, k, nf = classify(e, env); multi.update(labels); prim[p] += 1; fr.append(nf / max(1, e["n_steps"]))
        if fail:
            print(f"   failures {len(fail)}: primary (first problem) {dict(prim.most_common())}")
            print(f"   prevalence (any) {dict((k, f'{v / len(fail):.2f}') for k, v in multi.most_common())}; mean share of failed actions per failed episode {sum(fr) / len(fr):.2f}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demos"); d.add_argument("--n", type=int, default=378); d.add_argument("--out", default="reports/textcraft/gold_train.jsonl")
    s = sub.add_parser("sft"); s.add_argument("--traj", default="reports/textcraft/gold_train.jsonl"); s.add_argument("--n_demos", type=int, default=100)
    s.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); s.add_argument("--out", required=True); s.add_argument("--epochs", type=float, default=3)
    s.add_argument("--lr", type=float, default=1e-4); s.add_argument("--bs", type=int, default=4); s.add_argument("--grad_accum", type=int, default=4)
    s.add_argument("--history", type=int, default=10); s.add_argument("--max_len", type=int, default=3072); s.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("collect"); c.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); c.add_argument("--adapter", default=None)
    c.add_argument("--split", default="test"); c.add_argument("--offset", type=int, default=0); c.add_argument("--n", type=int, default=0); c.add_argument("--budget", type=int, default=0)
    c.add_argument("--temperature", type=float, default=0.0); c.add_argument("--history", type=int, default=10); c.add_argument("--seed", type=int, default=0)
    c.add_argument("--save_logprobs", action="store_true"); c.add_argument("--out", required=True); c.add_argument("--tag", default="x")
    r = sub.add_parser("report"); r.add_argument("files", nargs="+")
    a = ap.parse_args(); {"demos": cmd_demos, "sft": cmd_sft, "collect": cmd_collect, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
