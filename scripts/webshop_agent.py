"""WebShop agent (mirrors sciworld_agent.py): generative policy, actions 'search[...]' / 'click[...]'.
  demos   -- scripted-oracle expert trajectories on train sessions (uses the goal's hidden target) -> reports/webshop/gold_train.jsonl
  sft     -- LoRA SFT on the first --n_demos demos (weak student)
  collect -- roll out a policy on a split (sessions), log trajectories in the shared schema (+ reward); won = reward == 1
A generated click whose value is not clickable is mapped to the closest clickable (difflib) and the mapping is logged.
"""
import argparse, difflib, json, os, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("ERROR_ATTR_ROOT", str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from error_attr.env.webshop_env import WebShopEnv, SPLITS

SYSTEM = ("You are a web shopping agent. You will be given a shopping instruction and the history of web pages and actions. "
          "Respond with exactly one next action and nothing else: either search[keywords] or click[value], where value is one "
          "of the clickable items shown on the current page (e.g. click[b09hkzs7lv], click[buy now], click[next >]).")
OBS_CHARS = 1500
# prompt v2 (Oct 8, user: "remember to buy at the end"): the agent is told the action budget and that only a purchase is scored, and
# every prompt shows how many actions were used.  Selected with WEBSHOP_PROMPT=v2 (default v1 = the original prompt, for old runs).
import os as _os
PROMPT_V2 = _os.environ.get("WEBSHOP_PROMPT", "v1") == "v2"; BUDGET = int(_os.environ.get("WEBSHOP_BUDGET", "15"))
PROMPT_V3 = _os.environ.get("WEBSHOP_PROMPT", "v1") == "v3"   # v3 (Oct 8, user): ONE system-prompt sentence only, no per-step counter
if PROMPT_V3:
    SYSTEM = SYSTEM + " Once you have found a product that matches all the requirements of the instruction and selected the required options, remember to click[buy now]."
if PROMPT_V2:
    SYSTEM = SYSTEM + (f" You have at most {BUDGET} actions in total, and the episode is only scored if you click[buy now] before they run out:"
                       " once you have found a product that matches the instruction and selected the required options, buy it.")


def render(goal, steps, history=6):
    lines = [f"Instruction: {goal}", ""]
    for s in steps[-history:]:
        lines.append(f"Page: {s['obs'][:OBS_CHARS]}")
        if s.get("action") is not None: lines.append(f"Action: {s['action']}")
    if PROMPT_V2:
        used = sum(1 for s in steps if s.get("action") is not None); lines += ["", f"Actions used: {used} of {BUDGET}."]
    lines += ["", "Next action:"]
    return "\n".join(lines)


def chat(tok, user):
    return tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)


def cmd_demos(a):
    env = WebShopEnv(worker=f"demo{a.shard}"); out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set(); f = open(out, "a"); n = ok = 0
    sessions = list(SPLITS["train"])[a.shard::a.n_shards][: a.n]
    for s in sessions:
        i = f"ws_oracle_train_{s:05d}"
        if i in done: continue
        r = env.oracle(s, a.budget); n += 1; ok += r["reward"] >= 1.0
        f.write(json.dumps({"id": i, "session": s, "split": "train", "goal": r["instruction"], "steps": r["steps"], "won": r["reward"] >= 1.0, "final_score": 100 * r["reward"], "n_steps": len(r["steps"]), "model": "oracle"}) + "\n"); f.flush()
        if n % 20 == 0: print(f"[{n}/{len(sessions)}] oracle success {ok}/{n}", flush=True)
    print(f"DONE demos {ok}/{n}", flush=True)


def cmd_sft(a):
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    rng = random.Random(a.seed); eps = [e for e in map(json.loads, open(ROOT / a.traj)) if e["won"]][: a.n_demos]
    pairs = []
    for e in eps:
        hist = []
        for s in e["steps"]:
            hist.append({"obs": s["obs"], "action": None}); pairs.append((render(e["goal"], hist, a.history), s["action"])); hist[-1]["action"] = s["action"]
    rng.shuffle(pairs); print(f"{len(eps)} demos, {len(pairs)} pairs", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model); tok.pad_token = tok.pad_token or tok.eos_token

    class DS(Dataset):
        def __init__(self):
            self.items = []
            for u, act in pairs:
                p = tok(chat(tok, u), add_special_tokens=False).input_ids; t = tok(act, add_special_tokens=False).input_ids + [tok.eos_token_id]
                self.items.append(((p + t)[-a.max_len:], ([-100] * len(p) + t)[-a.max_len:]))
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
                             learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=20, logging_steps=20, save_strategy="no", bf16=True, report_to=[], remove_unused_columns=False)
    Trainer(model=model, args=args, train_dataset=DS(), data_collator=collate).train()
    model.save_pretrained(str(ROOT / a.out / "final")); tok.save_pretrained(str(ROOT / a.out / "final")); print("SAVED", flush=True)


def map_action(raw, clicks, has_search):
    """map the generated text to a valid env action; returns (action, mapped?)."""
    raw = raw.strip()
    if raw.startswith("search[") and raw.endswith("]") and has_search: return raw, False
    if raw.startswith("click[") and raw.endswith("]"):
        v = raw[6:-1].strip().lower()
        if f"click[{v}]" in clicks: return f"click[{v}]", v != raw[6:-1].strip()
    m = difflib.get_close_matches(raw.lower(), clicks, n=1, cutoff=0.0)
    if m: return m[0], True
    return (f"search[{raw}]", True) if has_search else (raw, True)


def cmd_collect(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(a.model); tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    if a.adapter: model = PeftModel.from_pretrained(model, a.adapter)
    model.eval(); env = WebShopEnv(worker=a.tag); out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    sessions = list(SPLITS[a.split])[a.offset: a.offset + a.n] if a.n else list(SPLITS[a.split])[a.offset:]
    f = open(out, "a"); t0 = time.time(); n = wins = 0; tot = 0.0
    for s in sessions:
        i = f"ws_{a.tag}_{a.split}_{s:05d}"
        if i in done: continue
        obs, goal = env.reset(s); hist = []; steps = []; reward = 0.0; dn = False
        for k in range(a.budget):
            hist.append({"obs": obs, "action": None})
            ids = tok(chat(tok, render(goal, hist, a.history)), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            with torch.no_grad():
                g = model.generate(ids, max_new_tokens=32, do_sample=a.temperature > 0, temperature=max(a.temperature, 1e-5), top_p=1.0, pad_token_id=tok.pad_token_id)
            raw = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]
            clicks, has_search = env.valid_actions(); act, mapped = map_action(raw, clicks, has_search)
            nxt, reward, dn = env.step(act); hist[-1]["action"] = act
            steps.append({"obs": obs, "action": act, "raw": raw, "mapped": mapped, "next_obs": nxt, "reward": reward, "clickables": clicks}); obs = nxt
            if dn: break
        won = reward >= 1.0; n += 1; wins += won; tot += reward
        f.write(json.dumps({"id": i, "session": s, "split": a.split, "goal": goal, "steps": steps, "won": won, "final_score": 100 * reward, "n_steps": len(steps),
                            "budget": a.budget, "temperature": a.temperature, "model": a.adapter or a.model}) + "\n"); f.flush()
        if n % 10 == 0: print(f"[{n}/{len(sessions)}] success {wins}/{n} = {wins / n:.3f} mean reward {tot / n:.3f} {time.time() - t0:.0f}s", flush=True)
    print(f"DONE collect {wins}/{n} mean_reward {tot / max(1, n):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demos"); d.add_argument("--n", type=int, default=400); d.add_argument("--budget", type=int, default=30); d.add_argument("--shard", type=int, default=0); d.add_argument("--n_shards", type=int, default=1); d.add_argument("--out", default="reports/webshop/gold_train.jsonl")
    s = sub.add_parser("sft"); s.add_argument("--traj", default="reports/webshop/gold_train.jsonl"); s.add_argument("--n_demos", type=int, required=True)
    s.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); s.add_argument("--out", required=True); s.add_argument("--epochs", type=float, default=3)
    s.add_argument("--lr", type=float, default=1e-4); s.add_argument("--bs", type=int, default=4); s.add_argument("--grad_accum", type=int, default=4)
    s.add_argument("--history", type=int, default=6); s.add_argument("--max_len", type=int, default=3072); s.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("collect"); c.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); c.add_argument("--adapter", default=None)
    c.add_argument("--split", default="dev"); c.add_argument("--offset", type=int, default=0); c.add_argument("--n", type=int, default=0); c.add_argument("--budget", type=int, default=15)
    c.add_argument("--temperature", type=float, default=0.0); c.add_argument("--history", type=int, default=6); c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True); c.add_argument("--tag", default="x")
    a = ap.parse_args(); {"demos": cmd_demos, "sft": cmd_sft, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    main()
