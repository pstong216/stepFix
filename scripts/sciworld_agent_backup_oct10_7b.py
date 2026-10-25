"""ScienceWorld agent: generative policy (no admissible list in the prompt; ~400+ valid actions per step), same ReAct-style
history as the ALFWorld student. A generated command that the environment does not accept is mapped to the closest
valid action (difflib), and the mapping is logged.
  sft     -- LoRA SFT on gold trajectories (--per_task caps demonstrations per task type -> weak student / strong teacher)
  collect -- roll out a policy (temperature T) on a split, log trajectories in the shared schema (+ score per step)
"""
import argparse, difflib, json, os, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("ERROR_ATTR_ROOT", str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

SYSTEM = ("You are an agent completing a science experiment task in a simulated text environment. You will be given "
          "the task and a history of observations and actions. Respond with the single next action as a short command "
          "(for example 'open door to kitchen', 'pick up thermometer', 'focus on water') and nothing else.")
OBS_CHARS = 700


def render(goal, steps, history=10):
    """steps: list of dicts with obs and (except the last) action."""
    lines = [f"Task: {goal}", ""]
    for s in steps[-history:]:
        lines.append(f"Observation: {s['obs'][:OBS_CHARS]}")
        if s.get("action") is not None: lines.append(f"Action: {s['action']}")
    lines += ["", "Next action:"]
    return "\n".join(lines)


def chat(tok, user):
    return tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)


def cmd_sft(a):
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    rng = random.Random(a.seed); eps = [json.loads(l) for l in open(ROOT / a.traj)]
    by = {}
    for e in eps: by.setdefault(e["task"], []).append(e)
    sel = []
    for t, es in sorted(by.items()): rng.shuffle(es); sel += es[: a.per_task]
    pairs = []
    for e in sel:
        hist = []
        for s in e["steps"]:
            hist.append({"obs": s["obs"], "action": None}); pairs.append((render(e["goal"], hist, a.history), s["action"])); hist[-1]["action"] = s["action"]
    rng.shuffle(pairs); print(f"{len(sel)} trajectories, {len(pairs)} pairs", flush=True)
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


def cmd_collect(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from error_attr.env.scienceworld_env import make, reset, step, valid_actions
    tok = AutoTokenizer.from_pretrained(a.model); tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    if a.adapter: model = PeftModel.from_pretrained(model, a.adapter)
    bmap = json.loads(a.budget_map) if a.budget_map else {}   # Oct 10: per-task budgets (long-budget supplementary eval); default = --budget for all
    model.eval(); env = make(max([a.budget] + list(bmap.values())) + 5); rng = random.Random(a.seed)
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():   # Oct 9: tolerate a line truncated by preemption (skip it; re-terminate the file so the next row starts on a new line)
        for l in open(out):
            try: done.add(json.loads(l)["id"])
            except Exception: pass
        if out.stat().st_size and not open(out, "rb").read()[-1:] == b"\n": open(out, "a").write("\n")
    jobs = []
    for task in env.get_task_names():
        env.load(task, 0)
        vs = list({"train": env.get_variations_train, "dev": env.get_variations_dev, "test": env.get_variations_test}[a.split]())
        rng.shuffle(vs); jobs += [(task, v) for v in (vs[a.job_slice[0]: a.job_slice[1]] if a.job_slice else vs[: a.per_task])]   # Oct 9 23:00: --job_slice lo hi = per-task positions lo..hi-1 of the same seeded shuffle (extend a collection without redoing it)
    if a.tasks: jobs = [j for j in jobs if j[0] in set(a.tasks)]   # Oct 10: task filter AFTER the seeded shuffle (same variations as the full list)
    if a.shard: jobs = jobs[a.shard[0]::a.shard[1]]   # Oct 9: round-robin shards over the same job list
    f = open(out, "a"); t0 = time.time(); n = wins = 0
    for task, v in jobs:
        i = f"sw_{a.tag}_{task}_{v}"
        if i in done: continue
        obs, info = reset(env, task, v); goal = info["taskDesc"]; hist = []; steps = []; score = 0; dn = False
        B = bmap.get(task, a.budget)
        for k in range(B):
            hist.append({"obs": obs, "action": None})
            ids = tok(chat(tok, render(goal, hist, a.history)), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            with torch.no_grad():
                g = model.generate(ids, max_new_tokens=24, do_sample=a.temperature > 0, temperature=max(a.temperature, 1e-5), top_p=1.0, pad_token_id=tok.pad_token_id)
            raw = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]
            va = valid_actions(env); act = raw; mapped = False
            if raw not in va and a.snap == "fuzzy":   # Oct 4 behaviour (BUG, Oct 9): snaps ANY text, e.g. 'go to hallway' -> fatal 'focus on hallway'
                m = difflib.get_close_matches(raw, va, n=1, cutoff=0.0); act = m[0] if m else raw; mapped = act != raw
            elif raw not in va and a.snap == "safe":   # Oct 9 fix: same verb only, high similarity, never onto 'focus on'; else pass raw (env ignores invalid text)
                vb = raw.split(" ")[0] if raw else ""
                pool = [x for x in va if x.split(" ")[0] == vb and not (x.startswith("focus on") and not raw.startswith("focus on"))]
                m = difflib.get_close_matches(raw, pool, n=1, cutoff=0.85); act = m[0] if m else raw; mapped = act != raw
            nxt, score, dn, info = step(env, act)
            hist[-1]["action"] = act
            steps.append({"obs": obs, "action": act, "raw": raw, "mapped": mapped, "next_obs": nxt, "score": score}); obs = nxt
            if dn: break
        won = score >= 100; n += 1; wins += won
        f.write(json.dumps({"id": i, "task": task, "variation": v, "split": a.split, "goal": goal, "steps": steps, "won": won,
                            "final_score": score, "n_steps": len(steps), "budget": B, "temperature": a.temperature, "model": a.adapter or a.model}) + "\n"); f.flush()
        if n % 10 == 0: print(f"[{n}/{len(jobs)}] success {wins}/{n} = {wins / n:.3f} {time.time() - t0:.0f}s", flush=True)
    print(f"DONE collect {wins}/{n}", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sft"); s.add_argument("--traj", default="reports/sciworld/gold_train.jsonl"); s.add_argument("--per_task", type=int, required=True)
    s.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); s.add_argument("--out", required=True); s.add_argument("--epochs", type=float, default=2.0)
    s.add_argument("--lr", type=float, default=1e-4); s.add_argument("--bs", type=int, default=4); s.add_argument("--grad_accum", type=int, default=4)
    s.add_argument("--history", type=int, default=10); s.add_argument("--max_len", type=int, default=3072); s.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("collect"); c.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); c.add_argument("--adapter", default=None)
    c.add_argument("--split", default="dev"); c.add_argument("--per_task", type=int, default=5); c.add_argument("--budget", type=int, default=150)
    c.add_argument("--temperature", type=float, default=0.0); c.add_argument("--history", type=int, default=10); c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True); c.add_argument("--tag", default="x")
    c.add_argument("--snap", default="fuzzy", choices=["fuzzy", "safe", "none"], help="invalid-output handling; fuzzy = Oct 4 bug (kept for reproducibility)"); c.add_argument("--shard", type=int, nargs=2, default=None); c.add_argument("--job_slice", type=int, nargs=2, default=None); c.add_argument("--tasks", nargs="+", default=None); c.add_argument("--budget_map", default=None, help="JSON {task: budget}")
    a = ap.parse_args(); {"sft": cmd_sft, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    main()
