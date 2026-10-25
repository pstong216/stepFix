"""Planner-free progress estimator learned from the student's own WON trajectories.
  train  -- SFT the student (LoRA, from its adapter) to answer "Remaining steps: k" after the policy prompt, where
            k = (length of the won trajectory) - t, capped at --cap.
  score  -- for every step of the given failed trajectories, E[k] under the model (teacher-forced over k = 0..cap).
  pointer-- first action t whose next state is predicted farther from success: E[k]_{t+1} - E[k]_t > --delta
            (the same definition as the planner's t_first_regress, with the planner replaced by the estimator).
"""
import argparse, copy, json, math, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from error_attr.policy.alfworld_agent import LLMPolicy
from alfworld_attr_opd import history_at, load_jsonl

PREFIX = "Remaining steps: "


def load(student, adapter, trainable):
    tok = AutoTokenizer.from_pretrained(student); tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(student, dtype=torch.bfloat16, device_map={"": 0})
    if trainable:
        base.gradient_checkpointing_enable(); base.enable_input_require_grads()
        if adapter:
            m = PeftModel.from_pretrained(base, adapter, is_trainable=True)
        else:   # untrained base model: fresh LoRA
            from peft import LoraConfig, get_peft_model
            m = get_peft_model(base, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
                                                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        m.train()
    else:
        m = PeftModel.from_pretrained(base, adapter).eval()
    return tok, m, LLMPolicy(m, tok)


def cmd_train(a):
    rng = random.Random(0); eps = [e for e in load_jsonl(ROOT / a.traj) if e["won"]]
    items = [(e, t, min(len(e["steps"]) - t, a.cap)) for e in eps for t in range(len(e["steps"]))]
    rng.shuffle(items); items = items[: a.n]
    print(f"{len(items)} states from {len(eps)} won trajectories", flush=True)
    tok, model, pol = load(a.student, a.adapter, True)
    params = [p for p in model.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True); t0 = time.time(); run = 0.0
    for k, (e, t, rem) in enumerate(items):
        prompt = pol._build_chat_prompt(history_at(e, t)) + PREFIX
        p_ids = tok(prompt, add_special_tokens=False).input_ids; t_ids = tok(f"{rem}\n", add_special_tokens=False).input_ids
        ids = torch.tensor([p_ids + t_ids], device=model.device); lab = torch.tensor([[-100] * len(p_ids) + t_ids], device=model.device)
        loss = model(input_ids=ids, labels=lab).loss; (loss / a.batch).backward(); run += loss.item()
        if (k + 1) % a.batch == 0 or k == len(items) - 1:
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
        if (k + 1) % 200 == 0: print(f"[{k + 1}/{len(items)}] loss {run / 200:.3f} {time.time() - t0:.0f}s", flush=True); run = 0.0
    model.save_pretrained(str(out / "adapter")); tok.save_pretrained(str(out / "adapter")); print("DONE train", flush=True)


@torch.no_grad()
def expected_remaining(model, tok, pol, h, cap, cands):
    ids = tok(pol._build_chat_prompt(h) + PREFIX, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device); plen = ids.shape[1]
    out = model(input_ids=ids, use_cache=True); first = torch.log_softmax(out.logits[0, -1].float(), -1)
    tails = [c[:-1] for c in cands]; L = max(len(t) for t in tails); m = len(cands); pad = tok.pad_token_id
    c = copy.deepcopy(out.past_key_values); c.batch_repeat_interleave(m)
    inp = torch.full((m, L), pad, dtype=torch.long); att = torch.zeros((m, plen + L), dtype=torch.long); att[:, :plen] = 1
    for i, t in enumerate(tails): inp[i, :len(t)] = torch.tensor(t); att[i, plen: plen + len(t)] = 1
    tl = torch.log_softmax(model(input_ids=inp.to(model.device), attention_mask=att.to(model.device), past_key_values=c).logits.float(), -1)
    lps = []
    for i, cid in enumerate(cands):
        lp = first[cid[0]].item()
        if len(cid) > 1: lp += tl[i, :len(cid) - 1].gather(1, torch.tensor(cid[1:], device=model.device).unsqueeze(1)).sum().item()
        lps.append(lp)
    mx = max(lps); p = [math.exp(x - mx) for x in lps]; z = sum(p)
    return sum(k * pk for k, pk in enumerate(p)) / z


def cmd_score(a):
    eps = {e["id"]: e for e in load_jsonl(ROOT / a.traj)}
    ids = [r["id"] for r in load_jsonl(ROOT / a.ids_from)] if a.ids_from else [i for i, e in eps.items() if not e["won"]]
    done = {json.loads(l)["id"] for l in open(ROOT / a.out)} if (ROOT / a.out).exists() else set()
    ids = [i for i in ids if i not in done]; print(f"{len(ids)} failures to score", flush=True)
    tok, model, pol = load(a.student, a.adapter, False)
    cands = [tok(f"{k}\n", add_special_tokens=False).input_ids for k in range(a.cap + 1)]
    with open(ROOT / a.out, "a") as f:
        for j, i in enumerate(ids):
            e = eps[i]; ex = [expected_remaining(model, tok, pol, history_at(e, t), a.cap, cands) for t in range(len(e["steps"]))]
            f.write(json.dumps({"id": i, "exp_remaining": ex}) + "\n"); f.flush()
            if j % 20 == 0: print(f"[{j + 1}/{len(ids)}]", flush=True)
    print("DONE score", flush=True)


def cmd_pointer(a):
    with open(ROOT / a.out, "w") as f:
        for r in load_jsonl(ROOT / a.scores):
            ex = r["exp_remaining"]; n = len(ex)
            t = next((t for t in range(n - 1) if ex[t + 1] - ex[t] > a.delta), None)
            if t is None: t = max(range(n - 1), key=lambda t: ex[t + 1] - ex[t]) if n > 1 else 0
            f.write(json.dumps({"id": r["id"], "t_hat": t, "n_actions": n, "pred": {"error_mode": None}}) + "\n")
    print("DONE pointer", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "score"):
        s = sub.add_parser(name); s.add_argument("--traj", default="reports/alfworld/student3b_low100_train_b50.jsonl")
        s.add_argument("--student", default="Qwen/Qwen2.5-3B-Instruct"); s.add_argument("--cap", type=int, default=40); s.add_argument("--out", required=True)
        if name == "train":
            s.add_argument("--adapter", default=str(ROOT / "reports/alfworld/student3b_low100/final")); s.add_argument("--n", type=int, default=4000)
            s.add_argument("--lr", type=float, default=1e-4); s.add_argument("--batch", type=int, default=16)
        else:
            s.add_argument("--adapter", required=True); s.add_argument("--ids_from", default=None)
    p = sub.add_parser("pointer"); p.add_argument("--scores", required=True); p.add_argument("--out", required=True); p.add_argument("--delta", type=float, default=1.0)
    a = ap.parse_args(); {"train": cmd_train, "score": cmd_score, "pointer": cmd_pointer}[a.cmd](a)


if __name__ == "__main__":
    main()
