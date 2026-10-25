"""Self-trained log-reading error attributor (3B, from the student's own weights, labels from attr_labels_env.py).
  train   -- LoRA (continued from --init adapter: the student, or the previous round's attributor) on {failed log -> "Step: t*"}.
             --labels may be given several times (accumulate rounds); each label file's trajectories come from --trajs in the same order.
  pointer -- read new failed logs, output the step; writes steps.jsonl (pointer schema) and raw predictions.
Prompt = SELFATTR_PROMPT of self_evolve.py (compact one-line-per-step log, observations truncated to 160 chars)."""
import argparse, json, random, re, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from alfworld_attr_opd import load_jsonl
from self_evolve import SELFATTR_PROMPT


# one type vocabulary across label sources (synthetic injection names -> the generic names used by env-evidence / divergence labels)
UNIFY = {"wrong_location": "wrong_target", "wrong_object": "wrong_target", "misplace": "wrong_value", "random": "wrong_op", "misorder": "misorder", "skip": "skip"}


def log_text(e): return "\n".join(f"Step {t}: action: {s['action']} | observation: {s['next_obs'][:160]}" for t, s in enumerate(e["steps"]))
def user_msg(e): return SELFATTR_PROMPT.format(goal=e["goal"], log=log_text(e))


def load(model_name, adapter, trainable):
    tok = AutoTokenizer.from_pretrained(model_name); tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map={"": 0})
    if trainable: base.gradient_checkpointing_enable(); base.enable_input_require_grads()
    m = PeftModel.from_pretrained(base, adapter, is_trainable=trainable)
    return tok, m


def cmd_train(a):
    rng = random.Random(0); items = []
    for lab, traj in zip(a.labels, a.trajs):
        eps = {e["id"]: e for e in load_jsonl(ROOT / traj)}
        items += [(eps[r["id"]], r["t_star"], UNIFY.get(r.get("type"), r.get("type"))) for r in load_jsonl(ROOT / lab) if r["id"] in eps]
    rng.shuffle(items); print(f"{len(items)} labelled failures", flush=True)
    tok, model = load(a.model, str(ROOT / a.init), True); model.train()
    params = [p for p in model.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True); t0 = time.time(); run = 0.0; n = 0
    for ep in range(a.epochs):
        rng.shuffle(items)
        for e, t, typ in items:
            msg = tok.apply_chat_template([{"role": "user", "content": user_msg(e)}], tokenize=False, add_generation_prompt=True)
            p_ids = tok(msg, add_special_tokens=False).input_ids[-a.max_len:]; t_ids = tok(f"step {t}" + (f", type {typ}" if typ else ""), add_special_tokens=False).input_ids + [tok.eos_token_id]
            ids = torch.tensor([p_ids + t_ids], device=model.device); lab = torch.tensor([[-100] * len(p_ids) + t_ids], device=model.device)
            loss = model(input_ids=ids, labels=lab).loss; (loss / a.batch).backward(); run += loss.item(); n += 1
            if n % a.batch == 0: torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
            if n % 50 == 0: print(f"[{n}/{len(items) * a.epochs}] loss {run / 50:.3f} {time.time() - t0:.0f}s", flush=True); run = 0.0
    model.save_pretrained(str(out / "adapter")); tok.save_pretrained(str(out / "adapter")); print("DONE train", flush=True)


@torch.no_grad()
def cmd_pointer(a):
    eps = {e["id"]: e for e in load_jsonl(ROOT / a.traj)}; ids = [r["id"] for r in load_jsonl(ROOT / a.ids)]
    tok, model = load(a.model, str(ROOT / a.adapter), False); model.eval()
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True); raw = out.with_suffix(".raw.jsonl")
    with open(out, "w") as f, open(raw, "w") as g:
        for k, i in enumerate(ids):
            e = eps[i]; msg = tok.apply_chat_template([{"role": "user", "content": user_msg(e)}], tokenize=False, add_generation_prompt=True)
            x = tok(msg, return_tensors="pt", add_special_tokens=False).to(model.device)
            o = tok.decode(model.generate(**x, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)[0, x.input_ids.shape[1]:], skip_special_tokens=True)
            m = re.search(r"\d+", o); t = min(int(m.group()), len(e["steps"]) - 1) if m else 0
            mt = re.search(r"type\s*[:=]?\s*([a-z_]+)", o); typ = mt.group(1) if mt else None
            f.write(json.dumps({"id": i, "steps": [t], "mode": None}) + "\n"); g.write(json.dumps({"id": i, "t_hat": t, "type": typ, "raw": o}) + "\n")
            if k % 50 == 0: print(f"[{k + 1}/{len(ids)}]", flush=True)
    print("DONE pointer", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--labels", nargs="+", required=True); t.add_argument("--trajs", nargs="+", required=True)
    t.add_argument("--init", default="reports/alfworld/student3b_low100/final"); t.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=2); t.add_argument("--lr", type=float, default=1e-4); t.add_argument("--batch", type=int, default=8); t.add_argument("--max_len", type=int, default=6000)
    p = sub.add_parser("pointer"); p.add_argument("--traj", required=True); p.add_argument("--ids", required=True); p.add_argument("--adapter", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); p.add_argument("--out", required=True)
    a = ap.parse_args(); {"train": cmd_train, "pointer": cmd_pointer}[a.cmd](a)


if __name__ == "__main__":
    main()
