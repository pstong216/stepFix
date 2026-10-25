"""Warm-start SFT of the ALFWorld student on expert trajectories: prompt = ReAct history (same render as the policy), target = action.
    python scripts/alfworld_sft.py --traj reports/alfworld/expert_train.jsonl --model Qwen/Qwen2.5-3B-Instruct --out reports/alfworld/student3b_sft
"""
import argparse, json, os, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("ERROR_ATTR_ROOT", str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from error_attr.policy.alfworld_agent import ALFWorldHistory, InteractionStep, SYSTEM_PROMPT


def examples(traj_path, history=10, only_won=True, max_games=0):
    eps = [json.loads(l) for l in open(traj_path)]
    if only_won: eps = [e for e in eps if e["won"]]
    if max_games: eps = eps[:max_games]
    out = []
    for e in eps:
        h = ALFWorldHistory(goal=e["goal"])
        for st in e["steps"]:
            h.steps.append(InteractionStep(observation=st["obs"], admissible_commands=st["admissible"]))
            out.append((h.render_prompt(history), st["action"]))
            h.steps[-1].action = st["action"]
    return out


class DS(Dataset):
    def __init__(self, pairs, tok, max_len=1536):
        self.items = []
        for user, action in pairs:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
            p_ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
            t_ids = tok(action, add_special_tokens=False).input_ids + [tok.eos_token_id]
            ids = (p_ids + t_ids)[-max_len:]; labels = ([-100] * len(p_ids) + t_ids)[-max_len:]
            self.items.append((ids, labels))
    def __len__(self): return len(self.items)
    def __getitem__(self, i): ids, labels = self.items[i]; return {"input_ids": ids, "labels": labels}


def collate(tok):
    def fn(batch):
        m = max(len(b["input_ids"]) for b in batch); pad = tok.pad_token_id
        ids = torch.tensor([b["input_ids"] + [pad] * (m - len(b["input_ids"])) for b in batch])
        lab = torch.tensor([b["labels"] + [-100] * (m - len(b["labels"])) for b in batch])
        att = torch.tensor([[1] * len(b["input_ids"]) + [0] * (m - len(b["input_ids"])) for b in batch])
        return {"input_ids": ids, "labels": lab, "attention_mask": att}
    return fn


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--traj", required=True); ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_games", type=int, default=0); ap.add_argument("--lora_r", type=int, default=32); ap.add_argument("--history", type=int, default=10)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model); tok.pad_token = tok.pad_token or tok.eos_token
    pairs = examples(a.traj, a.history, max_games=a.max_games); random.Random(0).shuffle(pairs)
    print(f"{len(pairs)} (state, action) pairs", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    ds = DS(pairs, tok)
    args = TrainingArguments(output_dir=a.out, per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.grad_accum, num_train_epochs=a.epochs, learning_rate=a.lr,
                             lr_scheduler_type="cosine", warmup_steps=20, logging_steps=20, save_strategy="no", bf16=True, report_to=[], remove_unused_columns=False)
    Trainer(model=model, args=args, train_dataset=ds, data_collator=collate(tok)).train()
    model.save_pretrained(os.path.join(a.out, "final")); tok.save_pretrained(os.path.join(a.out, "final")); print("SAVED", os.path.join(a.out, "final"), flush=True)


if __name__ == "__main__":
    main()
