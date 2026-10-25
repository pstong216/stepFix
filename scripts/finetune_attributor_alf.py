"""Continue-train the 7B agent-text-only attributor on ALFWorld self-play labels (LoRA resumed from the existing adapter).
    python scripts/finetune_attributor_alf.py --init reports/attributor_7b_no_observations/final --data data/attributor_sft_alf/train.jsonl --out reports/attributor_7b_alf
"""
import argparse, json, os, random
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import PeftModel, PeftConfig


class DS(Dataset):
    def __init__(self, rows, tok, max_len=8192):
        self.items = []
        for r in rows:
            p = tok(tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
            t = tok(r["target"], add_special_tokens=False).input_ids + [tok.eos_token_id]
            if len(p) + len(t) > max_len: continue
            self.items.append((p + t, [-100] * len(p) + t))
    def __len__(self): return len(self.items)
    def __getitem__(self, i): ids, lab = self.items[i]; return {"input_ids": ids, "labels": lab}


def collate(tok):
    def fn(b):
        m = max(len(x["input_ids"]) for x in b); pad = tok.pad_token_id
        return {"input_ids": torch.tensor([x["input_ids"] + [pad] * (m - len(x["input_ids"])) for x in b]),
                "labels": torch.tensor([x["labels"] + [-100] * (m - len(x["labels"])) for x in b]),
                "attention_mask": torch.tensor([[1] * len(x["input_ids"]) + [0] * (m - len(x["input_ids"])) for x in b])}
    return fn


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--init", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0); ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--grad_accum", type=int, default=16)
    a = ap.parse_args()
    cfg = PeftConfig.from_pretrained(a.init); tok = AutoTokenizer.from_pretrained(a.init); tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path, dtype=torch.bfloat16, device_map={"": 0})
    base.gradient_checkpointing_enable(); base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, a.init, is_trainable=True); model.print_trainable_parameters()
    rows = [json.loads(l) for l in open(a.data)]; random.Random(0).shuffle(rows); ds = DS(rows, tok); print(f"{len(ds)} examples (of {len(rows)} rows)", flush=True)
    args = TrainingArguments(output_dir=a.out, per_device_train_batch_size=1, gradient_accumulation_steps=a.grad_accum, num_train_epochs=a.epochs, learning_rate=a.lr,
                             lr_scheduler_type="cosine", warmup_steps=5, logging_steps=10, save_strategy="no", bf16=True, report_to=[], remove_unused_columns=False)
    Trainer(model=model, args=args, train_dataset=ds, data_collator=collate(tok)).train()
    model.save_pretrained(os.path.join(a.out, "final")); tok.save_pretrained(os.path.join(a.out, "final")); print("SAVED", os.path.join(a.out, "final"), flush=True)


if __name__ == "__main__":
    main()
