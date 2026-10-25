"""LoRA SFT training for the specialized failure attributor (Research Plan
Sec. 9.C). Input: the identical `all_at_once` judge prompt (Sec. 9.B) built
by sft_format.py. Target: a 3-line (Agent Name / Step Number / Error Mode)
completion. Loss is computed only on the target tokens (prompt tokens
masked to -100).

Launch with 2 GPUs via accelerate/torchrun for DDP:
    accelerate launch --num_processes 2 scripts/train_attributor.py
Single GPU also works: python scripts/train_attributor.py
"""
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_SEQ_LEN = 8192
OUTPUT_DIR = ROOT / "reports" / "attributor_qwen2.5-3b-lora"
DATA_DIR = ROOT / "data" / "attributor_sft"


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]


class AttributorDataset(Dataset):
    """Tokenizes (prompt, target) pairs, masking prompt tokens in labels.
    Examples whose combined length exceeds MAX_SEQ_LEN are dropped rather
    than truncated -- truncating would risk silently cutting off the
    Response Format section or the gold target itself."""

    def __init__(self, path: Path, tokenizer, max_seq_len: int = MAX_SEQ_LEN):
        self.examples: list[Example] = []
        n_dropped = 0
        with open(path) as f:
            rows = [json.loads(line) for line in f]

        for row in rows:
            messages = [{"role": "user", "content": row["prompt"]}]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            target_ids = tokenizer(row["target"], add_special_tokens=False).input_ids
            eos = [tokenizer.eos_token_id]

            full_ids = prompt_ids + target_ids + eos
            if len(full_ids) > max_seq_len:
                n_dropped += 1
                continue

            labels = [-100] * len(prompt_ids) + target_ids + eos
            self.examples.append(Example(input_ids=full_ids, labels=labels))

        print(f"{path.name}: {len(self.examples)} kept, {n_dropped} dropped (> {max_seq_len} tokens)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {"input_ids": ex.input_ids, "labels": ex.labels}


def make_collate_fn(pad_token_id: int):
    def collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attention_mask = [], [], []
        for b in batch:
            pad_len = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_token_id] * pad_len)
            labels.append(b["labels"] + [-100] * pad_len)
            attention_mask.append([1] * len(b["input_ids"]) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
    return collate


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DATA_DIR), help="SFT data dir (e.g. a --variant build for TODO item 3)")
    ap.add_argument("--output_dir", default=str(OUTPUT_DIR))
    ap.add_argument("--model", default=MODEL_NAME, help="base model (default Qwen2.5-3B-Instruct; e.g. Qwen/Qwen2.5-7B-Instruct for the scale test)")
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--keep_ids", default=None,
                     help="JSON list of train ids to keep (e.g. data/attributor_sft/kept_ids_full_8192.json, the 4,767 rows "
                          "the full-trace SFT actually trained on) -- controls the extra-data confound when comparing "
                          "compressed-input retrains (TODO item 3)")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--save_each_epoch", action="store_true", help="keep a checkpoint at the end of every epoch")
    ap.add_argument("--resume", action="store_true", help="resume from the latest checkpoint in output_dir if one exists")
    cli = ap.parse_args()
    data_dir, output_dir = Path(cli.data_dir), Path(cli.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(cli.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cli.model,
        torch_dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = AttributorDataset(data_dir / "train.jsonl", tokenizer)
    if cli.keep_ids:
        keep = set(json.load(open(cli.keep_ids)))
        rows = [json.loads(l) for l in open(data_dir / "train.jsonl")]
        ids_in_order = [r["id"] for r in rows]
        # AttributorDataset drops over-length rows silently; rebuild with the id filter applied first.
        tmp = data_dir / "train.keep_filtered.jsonl"
        with open(tmp, "w") as f:
            for r in rows:
                if r["id"] in keep:
                    f.write(json.dumps(r) + "\n")
        train_ds = AttributorDataset(tmp, tokenizer)
        print(f"--keep_ids: {len(keep)} ids requested, {len(train_ds)} rows kept after length filter")
    val_ds = AttributorDataset(data_dir / "val.jsonl", tokenizer)

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cli.grad_accum,
        num_train_epochs=cli.epochs,
        learning_rate=1e-4,
        warmup_steps=30,  # ~3% of ~900 total optimizer steps (4767 train / (2*8) * 3 epochs)
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch" if cli.save_each_epoch else "steps",
        eval_steps=100,
        save_strategy="epoch" if cli.save_each_epoch else "steps",
        save_steps=100,
        save_total_limit=None if cli.save_each_epoch else 2,
        bf16=True,
        report_to=[],
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=make_collate_fn(tokenizer.pad_token_id),
    )
    import glob
    trainer.train(resume_from_checkpoint=bool(cli.resume and glob.glob(str(output_dir / 'checkpoint-*'))))
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))


if __name__ == "__main__":
    main()
