"""Evaluate a trained (or base, for a before/after comparison) attributor
against the official whowhen_eval scorer, on val.jsonl and
transfer_alfworld.jsonl separately -- the latter is the RQ3 transfer
check (Research Plan Sec. 19/23): a benchmark never seen during training.

Usage:
    python scripts/eval_attributor.py --model reports/attributor_qwen2.5-3b-lora/final --split val
    python scripts/eval_attributor.py --model reports/attributor_qwen2.5-3b-lora/final --split transfer_alfworld
    python scripts/eval_attributor.py --model Qwen/Qwen2.5-3B-Instruct --split val   # base-model baseline
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from whowhen_eval.score import score as official_score  # noqa: E402

from error_attr.attribution.parse import parse_prediction  # noqa: E402  (shared with GRPO reward)

DATA_DIR = ROOT / "data" / "attributor_sft"


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local checkpoint path")
    ap.add_argument("--split", required=True, choices=["val", "transfer_alfworld"])
    ap.add_argument("--limit", type=int, default=None, help="cap number of examples (for a quick check)")
    ap.add_argument("--out", default=None, help="optional path to dump per-example predictions")
    ap.add_argument("--data_dir", default=str(DATA_DIR),
                     help="dir holding {split}.jsonl; point at data/attributor_sft_<variant> to evaluate a "
                          "variant-trained model on its own ablated prompts (TODO item 3)")
    ap.add_argument("--max_new_tokens", type=int, default=64,
                     help="64 fits the SFT model's ~18-token answers; use ~160 for base-model-derived (GRPO-from-base) "
                          "policies, which add a Reason line and ramble inside fields")
    args = ap.parse_args()

    model_path = Path(args.model)
    is_lora_adapter = (model_path / "adapter_config.json").is_file()

    if is_lora_adapter:
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, args.model)
        model = model.merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    rows = [json.loads(line) for line in open(Path(args.data_dir) / f"{args.split}.jsonl")]
    if args.limit:
        rows = rows[: args.limit]

    n = 0
    agent_correct = step_correct = mode_correct = 0
    all_three_correct = 0
    records = []

    for row in rows:
        raw_pred = generate(model, tokenizer, row["prompt"], max_new_tokens=args.max_new_tokens)
        pred = parse_prediction(raw_pred)
        # Use the RAW ground_truth (round/position/agents, framework-native
        # schema), not a synthetic {"step": gold_step} reconstruction --
        # whowhen_eval.score's per-framework scorers for mathchat
        # (2*round+position) and debate/dylan (round-only) read gt.round/
        # gt.position directly, and silently produce an empty accept-set
        # (i.e. every prediction scores wrong) if those keys are missing.
        # Bug found 2026-09-08 via a suspicious exact 0.000 step accuracy
        # on exactly those 3 frameworks; see DECISIONS.md.
        gt = row["ground_truth"]
        result = official_score(pred, gt, row["framework"])

        n += 1
        agent_correct += result["agent"]
        step_correct += result["step"]
        mode_correct += result["mode"]
        all_three_correct += all(result.values())

        records.append({"id": row["id"], "framework": row["framework"], "raw_pred": raw_pred,
                         "parsed": pred, "gt": gt, "result": result})

        if n % 25 == 0:
            print(f"[{n}/{len(rows)}] agent={agent_correct/n:.3f} step={step_correct/n:.3f} "
                  f"mode={mode_correct/n:.3f} all3={all_three_correct/n:.3f}")

    print()
    print(f"=== {args.model} on {args.split} (n={n}) ===")
    print(f"agent accuracy: {agent_correct/n:.3f}")
    print(f"step  accuracy: {step_correct/n:.3f}")
    print(f"mode  accuracy: {mode_correct/n:.3f}")
    print(f"all-3 accuracy: {all_three_correct/n:.3f}")

    if args.out:
        with open(args.out, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"per-example predictions written to {args.out}")


if __name__ == "__main__":
    main()
