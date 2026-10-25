"""GRPO (RL with verifiable rewards) fine-tuning of the attributor, on top of
the SFT checkpoint. See docs/feedback_granularity_study.md ("RL (GRPO)
training") and DECISIONS.md for the rationale.

Setup
-----
- Policy / start point: the SFT LoRA merged into Qwen2.5-3B-Instruct
  (reports/attributor_qwen2.5-3b-sft-merged). A FRESH LoRA adapter is
  trained on top of it. With PEFT, trl uses the same base with adapters
  disabled as the frozen KL reference -- i.e. the reference policy IS the
  SFT model, which is exactly the anchor we want (beta > 0 keeps the
  policy from drifting into reward-hacking outputs).
- Reward: fully rule-based / verifiable. Each sampled completion is parsed
  with the SAME parser used by eval (error_attr.attribution.parse) and
  graded with the official Who&When Pro scorer (`whowhen_eval.score`)
  against the trace's RAW ground_truth. Three separate 0/1 reward
  functions (agent / step / mode), equally weighted -- trl logs each
  function's mean separately, giving per-axis accuracy curves during
  training for free (mode is the weak axis after SFT: 82.8%).
- Data: data/attributor_grpo/train.jsonl = SFT train prompts filtered to
  <= 4096 prompt tokens (3,639 of 5,353) so G generations per prompt fit
  on one 46GB L40S. Conversational prompt format so the chat template is
  applied identically to SFT/eval.

Launch (inside tmux, on the GPU node):
    python scripts/train_attributor_grpo.py
"""
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from whowhen_eval.score import score as official_score  # noqa: E402

from error_attr.attribution.parse import parse_prediction  # noqa: E402

BASE_MODEL = str(ROOT / "reports" / "attributor_qwen2.5-3b-sft-merged")
DATA_PATH = ROOT / "data" / "attributor_grpo" / "train.jsonl"
OUTPUT_DIR = ROOT / "reports" / "attributor_qwen2.5-3b-grpo"


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

def _completion_text(completion) -> str:
    # Conversational format: completion is [{"role": "assistant", "content": ...}]
    if isinstance(completion, list):
        return "".join(m.get("content", "") for m in completion if isinstance(m, dict))
    return str(completion)


def _score_all(completions, framework, ground_truth_json):
    out = []
    for c, fw, gt_json in zip(completions, framework, ground_truth_json, strict=True):
        pred = parse_prediction(_completion_text(c))
        out.append(official_score(pred, json.loads(gt_json), fw))
    return out


def reward_agent(completions, framework, ground_truth_json, **kwargs):
    return [float(s["agent"]) for s in _score_all(completions, framework, ground_truth_json)]


def reward_step(completions, framework, ground_truth_json, **kwargs):
    return [float(s["step"]) for s in _score_all(completions, framework, ground_truth_json)]


def reward_mode(completions, framework, ground_truth_json, **kwargs):
    return [float(s["mode"]) for s in _score_all(completions, framework, ground_truth_json)]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_dataset(data_path: Path = DATA_PATH) -> Dataset:
    rows = []
    with open(data_path) as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "prompt": [{"role": "user", "content": r["prompt"]}],
                "framework": r["framework"],
                # JSON string, not a dict: ground_truth schemas differ per
                # framework (round/position vs step vs agents list), and HF
                # datasets would otherwise force them into one struct.
                "ground_truth_json": json.dumps(r["ground_truth"]),
                "id": r["id"],
            })
    return Dataset.from_list(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_PATH), help="prompts jsonl (e.g. train_hard.jsonl from hard mining)")
    ap.add_argument("--output_dir", default=str(OUTPUT_DIR))
    ap.add_argument("--model", default=BASE_MODEL, help="start point; the SFT-merged model by default")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--max_completion_length", type=int, default=64,
                     help="64 suits the SFT model (~18-token answers); the un-fine-tuned base model rambles "
                          "inside fields and adds a Reason line, so it needs ~160 to keep Error Mode uncut")
    ap.add_argument("--save_steps", type=int, default=50,
                     help="checkpoint cadence; use 25 on any allocation that could be preempted/killed "
                          "(arm 3's first run was preempted at step 40 with save_steps=50 -> nothing saved)")
    ap.add_argument("--mask_truncated_completions", action="store_true",
                     help="exclude completions that hit the length cap from the loss (trl option). Contingency for "
                          "the base arm if outputs lengthen and the Error Mode line starts getting truncated.")
    cli = ap.parse_args()
    output_dir = Path(cli.output_dir)

    train_ds = load_dataset(Path(cli.data))
    print(f"GRPO train prompts: {len(train_ds)} (from {cli.data}) | start model: {cli.model}")

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    args = GRPOConfig(
        output_dir=str(output_dir),
        model_init_kwargs={"dtype": torch.bfloat16},
        # Rollout: 8 samples per prompt; batch of 8 = 1 prompt per micro-step,
        # 4 micro-steps accumulated = 4 prompts x 8 generations per update.
        num_generations=8,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        max_completion_length=cli.max_completion_length,
        mask_truncated_completions=cli.mask_truncated_completions,
        temperature=cli.temperature,
        # Optimization
        learning_rate=cli.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        max_steps=cli.max_steps,        # first run: 200 (~800 prompts) to validate the signal before scaling
        beta=cli.beta,                  # KL anchor to the SFT reference (trl default is 0.0)
        bf16=True,
        gradient_checkpointing=True,
        # Logging / saving
        logging_steps=5,
        save_steps=cli.save_steps,
        save_total_limit=3,
        report_to="none",
        log_completions=True,
        seed=0,
    )

    trainer = GRPOTrainer(
        model=cli.model,
        reward_funcs=[reward_agent, reward_step, reward_mode],
        args=args,
        train_dataset=train_ds,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    trainer.processing_class.save_pretrained(str(output_dir / "final"))
    print("saved GRPO adapter to", output_dir / "final")


if __name__ == "__main__":
    main()
