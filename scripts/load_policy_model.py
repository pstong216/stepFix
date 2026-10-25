"""Load the Qwen2.5-7B-Instruct policy across the allocated GPUs and run a
minimal sanity check (a few forward passes) before it's wired into rollout
collection. Not a training script -- just confirms the model loads and
`LLMPolicy.score_admissible_commands` produces sane output.
"""
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from error_attr.policy.alfworld_agent import ALFWorldHistory, InteractionStep, LLMPolicy  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def main():
    print(f"CUDA available: {torch.cuda.is_available()}, device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    policy = LLMPolicy(model, tokenizer)

    history = ALFWorldHistory(goal="put a clean tomato in microwave")
    history.steps.append(
        InteractionStep(
            observation="You are in the middle of a kitchen. You see a countertop, a fridge, a microwave.",
            admissible_commands=[
                "go to fridge 1",
                "go to countertop 1",
                "go to microwave 1",
                "inventory",
                "look",
            ],
        )
    )

    logprobs = policy.score_admissible_commands(history)
    for cmd, lp in sorted(logprobs.items(), key=lambda x: -x[1]):
        print(f"{lp:8.3f}  {cmd}")

    chosen = policy.act(history)
    print(f"\nChosen action: {chosen!r}")
    print("MODEL LOAD + SCORING SANITY CHECK PASSED")


if __name__ == "__main__":
    main()
