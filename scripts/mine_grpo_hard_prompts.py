"""Hard-example mining for GRPO: keep only prompts where the SFT policy's
sampled group actually has reward variance.

Why: the first GRPO run (SFT start, all 3,639 filtered train prompts)
logged frac_reward_zero_std = 0.85 -- 85% of prompt groups had identical
rewards across all 8 samples, so they contributed zero gradient (GRPO's
group-relative advantage is 0 when every sample scores the same). The SFT
model has memorized these training prompts (agent/step reward = 1.0,
mode = 0.91 on train vs 0.83 on val), so most groups are all-correct.

This script does ONE offline sampling pass with the SFT-merged model
(K completions per prompt, same temperature GRPO will use), scores each
completion with the same 3 reward functions, and writes out only prompts
whose group is informative (reward std > 0), plus per-prompt stats so the
yield and difficulty distribution can be inspected. GRPO then spends its
rollout budget on prompts that can actually teach it something.

Caveat recorded up front: mining against the SFT policy at t0 selects
prompts that are hard *for the SFT model right now*; as the policy improves
some become easy again. Fine for a first pass -- re-mining periodically is
the natural extension if this variant shows a clear gain.

Usage (on a GPU node):
    python scripts/mine_grpo_hard_prompts.py --k 8 --temperature 1.0 --batch_size 4
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

from error_attr.attribution.parse import parse_prediction  # noqa: E402

MODEL = ROOT / "reports" / "attributor_qwen2.5-3b-sft-merged"
IN_PATH = ROOT / "data" / "attributor_grpo" / "train.jsonl"
OUT_PATH = ROOT / "data" / "attributor_grpo" / "train_hard.jsonl"
STATS_PATH = ROOT / "data" / "attributor_grpo" / "mining_stats.jsonl"


@torch.no_grad()
def sample_group(model, tok, prompt: str, k: int, temperature: float, max_new_tokens: int = 64) -> list[str]:
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **inputs, do_sample=True, temperature=temperature, top_p=1.0, num_return_sequences=k,
        max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    new = out[:, inputs["input_ids"].shape[1]:]
    return tok.batch_decode(new, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()

    rows = [json.loads(l) for l in open(IN_PATH)]
    if args.limit:
        rows = rows[: args.limit]

    n_kept = 0
    with open(OUT_PATH, "w") as out, open(STATS_PATH, "w") as stats:
        for i, r in enumerate(rows):
            comps = sample_group(model, tok, r["prompt"], args.k, args.temperature)
            scores = [official_score(parse_prediction(c), r["ground_truth"], r["framework"]) for c in comps]
            rewards = [(s["agent"] + s["step"] + s["mode"]) / 3 for s in scores]
            mean = sum(rewards) / len(rewards)
            std = (sum((x - mean) ** 2 for x in rewards) / len(rewards)) ** 0.5
            mode_acc = sum(s["mode"] for s in scores) / len(scores)
            informative = std > 0
            stats.write(json.dumps({"id": r["id"], "framework": r["framework"], "reward_mean": mean,
                                    "reward_std": std, "mode_acc": mode_acc, "informative": informative}) + "\n")
            if informative:
                out.write(json.dumps(r) + "\n")
                n_kept += 1
            if (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(rows)}] kept={n_kept} ({n_kept/(i+1):.1%} informative)", flush=True)

    print(f"done: kept {n_kept}/{len(rows)} informative prompts -> {OUT_PATH}")


if __name__ == "__main__":
    main()
