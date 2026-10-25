"""Generate corrected answers under each of the 4 feedback conditions and
grade them against gold, for the feedback-granularity collaboration study
(docs/feedback_granularity_study.md).

Grading note: debate/dylan/mathchat expect `\\boxed{X}`; magentic-one
expects `FINAL ANSWER: X` (per the task's own instructions -- see
prepare_feedback_study_data.py output). Parsing is intentionally simple
(exact-match after normalization) rather than an LLM-judge: the same
parser is applied uniformly across all 4 conditions, so any parsing noise
is symmetric across conditions and shouldn't bias the CROSS-CONDITION
comparison, even if it adds some noise to the absolute recovery-rate
numbers.

Usage:
    python scripts/run_feedback_study.py --model Qwen/Qwen2.5-7B-Instruct --limit 200
    python scripts/run_feedback_study.py --model Qwen/Qwen2.5-7B-Instruct   # full run
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "feedback_study" / "prompts.jsonl"

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER:\s*(.+)", re.IGNORECASE)


def _last_boxed(text: str) -> str | None:
    """Brace-matched content of the last \\boxed{...}. `_BOXED_RE` stopped at the
    first '}' and truncated nested answers (\\boxed{\\frac{1}{64}} -> '\\frac{1');
    fixed 2026-09-09 (DECISIONS #25). Re-grading the stored raw responses of
    the finished study with this parser moved every condition up ~1pp and
    left all four McNemar conclusions unchanged."""
    start = text.rfind("\\boxed{")
    while start != -1:
        i, depth = start + 7, 1
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if depth == 0:
            return text[start + 7: i - 1].strip()
        start = text.rfind("\\boxed{", 0, start)
    return None


def parse_answer(framework: str, raw_text: str) -> str | None:
    if framework in ("debate", "dylan", "mathchat"):
        return _last_boxed(raw_text)
    if framework == "magentic-one":
        matches = _FINAL_ANSWER_RE.findall(raw_text)
        if not matches:
            return None
        return matches[-1].strip().split("\n")[0].strip()
    raise ValueError(f"no parser registered for framework={framework!r}")


sys.path.insert(0, str(ROOT / "src"))
from error_attr.selfplay.debate_runner import normalize, is_correct as _shared_is_correct  # noqa: E402


def is_correct(framework: str, parsed: str | None, gold: str) -> bool:
    """Shared grader (exact after normalization OR numeric within 1%); see DECISIONS #26.
    Re-grading this study's stored raw responses with it: every condition +~11pp,
    all four McNemar conclusions unchanged."""
    return _shared_is_correct(framework, parsed, gold)


@torch.no_grad()
def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int = 512) -> list[str]:
    """Batched greedy generation: left-pads variable-length prompts so all
    sequences finish generating new tokens at the same right-aligned
    position. Left-padding (not right) is required for causal-LM batched
    generation -- otherwise position ids for the padded-shorter sequences
    would be wrong relative to their real content."""
    tokenizer.padding_side = "left"
    texts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    inputs = tokenizer(texts, return_tensors="pt", add_special_tokens=False, padding=True).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    new_tokens = out[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def generate_batch_oom_safe(model, tokenizer, prompts: list[str], max_new_tokens: int = 512) -> list[str]:
    """generate_batch with a halving fallback on CUDA OOM. The dataset's
    prompt lengths vary widely (median ~3k tokens, p99 ~14k+), so a batch
    size safe for the common case can still OOM on a batch that happens to
    contain several long transcripts. Rather than let one bad batch crash
    a multi-hour run, retry at half the batch size (down to 1) after
    clearing the CUDA cache."""
    if len(prompts) == 1:
        try:
            return generate_batch(model, tokenizer, prompts, max_new_tokens)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return ["[SKIPPED: CUDA OOM even at batch size 1]"]

    try:
        return generate_batch(model, tokenizer, prompts, max_new_tokens)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        mid = len(prompts) // 2
        return (
            generate_batch_oom_safe(model, tokenizer, prompts[:mid], max_new_tokens)
            + generate_batch_oom_safe(model, tokenizer, prompts[mid:], max_new_tokens)
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=None, help="cap total (trace,condition) rows processed")
    ap.add_argument("--max_traces_per_framework", type=int, default=None,
                     help="stratified subsample: keep at most N traces per framework (all 4 conditions kept per trace)")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--prompts_path", default=str(DATA_PATH),
                     help="override the input prompts jsonl (e.g. prompts_predicted.jsonl for the Model-A-predictions variant)")
    ap.add_argument("--out", default=str(ROOT / "reports" / "feedback_study_results.jsonl"))
    ap.add_argument("--resume", action="store_true", help="append to --out, skipping (id, condition) pairs already present")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    rows = [json.loads(line) for line in open(args.prompts_path)]

    if args.max_traces_per_framework:
        seen_per_framework = defaultdict(set)
        kept_ids = set()
        for row in rows:
            fw, tid = row["framework"], row["id"]
            if tid in kept_ids or len(seen_per_framework[fw]) < args.max_traces_per_framework:
                seen_per_framework[fw].add(tid)
                kept_ids.add(tid)
        rows = [r for r in rows if r["id"] in kept_ids]

    if args.limit:
        rows = rows[: args.limit]

    # Sort by prompt length before batching: batches of similar-length
    # prompts waste far less padding (and are far less likely to OOM) than
    # batches drawn from a dataset whose prompt lengths vary ~5x. Order
    # doesn't matter for the aggregate analysis -- each record keeps its
    # own id/framework/condition.
    rows.sort(key=lambda r: len(r["prompt"]))

    n_by_condition = defaultdict(int)
    correct_by_condition = defaultdict(int)
    n_by_cell = defaultdict(int)
    correct_by_cell = defaultdict(int)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    done = set()
    if args.resume and Path(args.out).exists():
        done = {(json.loads(l)["id"], json.loads(l)["condition"]) for l in open(args.out)}
        rows = [r for r in rows if (r["id"], r["condition"]) not in done]
        print(f"--resume: skipping {len(done)} finished rows, {len(rows)} to go", flush=True)
    with open(args.out, "a" if args.resume else "w") as out_f:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            raw_responses = generate_batch_oom_safe(model, tokenizer, [r["prompt"] for r in batch])

            for row, raw in zip(batch, raw_responses):
                parsed = parse_answer(row["framework"], raw)
                correct = is_correct(row["framework"], parsed, row["gold_answer"])

                cond = row["condition"]
                cell = (row["framework"], cond)
                n_by_condition[cond] += 1
                correct_by_condition[cond] += correct
                n_by_cell[cell] += 1
                correct_by_cell[cell] += correct

                out_f.write(json.dumps({
                    "id": row["id"], "framework": row["framework"], "condition": cond,
                    "gold_answer": row["gold_answer"], "parsed_answer": parsed,
                    "correct": correct, "raw_response": raw,
                }) + "\n")
            out_f.flush()
            n_done += len(batch)
            print(f"[{n_done}/{len(rows)}]", {c: f"{correct_by_condition[c]}/{n_by_condition[c]}" for c in n_by_condition})

    print()
    print("=== Recovery rate by condition ===")
    for cond in ("reason_only", "vague_reason", "precise_reason", "precise_only"):
        n = n_by_condition[cond]
        if n:
            print(f"{cond:16s}: {correct_by_condition[cond]}/{n} = {correct_by_condition[cond]/n:.3f}")

    print()
    print("=== Recovery rate by (framework, condition) ===")
    for (fw, cond), n in sorted(n_by_cell.items()):
        print(f"{fw:14s} {cond:16s}: {correct_by_cell[(fw, cond)]}/{n} = {correct_by_cell[(fw, cond)]/n:.3f}")


if __name__ == "__main__":
    main()
