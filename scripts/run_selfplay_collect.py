"""Collect SELF-GENERATED trajectories for the non-debate frameworks
(mathchat, magentic-one) on the shared task pool, same model as the
debaters (Qwen2.5-7B-Instruct), same outputs as run_selfplay_debate.py:

    data/selfplay/{fw}_trajectories.jsonl    (release + final_correct)
    data/selfplay/{fw}_release_rows.jsonl    (text.jsonl row format for the renderers)

Tasks run in lockstep batches so each agent turn is one generate() call.

Usage (GPU):
    python scripts/run_selfplay_collect.py --framework mathchat  --benchmarks math,scibench,mmlu_pro,gpqa
    python scripts/run_selfplay_collect.py --framework magentic-one --benchmarks math,scibench,mmlu_pro,gpqa
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from error_attr.selfplay.debate_runner import to_release_row  # noqa: E402

POOL = ROOT / "data" / "selfplay" / "task_pool.jsonl"
FW_SLUG = {"mathchat": "mathchat", "magentic-one": "magentic"}


def build_runner(framework, model, tok, args):
    if framework == "mathchat":
        from error_attr.selfplay.mathchat_runner import MathChatConfig, QwenMathChat
        return QwenMathChat(model, tok, MathChatConfig(max_assistant_turns=args.max_turns, max_new_tokens=args.max_new_tokens,
                                                        temperature=args.temperature))
    if framework == "magentic-one":
        from error_attr.selfplay.magentic_runner import MagenticConfig, QwenMagentic
        return QwenMagentic(model, tok, MagenticConfig(max_rounds=args.max_turns, max_new_tokens=args.max_new_tokens,
                                                        temperature=args.temperature))
    raise ValueError(framework)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", required=True, choices=list(FW_SLUG))
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max_turns", type=int, default=None, help="assistant turns (mathchat, default 5) / rounds (magentic, default 6)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=8, help="dialogs run in lockstep per generate()")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--benchmarks", default="math,scibench,mmlu_pro,gpqa",
                    help="self-contained reasoning tasks only: these frameworks have no web access, so gaia/simpleqa are out")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.max_turns is None:
        args.max_turns = 5 if args.framework == "mathchat" else 6
    slug = FW_SLUG[args.framework]
    out_path = ROOT / "data" / "selfplay" / f"{slug}_trajectories.jsonl"
    rows_path = ROOT / "data" / "selfplay" / f"{slug}_release_rows.jsonl"

    tasks = [json.loads(l) for l in open(POOL)]
    keep = set(args.benchmarks.split(","))
    tasks = [t for t in tasks if t["benchmark"] in keep]
    if args.limit:
        tasks = tasks[: args.limit]
    done = set()
    if args.resume and out_path.exists():
        done = {json.loads(l)["task_source_id"] for l in open(out_path)}
        tasks = [t for t in tasks if t["source_trace_id"] not in done]
    print(f"framework={args.framework} tasks to run: {len(tasks)} (skipping {len(done)}) -> {out_path}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto").eval()
    runner = build_runner(args.framework, model, tok, args)

    n_by, ok_by = defaultdict(int), defaultdict(int)
    t0 = time.time()
    mode = "a" if args.resume else "w"
    with open(out_path, mode) as out, open(rows_path, mode) as out_rows:
        for s in range(0, len(tasks), args.batch_size):
            chunk = tasks[s:s + args.batch_size]
            ids = [f"selfplay_{slug}_{t['benchmark']}_{s + k:04d}" for k, t in enumerate(chunk)]
            try:
                rels = runner.run_many(chunk, ids)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                rels = []
                for k in range(0, len(chunk), max(1, len(chunk) // 4)):  # fall back to smaller lockstep groups
                    rels += runner.run_many(chunk[k:k + max(1, len(chunk) // 4)], ids[k:k + max(1, len(chunk) // 4)])
            for t, rel in zip(chunk, rels):
                rel["task_source_id"] = t["source_trace_id"]
                out.write(json.dumps(rel) + "\n"); out_rows.write(json.dumps(to_release_row(rel)) + "\n")
                n_by[t["benchmark"]] += 1; ok_by[t["benchmark"]] += rel["final_correct"]
            out.flush(); out_rows.flush()
            acc = {b: f"{ok_by[b]}/{n_by[b]}" for b in n_by}
            print(f"[{min(s + args.batch_size, len(tasks))}/{len(tasks)}] {time.time() - t0:.0f}s acc={acc}", flush=True)

    print("done. accuracy by benchmark:")
    for b in n_by:
        print(f"  {b:18s} {ok_by[b]}/{n_by[b]} = {ok_by[b] / n_by[b]:.3f}")


if __name__ == "__main__":
    main()
