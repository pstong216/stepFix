"""Collect SELF-GENERATED multi-agent trajectories: Qwen2.5-7B-Instruct
debaters (the same model that later receives feedback as Model B) on the
task pool shared with the external-transcript study
(data/selfplay/task_pool.jsonl, 862 tasks).

Output: data/selfplay/debate_trajectories.jsonl -- one Who&When Pro
`debate`-schema release per task, plus `final_correct`. The FAILED ones
(final_correct == False) are route A's input: natural failures of the very
system that will be asked to correct them. The SUCCESSFUL ones are route
B's input (error injection into Qwen's own successful runs).

Also writes data/selfplay/debate_release_rows.jsonl in text.jsonl row
format so whowhen_eval.run.release_from_row / the renderers accept them
unchanged.

Usage (GPU):
    python scripts/run_selfplay_debate.py --n_agents 3 --n_rounds 2 [--limit N] [--benchmarks math,gpqa]
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

from error_attr.selfplay.debate_runner import DebateConfig, QwenDebate, to_release_row  # noqa: E402

POOL = ROOT / "data" / "selfplay" / "task_pool.jsonl"
OUT = ROOT / "data" / "selfplay" / "debate_trajectories.jsonl"
OUT_ROWS = ROOT / "data" / "selfplay" / "debate_release_rows.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n_agents", type=int, default=3)
    ap.add_argument("--n_rounds", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--benchmarks", default=None, help="comma-separated subset, e.g. math,gpqa")
    ap.add_argument("--resume", action="store_true", help="skip tasks already in the output file")
    ap.add_argument("--tag", default="debate", help="output prefix: data/selfplay/<tag>_trajectories.jsonl (e.g. debate3b for a 3B student, OPD)")
    args = ap.parse_args()
    global OUT, OUT_ROWS
    OUT = ROOT / "data" / "selfplay" / f"{args.tag}_trajectories.jsonl"
    OUT_ROWS = ROOT / "data" / "selfplay" / f"{args.tag}_release_rows.jsonl"

    tasks = [json.loads(l) for l in open(POOL)]
    if args.benchmarks:
        keep = set(args.benchmarks.split(","))
        tasks = [t for t in tasks if t["benchmark"] in keep]
    if args.limit:
        tasks = tasks[: args.limit]

    done = set()
    if args.resume and OUT.exists():
        done = {json.loads(l)["task_source_id"] for l in open(OUT)}
        tasks = [t for t in tasks if t["source_trace_id"] not in done]
    print(f"tasks to run: {len(tasks)} (skipping {len(done)} already done)", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto").eval()
    debate = QwenDebate(model, tok, DebateConfig(args.n_agents, args.n_rounds, args.max_new_tokens, args.temperature))

    n_by_bench, correct_by_bench = defaultdict(int), defaultdict(int)
    t0 = time.time()
    mode = "a" if args.resume else "w"
    with open(OUT, mode) as out, open(OUT_ROWS, mode) as out_rows:
        for i, task in enumerate(tasks):
            trace_id = f"selfplay_{args.tag}_{task['benchmark']}_{i:04d}"
            rel = debate.run(task, trace_id)
            rel["task_source_id"] = task["source_trace_id"]
            out.write(json.dumps(rel) + "\n")
            out_rows.write(json.dumps(to_release_row(rel)) + "\n")
            out.flush(); out_rows.flush()
            n_by_bench[task["benchmark"]] += 1
            correct_by_bench[task["benchmark"]] += rel["final_correct"]
            if (i + 1) % 10 == 0:
                acc = {b: f"{correct_by_bench[b]}/{n_by_bench[b]}" for b in n_by_bench}
                print(f"[{i+1}/{len(tasks)}] {time.time()-t0:.0f}s acc={acc}", flush=True)

    print("done. accuracy by benchmark:")
    for b in n_by_bench:
        print(f"  {b:18s} {correct_by_bench[b]}/{n_by_bench[b]} = {correct_by_bench[b]/n_by_bench[b]:.3f}")
    print(f"-> {OUT}\n-> {OUT_ROWS}")


if __name__ == "__main__":
    main()
