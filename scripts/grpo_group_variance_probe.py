"""Pre-check for Attr-GRPO: with G independent team rollouts per task at T=1.0, what
fraction of train tasks give an INFORMATIVE group (some correct, some wrong)? Zero-
variance groups carry no GRPO gradient (the failure mode of the earlier attributor
GRPO); this measures how much of the training set is usable before we spend days.

    python scripts/grpo_group_variance_probe.py --model Qwen/Qwen2.5-3B-Instruct --n_tasks 60 --group 8
"""
import argparse, json, random, sys, time
from collections import Counter
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from error_attr.selfplay.debate_runner import DebateConfig, QwenDebate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n_tasks", type=int, default=60)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    split = json.load(open(ROOT / "data" / "selfplay" / "opd_task_split.json"))
    tasks = [json.loads(l) for l in open(ROOT / "data" / "selfplay" / "task_pool.jsonl")]
    tasks = [t for t in tasks if t["benchmark"] in ("math", "gpqa", "mmlu_pro", "scibench") and split.get(t["source_trace_id"]) == "train"]
    random.Random(a.seed).shuffle(tasks); tasks = tasks[: a.n_tasks]
    tok = AutoTokenizer.from_pretrained(a.model); tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto").eval()
    deb = QwenDebate(model, tok, DebateConfig(3, 2, a.max_new_tokens, a.temperature))
    kinds = Counter(); by_bench = {}; t0 = time.time(); recs = []
    for i, t in enumerate(tasks):
        outs = [deb.run(t, f"probe_{i}_{g}")["final_correct"] for g in range(a.group)]
        k = sum(outs); kind = "all_wrong" if k == 0 else ("all_correct" if k == a.group else "mixed")
        kinds[kind] += 1; by_bench.setdefault(t["benchmark"], Counter())[kind] += 1
        recs.append({"task": t["source_trace_id"], "benchmark": t["benchmark"], "pass_rate": k / a.group})
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(tasks)}] {time.time()-t0:.0f}s {dict(kinds)}", flush=True)
    print(f"\n=== G={a.group}, T={a.temperature}, {a.model}: informative (mixed) groups {kinds['mixed']}/{len(tasks)} = {kinds['mixed']/len(tasks):.2f}")
    for b, c in by_bench.items():
        print(f"  {b:10s} {dict(c)}")
    if a.out:
        json.dump(recs, open(a.out, "w"), indent=0)


if __name__ == "__main__":
    main()
