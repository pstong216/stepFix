"""Evaluate a (possibly OPD-trained) student as the SAME debate team it was rolled
out as: 3 debaters sharing weights, 2 rounds, majority vote, GREEDY decoding so
arms differ only by weights. Runs on the OPD task split (default: test).

    python scripts/opd_eval.py --student_model Qwen/Qwen2.5-7B-Instruct --name 7b_base
    python scripts/opd_eval.py --student_model Qwen/Qwen2.5-7B-Instruct \
        --adapter reports/opd_debate/student_attr/final --name 7b_attr
Output: reports/opd_eval/<name>_<split>.jsonl (+ accuracy by benchmark on stdout).
"""
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from error_attr.selfplay.debate_runner import DebateConfig, QwenDebate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test", choices=["test", "train", "all"])
    ap.add_argument("--split_file", default=str(ROOT / "data" / "selfplay" / "opd_task_split.json"))
    ap.add_argument("--benchmarks", default="math,gpqa,mmlu_pro,scibench")
    ap.add_argument("--n_agents", type=int, default=3)
    ap.add_argument("--n_rounds", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    split = json.load(open(a.split_file)); keep = set(a.benchmarks.split(","))
    tasks = [json.loads(l) for l in open(ROOT / "data" / "selfplay" / "task_pool.jsonl")]
    tasks = [t for t in tasks if t["benchmark"] in keep and (a.split == "all" or split.get(t["source_trace_id"]) == a.split)]
    if a.limit:
        tasks = tasks[: a.limit]
    out_dir = ROOT / "reports" / "opd_eval"; out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{a.name}_{a.split}.jsonl"
    tok = AutoTokenizer.from_pretrained(a.student_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.student_model, dtype=torch.bfloat16, device_map="auto")
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.eval()
    debate = QwenDebate(model, tok, DebateConfig(a.n_agents, a.n_rounds, a.max_new_tokens, temperature=0.0))
    n_by, ok_by = defaultdict(int), defaultdict(int); t0 = time.time()
    print(f"{a.name}: {len(tasks)} {a.split} tasks -> {out}", flush=True)
    with open(out, "w") as f:
        for i, task in enumerate(tasks):
            rel = debate.run(task, f"opd_{a.name}_{task['benchmark']}_{i:04d}")
            rel["task_source_id"] = task["source_trace_id"]
            f.write(json.dumps({"task_source_id": task["source_trace_id"], "benchmark": task["benchmark"], "final_answer": rel["final_answer"],
                                "final_correct": rel["final_correct"], "per_agent_correct_last_round": rel["per_agent_correct_last_round"]}) + "\n")
            f.flush(); n_by[task["benchmark"]] += 1; ok_by[task["benchmark"]] += rel["final_correct"]
            if (i + 1) % 20 == 0:
                print(f"[{i+1}/{len(tasks)}] {time.time()-t0:.0f}s acc={{ {', '.join(f'{b}: {ok_by[b]}/{n_by[b]}' for b in n_by)} }}", flush=True)
    tot_n, tot_ok = sum(n_by.values()), sum(ok_by.values())
    print(f"=== {a.name} on {a.split} (n={tot_n}) === accuracy {tot_ok}/{tot_n} = {tot_ok/tot_n:.3f}")
    for b in sorted(n_by):
        print(f"  {b:10s} {ok_by[b]}/{n_by[b]} = {ok_by[b]/n_by[b]:.3f}")


if __name__ == "__main__":
    main()
