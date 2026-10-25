"""Re-grade a self-play pool file in place under the CURRENT parser/grader
(DECISIONS #25 brace-matched \\boxed{}, #26 numeric tolerance), from the
stored trajectory text. Keeps the previous labels as *_v1.

    python scripts/regrade_selfplay.py --file data/selfplay/debate_trajectories.jsonl
    python scripts/regrade_selfplay.py --file data/selfplay/mathchat_trajectories.jsonl \
        --ids_from reports/X/corrections.jsonl --out_ids reports/X/true_failure_ids.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from error_attr.selfplay.debate_runner import is_correct, normalize, parse_task_answer  # noqa: E402


def final_answer(rel: dict) -> str | None:
    fw, bench, traj = rel["framework"], rel["benchmark"], rel["trajectory"]
    if fw == "debate":
        last = [e for e in traj if e.get("kind") == "round"][-1]["turns"]
        parsed = [parse_task_answer(bench, t["output"]) for t in last]
        votes = Counter(normalize(p) for p in parsed if p is not None)
        if votes:
            top, _ = votes.most_common(1)[0]
            return next(p for p in parsed if p is not None and normalize(p) == top)
        return parsed[0]
    if fw == "mathchat":
        turns = [e for e in traj if e.get("agent_id") == "assistant"]
        return parse_task_answer(bench, turns[-1]["output"]) if turns else None
    if fw == "magentic-one":
        fin = [e for e in traj if e.get("kind") == "agent" and "final_answer" in (e.get("phase") or [])]
        return parse_task_answer(bench, fin[-1]["output"]) if fin else None
    raise ValueError(fw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--ids_from", default=None, help="a corrections.jsonl; list its traces that are TRUE failures")
    ap.add_argument("--out_ids", default=None)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.file)]
    flips = Counter()
    for r in rows:
        if "final_correct_v1" not in r:
            r["final_correct_v1"], r["final_answer_v1"] = r["final_correct"], r.get("final_answer")
        fa = final_answer(r)
        ok = is_correct(r["benchmark"], fa, r["task"]["answer"])
        if ok != r["final_correct_v1"]:
            flips[(r["benchmark"], "wrong->right" if ok else "right->wrong")] += 1
        r["final_answer"], r["final_correct"] = fa, ok
    with open(a.file, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    acc = Counter(); n = Counter()
    for r in rows:
        acc[r["benchmark"]] += r["final_correct"]; n[r["benchmark"]] += 1
    print(f"{a.file}: {len(rows)} rows re-graded; flips vs v1: {dict(flips)}")
    print("accuracy now:", {b: f"{acc[b]}/{n[b]}" for b in n})
    if a.ids_from and a.out_ids:
        ok_by = {r["id"]: r["final_correct"] for r in rows}
        ids = sorted({json.loads(l)["id"] for l in open(a.ids_from)})
        true_fail = [i for i in ids if not ok_by[i]]
        json.dump(true_fail, open(a.out_ids, "w"), indent=0)
        print(f"{len(true_fail)}/{len(ids)} traces in {a.ids_from} are true failures -> {a.out_ids}")


if __name__ == "__main__":
    main()
