"""Build ALFWorld self-play attribution SFT data for Model A from progress-oracle labels (first harmful action).
Prompt = Who&When alfagent rendering, agent-text-only (no_observations) variant; target = the 3-line diagnosis with
Step Number = t_first_regress + 1 and the mode taken from Model A's own earlier prediction on that trace (or R.2).
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src"))
from whowhen_eval.prompts import load_taxonomy
from whowhen_eval.render import get_renderer
from error_attr.attribution.ablation import build_prompt
DATA_ROOT = ROOT / "data" / "whowhen_pro_official"
SETS = {  # tag: (trajectories, progress labels, model-A predictions or None)
    "zeroshot": ("reports/alfworld/student3b_train.jsonl", "reports/alfworld/progress_student3b_train.jsonl", "reports/alfworld/attr7bnoobs_student3b_train.jsonl"),
    "low100_r1": ("reports/alfworld/student3b_low100_train_b50.jsonl", "reports/alfworld/progress_student3b_low100_train_b50.jsonl", "reports/alfworld/attr7bnoobs_student3b_low100_train_b50.jsonl"),
    "oracle_r2": ("reports/alfworld/student3b_low100_r2_oracle_train_b50.jsonl", "reports/alfworld/progress_student3b_low100_r2_oracle.jsonl", None),
    "strong": ("reports/alfworld/sft3b_train_b50.jsonl", "reports/alfworld/progress_sft3b_train_b50.jsonl", "reports/alfworld/attr7bnoobs_sft3b_train_b50.jsonl"),
    "oracle_r3": ("reports/alfworld/student3b_low100_round3_oracle_train_b50.jsonl", "reports/alfworld/progress_student3b_low100_round3_oracle.jsonl", None),
    # round-2 failure sets (v2 curriculum, 2026-09-21): the policies the round-3 attributor will actually be pointed at
    "attr_nofb_r2": ("reports/alfworld/student3b_low100_round2_attr_nofb_train_b50.jsonl", "reports/alfworld/progress_student3b_low100_round2_attr_nofb.jsonl", "reports/alfworld/attr7bnoobs_student3b_low100_round2_attr_nofb.jsonl"),
    "attralf_r2": ("reports/alfworld/student3b_low100_round2_attralf_nofb_train_b50.jsonl", "reports/alfworld/progress_student3b_low100_round2_attralf_nofb.jsonl", "reports/alfworld/attr7balf_student3b_low100_round2_attralf_nofb.jsonl"),
    "oracle_nofb_r2": ("reports/alfworld/student3b_low100_round2_oracle_nofb_train_b50.jsonl", "reports/alfworld/progress_student3b_low100_round2_oracle_nofb.jsonl", None),
}


def to_release(ep):
    traj = [{"kind": "user", "content": ep["initial_obs"]}] + [{"kind": "action", "step_number": i + 1, "reasoning": None, "action": st["action"], "observation": st["next_obs"], "is_final_answer": False} for i, st in enumerate(ep["steps"])]
    return {"id": ep["id"], "framework": "alfagent", "benchmark": "alfworld", "modality": "text", "task": {"query": ep["goal"], "answer": ""}, "trajectory": traj, "ground_truth": {}, "agents": [{"id": "agent", "role": "agent"}]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--heldout", nargs="*", default=["oracle_r3"]); ap.add_argument("--out", default="data/attributor_sft_alf")
    a = ap.parse_args(); tax = load_taxonomy(DATA_ROOT); render = get_renderer("alfagent")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); ftr = open(out / "train.jsonl", "w"); fho = open(out / "heldout.jsonl", "w")
    n = {"train": 0, "heldout": 0}
    for tag, (tp, pp, ap_) in SETS.items():
        if not Path(pp).exists(): print("skip", tag); continue
        eps = {json.loads(l)["id"]: json.loads(l) for l in open(tp)}; prog = {json.loads(l)["id"]: json.loads(l) for l in open(pp)}
        modes = {json.loads(l)["id"]: (json.loads(l).get("pred") or {}).get("error_mode") for l in open(ap_)} if ap_ and Path(ap_).exists() else {}
        k = 0
        for i, r in prog.items():
            t = r.get("t_first_regress")
            if t is None or i not in eps: continue
            e = eps[i]; rel = to_release(e); rr = render(rel)
            prompt, _, _ = build_prompt("no_observations", rel, rr, tax)
            mode = modes.get(i) or "R.2"
            row = {"id": i, "framework": "alfagent", "benchmark": "alfworld", "prompt": prompt, "target": f"Agent Name: agent\nStep Number: {t + 1}\nError Mode: {mode}",
                   "gold_agent": "agent", "gold_step": t + 1, "gold_mode": mode, "ground_truth": {"agent": None, "step": t + 1, "mode": mode}, "source": tag}
            (fho if tag in a.heldout else ftr).write(json.dumps(row) + "\n"); n["heldout" if tag in a.heldout else "train"] += 1; k += 1
        print(f"{tag}: {k} rows ({'heldout' if tag in a.heldout else 'train'})")
    print("total", n)


if __name__ == "__main__":
    main()
