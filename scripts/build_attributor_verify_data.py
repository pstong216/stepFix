"""Build planner-free refresh data for Model A from the training loop's own verification outcomes.
Label for a failed episode = the earliest step, among the steps that were actually tried in the given round-1 arms,
at which a replacement action was verified to win (examples.jsonl, kind=verified). Only arms whose step positions
were chosen without the PDDL planner may be passed in. Prompt/target format is identical to build_attributor_alf_data.py.
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from whowhen_eval.prompts import load_taxonomy
from whowhen_eval.render import get_renderer
from error_attr.attribution.ablation import build_prompt
from build_attributor_alf_data import to_release, DATA_ROOT

PLANNER_ARMS = ("oracle", "tstar")


def load_jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in open(p)] if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="reports/alfworld/student3b_low100_train_b50.jsonl")
    ap.add_argument("--arms", nargs="+", default=["attr_nofb", "attr_nofb_p1", "attr_nofb_p2", "const1_nofb", "self_nofb"])
    ap.add_argument("--opd_dir", default="reports/alfworld/opd/student3b_low100")
    ap.add_argument("--modes", default="reports/alfworld/attr7bnoobs_student3b_low100_train_b50.jsonl", help="Model A's own earlier predictions, for the Error Mode line")
    ap.add_argument("--out", default="data/attributor_sft_verify")
    a = ap.parse_args()
    for arm in a.arms:
        if arm.startswith(PLANNER_ARMS):
            sys.exit(f"refusing arm {arm}: its step positions come from the planner")

    eps = {e["id"]: e for e in load_jsonl(ROOT / a.traj)}
    modes = {r["id"]: (r.get("pred") or {}).get("error_mode") for r in load_jsonl(ROOT / a.modes)}
    tried, verified = {}, {}
    for arm in a.arms:
        ex = load_jsonl(ROOT / a.opd_dir / arm / "round1" / "examples.jsonl")
        sk = load_jsonl(ROOT / a.opd_dir / arm / "round1" / "skipped.jsonl")
        if not ex and not sk:
            sys.exit(f"no round1 examples/skipped for {arm}")
        for r in sk:
            tried.setdefault(r["id"], set()).add(r["t"])
        for r in ex:
            tried.setdefault(r["id"], set()).add(r["t"])
            if r.get("kind") == "verified":
                verified.setdefault(r["id"], set()).add(r["t"])
        print(f"{arm}: {len(ex)} verified, {len(sk)} skipped")

    tax = load_taxonomy(DATA_ROOT); render = get_renderer("alfagent")
    out = Path(ROOT / a.out); out.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out / "train.jsonl", "w") as f:
        for i, ts in sorted(verified.items()):
            if i not in eps:
                continue
            t = min(ts)
            e = eps[i]; rel = to_release(e); rr = render(rel)
            prompt, _, _ = build_prompt("no_observations", rel, rr, tax)
            mode = modes.get(i) or "R.2"
            f.write(json.dumps({"id": i, "framework": "alfagent", "benchmark": "alfworld", "prompt": prompt,
                                "target": f"Agent Name: agent\nStep Number: {t + 1}\nError Mode: {mode}",
                                "gold_agent": "agent", "gold_step": t + 1, "gold_mode": mode,
                                "ground_truth": {"agent": None, "step": t + 1, "mode": mode},
                                "source": "verify_r1", "tried_steps": sorted(tried[i]), "verified_steps": sorted(ts)}) + "\n")
            n += 1
    print(f"failures tried: {len(tried)} | with >=1 verified step (labelled): {n} -> {out / 'train.jsonl'}")


if __name__ == "__main__":
    main()
