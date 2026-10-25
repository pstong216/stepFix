"""Counterfactual check of the verified-only targets: for each verified example (state t of a failed episode, the
student's original action a_minus, the verified alternative), replay the prefix and let the INITIAL student continue
greedily (cap 20 steps) after (a) a_minus and (b) the alternative. A 'fix' is causal only if (b) wins and (a) loses;
if (a) also wins the game was recoverable anyway (the failure was a T=1 sample) and the target is not a correction.
Usage: python scripts/alfworld_verify_counterfactual.py --arm attr_nofb --round round1 [--adapter reports/alfworld/student3b_low100/final]
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from error_attr.policy.alfworld_agent import LLMPolicy
from alfworld_attr_opd import verify_alternatives, load_jsonl


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True); ap.add_argument("--round", default="round1")
    ap.add_argument("--tag", default="student3b_low100"); ap.add_argument("--adapter", default="reports/alfworld/student3b_low100/final")
    ap.add_argument("--student", default="Qwen/Qwen2.5-3B-Instruct"); ap.add_argument("--max_continue_steps", type=int, default=20); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--traj", default=None, help="trajectories the examples came from (default: the round-1 pool)")
    a = ap.parse_args()
    d = Path("reports/alfworld/opd") / a.tag / a.arm; out = d / a.round / "counterfactual.jsonl"
    eps = {e["id"]: e for e in load_jsonl(a.traj or f"reports/alfworld/{a.tag}_train_b50.jsonl")}
    ex = [r for r in load_jsonl(d / a.round / "examples.jsonl") if r.get("kind") == "verified"]
    done = {(r["id"], r["t"]) for r in load_jsonl(out)} if out.exists() else set()
    ex = [r for r in ex if (r["id"], r["t"]) not in done]
    if a.limit: ex = ex[: a.limit]
    print(f"{len(ex)} verified examples to check ({len(done)} done)", flush=True)
    tok = AutoTokenizer.from_pretrained(a.student); tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(a.student, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, a.adapter); model.eval(); pol = LLMPolicy(model, tok)
    f = open(out, "a"); t0 = time.time(); n = 0; c = {"alt_win": 0, "aminus_win": 0, "causal": 0, "both_win": 0, "neither": 0}
    for i, r in enumerate(ex):
        e = eps[r["id"]]; t = r["t"]; alts = [r["a_minus"], r["target_top"]]
        with torch.no_grad():
            res, used = verify_alternatives(e, t, alts, pol, e["budget"], 0.0, a.max_continue_steps)
        aw = bool(res.get(r["a_minus"])); bw = bool(res.get(r["target_top"]))
        c["alt_win"] += bw; c["aminus_win"] += aw; c["causal"] += (bw and not aw); c["both_win"] += (aw and bw); c["neither"] += (not aw and not bw); n += 1
        f.write(json.dumps({"id": r["id"], "t": t, "a_minus": r["a_minus"], "alt": r["target_top"], "a_minus_won": aw, "alt_won": bw, "env_steps": used}) + "\n"); f.flush()
        if (i + 1) % 10 == 0:
            print(f"[{i + 1}/{len(ex)}] " + " ".join(f"{k}={v}" for k, v in c.items()) + f"  {(time.time() - t0) / (i + 1):.1f}s/example", flush=True)
    print("DONE counterfactual", json.dumps({"n": n, **c}), flush=True)


if __name__ == "__main__":
    main()
