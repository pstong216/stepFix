"""Rollout-verify, for one arm, the top-2 hindsight-teacher candidates and the logged action a- at every chosen state
(untrained 7B continues greedily, cap 20). Shardable: --shard i --nshards n."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch
from alfworld_attr_opd import load_jsonl, verify_alternatives
from alfworld_opd_plus import load_policy
O = ROOT / "reports/alfworld/self_evolve7b"; arm, shard, nsh = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
import os
eps = {e["id"]: e for e in load_jsonl(ROOT / os.environ.get("SE_TRAJ", "reports/alfworld/self_evolve7b/r1_train.jsonl"))}; out = O / arm / f"verify_s{shard}.jsonl"
done = {(json.loads(l)["id"], json.loads(l)["t"]) for l in open(out)} if out.exists() else set()
T = [r for k, r in enumerate(load_jsonl(O / arm / (sys.argv[4] if len(sys.argv) > 4 else "targets.jsonl"))) if k % nsh == shard]
tok, model, pol = load_policy("Qwen/Qwen2.5-7B-Instruct", None)
with open(out, "a") as f, torch.no_grad():
    for k, r in enumerate(T):
        if (r["id"], r["t"]) in done: continue
        lp = r["teacher_logprobs"]; cands = [c for c in sorted(lp, key=lambda c: -lp[c]) if c != r["a_minus"]][:2]
        e = eps[r["id"]]; res, _ = verify_alternatives(e, r["t"], cands + [r["a_minus"]], pol, e["budget"], max_continue_steps=20)
        f.write(json.dumps({"id": r["id"], "t": r["t"], "a_minus": r["a_minus"], "cands": cands, "won": {c: bool(res.get(c)) for c in cands + [r["a_minus"]]}}) + "\n"); f.flush()
        if k % 10 == 0: print(f"[{k}/{len(T)}]", flush=True)
print("DONE verify", flush=True)
