"""Rollout audit of self-proposed corrections: for a sample of each arm's DPO pairs, continue greedily with the untrained 7B
after (a) the chosen correction and (b) the logged action a-. Reports win rate and causal-fix rate (chosen wins, a- loses)."""
import json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch
from alfworld_attr_opd import load_jsonl, verify_alternatives
from alfworld_opd_plus import load_policy
O = ROOT / "reports/alfworld/self_evolve7b"; arms = sys.argv[1].split(","); n = int(sys.argv[2]); out = O / sys.argv[3]
eps = {e["id"]: e for e in load_jsonl(O / "r1_train.jsonl")}
done = {(json.loads(l)["arm"], json.loads(l)["id"], json.loads(l)["t"]) for l in open(out)} if out.exists() else set()
tok, model, pol = load_policy("Qwen/Qwen2.5-7B-Instruct", None)
with open(out, "a") as f, torch.no_grad():
    for arm in arms:
        rows = load_jsonl(O / arm / "dpo_pairs.jsonl"); random.Random(0).shuffle(rows)
        for r in rows[:n]:
            if (arm, r["id"], r["t"]) in done: continue
            e = eps[r["id"]]
            res, _ = verify_alternatives(e, r["t"], [r["chosen"], r["rejected"]], pol, e["budget"], max_continue_steps=20)
            f.write(json.dumps({"arm": arm, "id": r["id"], "t": r["t"], "chosen": r["chosen"], "rejected": r["rejected"],
                                "chosen_won": bool(res.get(r["chosen"])), "rejected_won": bool(res.get(r["rejected"]))}) + "\n"); f.flush()
        print("done", arm, flush=True)
print("DONE audit", flush=True)
