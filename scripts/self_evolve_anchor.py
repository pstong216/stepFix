"""KL anchor to the UNTRAINED model's own action distribution on its own states (150 from wins + 150 from failures)."""
import json, math, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch
from alfworld_attr_opd import history_at, load_jsonl
from alfworld_opd_plus import load_policy
O = "reports/alfworld/self_evolve7b"; T = f"{O}/r1_train.jsonl"
E = load_jsonl(ROOT / T); rng = random.Random(1)
def sample(eps, n):
    pool = [(e["id"], t) for e in eps for t in range(len(e["steps"]))]; rng.shuffle(pool); return pool[:n]
states = sample([e for e in E if e["won"]], 150) + sample([e for e in E if not e["won"]], 150)
eps = {e["id"]: e for e in E}
tok, model, pol = load_policy("Qwen/Qwen2.5-7B-Instruct", None)
with open(ROOT / O / "anchor_kl300.jsonl", "w") as f, torch.no_grad():
    for i, t in states:
        lp = pol.score_admissible_commands_cached(history_at(eps[i], t)); cmds = list(lp)
        z = max(lp.values()); p = [math.exp(lp[c] - z) for c in cmds]; s = sum(p)
        f.write(json.dumps({"kind": "anchor", "src": T, "id": i, "t": t, "cmds": cmds, "target": [x / s for x in p]}) + "\n")
print("DONE anchor", len(states), flush=True)
