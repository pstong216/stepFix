"""Student action entropy at every step of given failed trajectories (KV-cached scoring over admissible commands)."""
import json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch
from alfworld_attr_opd import history_at, load_jsonl
from alfworld_opd_plus import load_policy
ids_file, traj, out = sys.argv[1:4]
ids = [json.loads(l)["id"] for l in open(ROOT / ids_file)]; eps = {e["id"]: e for e in load_jsonl(ROOT / traj)}
done = {json.loads(l)["id"] for l in open(ROOT / out)} if (ROOT / out).exists() else set()
tok, model, pol = load_policy("Qwen/Qwen2.5-3B-Instruct", str(ROOT / "reports/alfworld/student3b_low100/final"))
with open(ROOT / out, "a") as f, torch.no_grad():
    for k, i in enumerate(ids):
        if i in done: continue
        e = eps[i]; H = []
        for t in range(len(e["steps"])):
            lp = pol.score_admissible_commands_cached(history_at(e, t)); z = max(lp.values())
            Z = z + math.log(sum(math.exp(v - z) for v in lp.values())); p = [math.exp(v - Z) for v in lp.values()]
            H.append(-sum(q * math.log(q + 1e-12) for q in p))
        f.write(json.dumps({"id": i, "entropy": H}) + "\n"); f.flush()
        print(f"[{k + 1}/{len(ids)}]", flush=True)
print("DONE entropy", flush=True)
