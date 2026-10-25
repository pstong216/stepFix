"""Student's full action distribution at given (id, t) states (for teacher-vs-student comparisons)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch
from alfworld_attr_opd import history_at, load_jsonl
from alfworld_opd_plus import load_policy
inp, traj, out = sys.argv[1], sys.argv[2], sys.argv[3]
adapter = sys.argv[4] if len(sys.argv) > 4 else str(ROOT / "reports/alfworld/student3b_low100/final")
rows = json.load(open(ROOT / inp)); eps = {e["id"]: e for e in load_jsonl(ROOT / traj)}
tok, model, pol = load_policy("Qwen/Qwen2.5-3B-Instruct", adapter)
with open(ROOT / out, "w") as f, torch.no_grad():
    for r in rows:
        lp = pol.score_admissible_commands(history_at(eps[r["id"]], r["t"]))
        f.write(json.dumps({"id": r["id"], "t": r["t"], "student_logprobs": lp}) + "\n")
print("DONE", flush=True)
