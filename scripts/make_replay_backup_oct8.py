"""Replay examples against forgetting: (state, taken action) SFT rows sampled from the student's OWN won trajectories of this
round, shortest wins first (compressed self-imitation), a few steps per episode, skipping look/inventory/help.
    python scripts/make_replay.py --traj D/traj.jsonl --n 86 --out D/replay.jsonl
"""
import argparse, json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from alfworld_attr_opd import load_jsonl

ap = argparse.ArgumentParser(); ap.add_argument("--traj", required=True); ap.add_argument("--n", type=int, default=86); ap.add_argument("--per_ep", type=int, default=2)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True); a = ap.parse_args()
rng = random.Random(a.seed); src = str(Path(a.traj)) if not Path(a.traj).is_absolute() else str(Path(a.traj).relative_to(ROOT))
wins = sorted([e for e in load_jsonl(ROOT / a.traj) if e["won"]], key=lambda e: len(e["steps"])); rows = []
for e in wins:
    ok = [t for t, s in enumerate(e["steps"]) if s["action"].split()[0] not in ("look", "inventory", "help", "examine")]
    for t in rng.sample(ok, min(a.per_ep, len(ok))): rows.append({"kind": "sft", "src": src, "id": e["id"], "t": t, "action": e["steps"][t]["action"]})
    if len(rows) >= a.n: break
rows = rows[: a.n]; Path(ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
with open(ROOT / a.out, "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(f"replay: {len(rows)} steps from {len({r['id'] for r in rows})} shortest wins (mean length {sum(len(e['steps']) for e in wins[:len({r['id'] for r in rows})]) / max(1, len({r['id'] for r in rows})):.1f})")
