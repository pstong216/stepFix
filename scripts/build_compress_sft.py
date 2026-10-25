"""SFT data for the trajectory-compression study: the same N won episodes of the student in three forms --
  raw     the logged trajectory (plain rejection-sampling fine-tuning),
  pruned  the replay-verified causal path (detours removed; scripts/compress_trajectories.py),
  keep1   the causal path plus the first detour block and its exit (falls back to pruned when the episode had no detour).
Rows are alfworld_opd_plus.py `sft` examples (src file, episode id, step, action); state counts differ by construction.
    python scripts/build_compress_sft.py --n 200
"""
import argparse, json, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
TRAJ = "reports/alfworld/student3b_low100_train_b50.jsonl"; COMP = "reports/alfworld/compress_traj/wins.jsonl"

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="", help="suffix for the output dirs (replications with another seed)"); a = ap.parse_args()
wins = {e["id"]: e for e in map(json.loads, open(ROOT / TRAJ)) if e["won"]}
pruned, keep1 = {}, {}
for r in map(json.loads, open(ROOT / COMP)):
    (pruned if r["variant"] == "pruned" else keep1)[r["src_id"]] = r
elig = sorted(i for i, r in pruned.items() if r["won"] and not r.get("timeout") and i in wins)
rng = random.Random(a.seed); sel = sorted(rng.sample(elig, min(a.n, len(elig))))
print(f"{len(wins)} wins, {len(elig)} with a replay-verified pruned path, {len(sel)} selected; "
      f"{sum(pruned[i]['n_removed'] > 0 for i in sel)} of them had a detour, keep1 available for {sum(i in keep1 for i in sel)}")
out = ROOT / "reports/alfworld/compress_traj"
for v in ("raw", "pruned", "keep1"):
    (out / (v + a.tag)).mkdir(parents=True, exist_ok=True); n_states = 0; lens = []
    with open(out / (v + a.tag) / "train.jsonl", "w") as f:
        for i in sel:
            if v == "raw": src, ep = TRAJ, wins[i]
            elif v == "pruned": src, ep = COMP, pruned[i]
            else: src, ep = COMP, keep1.get(i, pruned[i])
            lens.append(len(ep["steps"]))
            for t, s in enumerate(ep["steps"]):
                f.write(json.dumps({"kind": "sft", "src": src, "id": ep["id"], "t": t, "action": s["action"]}) + "\n"); n_states += 1
    print(f"{v:7s} {n_states:5d} states, mean length {sum(lens) / len(lens):.2f}")
