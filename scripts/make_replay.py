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
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
ap.add_argument("--mode", default="shortest", choices=["shortest", "random_full", "all", "random_eps", "fail_weighted"],
                help="shortest = old rule (shortest wins, per_ep random steps each); random_full (Oct 8, user) = RANDOM wins, every step of each\n                win, until --n steps; all = every step of every win (standard RFT data); random_eps = --n RANDOM wins, every step of each (full-trajectory SFT); fail_weighted (Oct 9) = --n steps split over ALFWorld task types in proportion to the round's OWN training failures of each type, filled with complete trajectories (non-look steps) of that type's shortest wins")
a = ap.parse_args()
rng = random.Random(a.seed); src = str(Path(a.traj)) if not Path(a.traj).is_absolute() else str(Path(a.traj).relative_to(ROOT))
ALL = load_jsonl(ROOT / a.traj)
wins = sorted([e for e in ALL if e["won"]], key=lambda e: len(e["steps"])); rows = []
if a.mode == "fail_weighted":   # replay where the agent itself fails most (training collection only, no test data)
    import re
    TY = lambda e: (re.search(r"(pick_and_place_simple|look_at_obj_in_light|pick_clean_then_place|pick_heat_then_place|pick_cool_then_place|pick_two_obj)", e.get("game_file", "")) or [None, "other"])[1]
    fails = {}
    for e in ALL:
        if not e["won"]: fails[TY(e)] = fails.get(TY(e), 0) + 1
    tot = sum(fails.values()); quota = {k: round(a.n * v / tot) for k, v in fails.items()}
    for k in sorted(quota, key=lambda k: -quota[k]):
        got = 0
        for e in [w for w in wins if TY(w) == k]:
            if got >= quota[k]: break
            for t, s_ in enumerate(e["steps"]):
                if s_["action"].split()[0] in ("look", "inventory", "help", "examine") or got >= quota[k]: continue
                rows.append({"kind": "sft", "src": src, "id": e["id"], "t": t, "action": s_["action"]}); got += 1
    print("fail_weighted quota by type:", quota)
    wins = []
if a.mode != "shortest": rng.shuffle(wins)   # random order of the round's own wins (no length preference)
for e in wins:
    ok = [t for t, s in enumerate(e["steps"]) if a.mode == "all" or s["action"].split()[0] not in ("look", "inventory", "help", "examine")]
    if a.mode == "random_eps": ok = list(range(len(e["steps"])))   # full trajectory, every action
    pick = rng.sample(ok, min(a.per_ep, len(ok))) if a.mode == "shortest" else ok
    for t in pick: rows.append({"kind": "sft", "src": src, "id": e["id"], "t": t, "action": e["steps"][t]["action"]})
    if a.mode == "random_eps" and len({r["id"] for r in rows}) >= a.n: break
    if a.mode not in ("all", "random_eps") and len(rows) >= a.n: break
rows = rows if a.mode in ("all", "random_eps", "fail_weighted") else rows[: a.n]; Path(ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
with open(ROOT / a.out, "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(f"replay ({a.mode}): {len(rows)} steps from {len({r['id'] for r in rows})} wins (mean length {sum(len(e['steps']) for e in wins[:len({r['id'] for r in rows})]) / max(1, len({r['id'] for r in rows})):.1f})")
