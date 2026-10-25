"""Repair selfattr_s2 round-2 trajectory files corrupted by two collectors appending at once (Oct 5 03:15): drop unparsable lines,
keep one row per game among the first 1500 train games, and regenerate the Model-A release file from the failures."""
import json, shutil, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from alfworld_collect import to_release, game_files
D = "reports/alfworld/self_attr_rounds/selfattr_s2/r2"
shutil.copy(f"{D}/traj.jsonl", f"{D}/traj.corrupt.bak.jsonl"); shutil.copy(f"{D}/traj.release.jsonl", f"{D}/traj.release.corrupt.bak.jsonl")
games = set(game_files("train")[:1500]); seen = set(); keep = []; bad = 0; extra = 0
for l in open(f"{D}/traj.corrupt.bak.jsonl"):
    try: e = json.loads(l)
    except Exception: bad += 1; continue
    if e["game_file"] not in games: extra += 1; continue
    if e["game_file"] in seen: continue
    seen.add(e["game_file"]); keep.append(e)
with open(f"{D}/traj.jsonl", "w") as f:
    for e in keep: f.write(json.dumps(e) + "\n")
with open(f"{D}/traj.release.jsonl", "w") as f:
    for e in keep:
        if not e["won"]: f.write(json.dumps(to_release(e)) + "\n")
print(f"kept {len(keep)} games (bad lines {bad}, outside first-1500 {extra}), won {sum(e['won'] for e in keep)}, release rows {sum(not e['won'] for e in keep)}, missing games {len(games - seen)}")
