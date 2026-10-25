"""Drop the one s2 round-2 episode whose step list was spliced by the concurrent-write corruption (a step with only 'obs')."""
import json
D = "reports/alfworld/self_attr_rounds/selfattr_s2/r2"
keep = []; dropped = []
for l in open(f"{D}/traj.jsonl"):
    e = json.loads(l)
    if all("action" in s and "next_obs" in s for s in e["steps"]): keep.append(l)
    else: dropped.append(e["id"])
open(f"{D}/traj.jsonl", "w").writelines(keep)
rel = [l for l in open(f"{D}/traj.release.jsonl") if json.loads(l)["id"] not in dropped]
open(f"{D}/traj.release.jsonl", "w").writelines(rel)
print(f"dropped {dropped}; traj {len(keep)} games, release {len(rel)} rows")
