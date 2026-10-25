import json
D = "reports/alfworld/self_attr_rounds/selfattr_s2/r2"
fail_ids = {json.loads(l)["id"] for l in open(f"{D}/traj.jsonl") if not json.loads(l)["won"]}
seen = set(); n = 0; bad = 0
with open(f"{D}/traj.release.jsonl", "w") as f:
    for l in open(f"{D}/traj.release.corrupt.bak.jsonl"):
        try: r = json.loads(l)
        except Exception: bad += 1; continue
        if r["id"] in fail_ids and r["id"] not in seen: seen.add(r["id"]); f.write(json.dumps(r) + "\n"); n += 1
print(f"traj games {sum(1 for _ in open(f'{D}/traj.jsonl'))}, failures {len(fail_ids)}, release rows written {n} (bad {bad}), failures without release row {len(fail_ids - seen)}")
