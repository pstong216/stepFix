import json
D = "reports/alfworld/self_attr_rounds/selfattr_s2/r2"
def to_release(ep):   # same as alfworld_collect.to_release but tolerant of a trailing observation-only step
    traj = [{"kind": "user", "content": ep["initial_obs"]}]
    for i, st in enumerate(ep["steps"]):
        if "action" not in st: continue
        traj.append({"kind": "action", "step_number": i + 1, "reasoning": None, "action": st["action"], "observation": st.get("next_obs", ""), "is_final_answer": False})
    return {"id": ep["id"], "framework": "alfagent", "benchmark": "alfworld", "modality": "text", "task": {"query": ep["goal"], "answer": ""}, "trajectory": traj, "ground_truth": {}, "extras": {"agents": [{"id": "agent", "role": "agent"}]}}
have = {json.loads(l)["id"] for l in open(f"{D}/traj.release.jsonl")}; n = 0
with open(f"{D}/traj.release.jsonl", "a") as f:
    for l in open(f"{D}/traj.jsonl"):
        e = json.loads(l)
        if not e["won"] and e["id"] not in have: f.write(json.dumps(to_release(e)) + "\n"); n += 1
print(f"appended {n}; release rows now {sum(1 for _ in open(f'{D}/traj.release.jsonl'))}")
