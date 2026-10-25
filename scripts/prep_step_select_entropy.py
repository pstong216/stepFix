import json, os
D = "reports/alfworld/step_select"; os.makedirs(f"{D}/entropy", exist_ok=True)
H = {}
for fn in ["entropy.jsonl", "entropy_b.jsonl", "entropy_c.jsonl"]:
    if os.path.exists(f"{D}/{fn}"):
        for l in open(f"{D}/{fn}"): r = json.loads(l); H[r["id"]] = r["entropy"]
ids = [json.loads(l)["id"] for l in open(f"{D}/ids100.jsonl")]
missing = [i for i in ids if i not in H]
if missing: print("missing", len(missing)); raise SystemExit(1)
with open(f"{D}/entropy/steps.jsonl", "w") as f:
    for i in ids: h = H[i]; f.write(json.dumps({"id": i, "steps": [max(range(len(h)), key=lambda k: h[k])], "mode": None}) + "\n")
json.dump(H, open(f"{D}/entropy_all.json", "w")); print("entropy steps written for", len(ids))
