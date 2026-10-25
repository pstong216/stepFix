"""Repair an eval_heldout_<tag>.jsonl that two daemons appended to at once: drop unparsable rows, keep the first row per game."""
import json, shutil, sys
p = sys.argv[1]; shutil.copy(p, p + ".corrupt.bak")
seen = set(); keep = []; bad = 0
for l in open(p + ".corrupt.bak"):
    try: r = json.loads(l)
    except Exception: bad += 1; continue
    if "game_file" not in r or "won" not in r: bad += 1; continue
    if r["game_file"] in seen: continue
    seen.add(r["game_file"]); keep.append(l if l.endswith("\n") else l + "\n")
open(p, "w").writelines(keep)
rp = p.replace(".jsonl", ".release.jsonl")
try:
    rel = []; ids = set(); rb = 0
    for l in open(rp):
        try: r = json.loads(l)
        except Exception: rb += 1; continue
        if r.get("id") in ids: continue
        ids.add(r.get("id")); rel.append(l if l.endswith("\n") else l + "\n")
    open(rp, "w").writelines(rel); relmsg = f"release rows {len(rel)} (bad {rb})"
except FileNotFoundError: relmsg = "no release file"
print(f"{p}: kept {len(keep)} games (bad {bad}), wins {sum(json.loads(l)['won'] for l in keep)} | {relmsg}")
