"""From the attr-arm rollout verification: DPO pairs only for REAL fixes (candidate wins, logged action loses), a matched
placebo (same states, random admissible action as 'chosen'), and how well the environment-proxy veto predicted the labels."""
import json, random, sys, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
O = ROOT / "reports/alfworld/self_evolve7b"; TRAJ = "reports/alfworld/self_evolve7b/r1_train.jsonl"
V = [json.loads(l) for s in (0, 1) for l in open(O / "attr" / f"verify_s{s}.jsonl")]
T = {(json.loads(l)["id"], json.loads(l)["t"]): json.loads(l) for l in open(O / "attr" / "targets.jsonl")}
rng = random.Random(0); pairs = []; plac = []; n_cand_win = 0; n_orig_win = 0
for v in V:
    w = v["won"]; n_orig_win += w[v["a_minus"]]; n_cand_win += any(w[c] for c in v["cands"])
    fixes = [c for c in v["cands"] if w[c] and not w[v["a_minus"]]]
    if fixes:
        pairs.append({"kind": "dpo", "src": TRAJ, "id": v["id"], "t": v["t"], "chosen": fixes[0], "rejected": v["a_minus"]})
        adm = [c for c in T[(v["id"], v["t"])]["admissible"] if c != v["a_minus"]]
        plac.append({"kind": "dpo", "src": TRAJ, "id": v["id"], "t": v["t"], "chosen": rng.choice(adm), "rejected": v["a_minus"]})
for name, rows in (("attr_verified", pairs), ("placebo_verified", plac)):
    (O / name).mkdir(exist_ok=True)
    with open(O / name / "dpo_pairs.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
n = len(V); print(f"{n} states verified: some candidate wins {n_cand_win} ({n_cand_win / n:.0%}), logged action wins {n_orig_win} ({n_orig_win / n:.0%}), REAL fixes {len(pairs)} ({len(pairs) / n:.0%})")
print("verbs of real fixes:", dict(collections.Counter(p["chosen"].split()[0] for p in pairs)))
# how good was the v4/v5 proxy selection (veto + hindsight ranking) on these labels?
sel = {(json.loads(l)["id"], json.loads(l)["t"]): json.loads(l)["chosen"] for l in open(O / "attr" / "dpo_pairs.jsonl")}
lab = {(v["id"], v["t"]): v for v in V}
k = [s for s in sel if s in lab]; good = sum(lab[s]["won"].get(sel[s], False) and not lab[s]["won"][lab[s]["a_minus"]] for s in k)
unsel = [s for s in lab if s not in sel]; fix_unsel = sum(any(lab[s]["won"][c] for c in lab[s]["cands"]) and not lab[s]["won"][lab[s]["a_minus"]] for s in unsel)
print(f"proxy-selected pairs (v5): {len(k)}, real-fix rate {good / max(1, len(k)):.0%} | states the proxy dropped: {len(unsel)}, of which with a real fix available {fix_unsel} ({fix_unsel / max(1, len(unsel)):.0%})")
