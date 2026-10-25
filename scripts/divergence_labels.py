"""Label source 3 for the self-trained attributor: success/failure contrast of the SAME game across the student's own collections.
For every failed trajectory, find the won trajectory of the same game_file (any collection) with the longest common action prefix;
the divergence step is the label, the type comes from how the two actions differ.  Also emits DPO pairs at the divergence step
(identical prefix => identical observation): chosen = the win's action, rejected = the failure's action.
    python scripts/divergence_labels.py --trajs A.jsonl B.jsonl ... --out_dir reports/alfworld/self_attr_labels/div
"""
import argparse, collections, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from alfworld_attr_opd import load_jsonl


def kind(a_fail, a_win):
    f, w = a_fail.split()[0], a_win.split()[0]
    if f == w and f in ("go", "take", "open", "close", "examine"): return "wrong_target"
    if f in ("put", "move") and w in ("put", "move"): return "wrong_value"
    if f in ("look", "inventory", "help", "examine") or (f == "go" and w != "go"): return "skip" if w != "go" else "wrong_target"
    return "wrong_op"


ap = argparse.ArgumentParser(); ap.add_argument("--trajs", nargs="+", required=True); ap.add_argument("--out_dir", required=True); ap.add_argument("--min_prefix", type=int, default=0)
a = ap.parse_args(); out = ROOT / a.out_dir; out.mkdir(parents=True, exist_ok=True)
wins = collections.defaultdict(list); fails = []
for p in a.trajs:
    for e in load_jsonl(ROOT / p):
        (wins if e["won"] else None) is not None and wins[e["game_file"]].append((p, e))
        if not e["won"]: fails.append((p, e))
labels, pairs, bysrc = [], [], collections.Counter()
for src, f in fails:
    best = None
    for _, w in wins.get(f["game_file"], []):
        k = 0
        while k < min(len(f["steps"]), len(w["steps"])) and f["steps"][k]["action"] == w["steps"][k]["action"]: k += 1
        if k >= len(w["steps"]): continue   # the failure contains the whole win as a prefix (budget/variant quirk): no divergence
        if best is None or k > best[0]: best = (k, w)
    if best is None or best[0] < a.min_prefix: continue
    t, w = best; typ = kind(f["steps"][t]["action"], w["steps"][t]["action"])
    labels.append({"id": f["id"], "src": src, "t_star": t, "type": typ, "win_id": w["id"], "win_len": len(w["steps"])}); bysrc[src] += 1
    pairs.append({"kind": "dpo", "src": src, "id": f["id"], "t": t, "chosen": w["steps"][t]["action"], "rejected": f["steps"][t]["action"], "win_id": w["id"]})
with open(out / "labels.jsonl", "w") as g:
    for r in labels: g.write(json.dumps(r) + "\n")
with open(out / "dpo_pairs.jsonl", "w") as g:
    for r in pairs: g.write(json.dumps(r) + "\n")
import statistics as st
ts = [r["t_star"] for r in labels]
print(f"=== divergence labels: {len(labels)} of {len(fails)} failures have a same-game win | t* mean {st.mean(ts):.1f} median {st.median(ts):.0f} t0 {sum(t == 0 for t in ts) / len(ts):.2f} t<=2 {sum(t <= 2 for t in ts) / len(ts):.2f} t>=5 {sum(t >= 5 for t in ts) / len(ts):.2f} | types {dict(collections.Counter(r['type'] for r in labels))} | per source {dict(bysrc)}")
