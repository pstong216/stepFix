"""Diagnostics for the plan-at-every-step pipeline (analysis of outputs only, Oct 9): does a plan's 'Need total' equal the base-ingredient
requirement computed from the recipes shown in the task?
  reference 'tree'   = leaves of the requirement tree textcraft_evidence.req_tree builds for Phi (oracle recipe choice per item, batches
                       rounded up per node, no sharing of leftovers between branches), summed per item
  reference 'oracle' = the oracle solver's total gets (what the SFT plans were rendered from; shares leftovers between branches)
  plans    -- corrected plans of the kept corrections (train_corrections.jsonl, role corr_plan) and replayed student plans (replay.jsonl,
              role replay_plan): match rates by depth
  eval     -- t=0 'Need total' match rate of eval episodes and success when it matches vs not
"""
import argparse, collections, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from textcraft_env import TextCraftEnv, item_id_to_str
from textcraft_evidence import req_tree
from textcraft_plan2 import oracle_plans

_ENV = TextCraftEnv(); _CACHE = {}


def L(p): return [json.loads(l) for l in open(p)]


def parse_need(plan):
    m = re.search(r"Need total:\s*(.*?)\.\s*Missing:", plan or "")
    if not m: return None
    s = re.sub(r"\s*\([^)]*\)", "", m.group(1)).strip()
    if s in ("", "nothing"): return {}
    out = {}
    try:
        for x in s.split(", "):
            n, it = x.strip().split(" ", 1); out[it.strip()] = out.get(it.strip(), 0) + int(n)
    except Exception:
        return None
    return out


def refs(task):
    if task not in _CACHE:
        _ENV.reset(task); nodes = req_tree(_ENV); kids = {n[1] for n in nodes if n[1] is not None}; tree = collections.Counter()
        for nid, par, name, is_tag, need in nodes:
            if nid not in kids and nid != 0: tree[item_id_to_str(name)] += need
        _ENV.reset(task); orc = parse_need(oracle_plans(_ENV, task, _ENV.expert_actions())[0])
        _CACHE[task] = (dict(tree), orc)
    return _CACHE[task]


def rate(rows):
    """rows: [(depth, plan, task)] -> {depth: (n, match_tree, match_oracle)} + overall."""
    by = collections.defaultdict(lambda: [0, 0, 0])
    for d, plan, task in rows:
        p = parse_need(plan); tree, orc = refs(task); by[d][0] += 1; by[d][1] += p == tree; by[d][2] += p == orc
    tot = [sum(v[i] for v in by.values()) for i in range(3)]
    return by, tot


def show(name, rows):
    by, tot = rate(rows)
    print(f"{name}: {tot[0]} plans; Need total == recipe tree {tot[1]}/{tot[0]} = {tot[1] / max(1, tot[0]):.3f}, == oracle gets {tot[2]}/{tot[0]} = {tot[2] / max(1, tot[0]):.3f}"
          + " | by depth (n, tree, oracle): " + ", ".join(f"d{d} ({v[0]}, {v[1]}, {v[2]})" for d, v in sorted(by.items())))


def cmd_plans(a):
    D = ROOT / a.round_dir; traj = {e["id"]: e for e in L(ROOT / a.traj)}
    ve = {e["id"]: e for e in L(D / "train_episodes.jsonl")} if (D / "train_episodes.jsonl").exists() else {}
    rows = []
    for r in L(D / "train_corrections.jsonl"):
        if r.get("role") != "corr_plan": continue
        e = ve.get(r["id"]) or traj[r["id"].split("__c")[0]]; rows.append((e["depth"], r["action"], e["task"]))
    show("kept corrections, corrected plan", rows)
    rows = []
    for r in L(D / "replay.jsonl"):
        if r.get("role") != "replay_plan": continue
        e = traj[r["id"]]; rows.append((e["depth"], r["action"], e["task"]))
    show("replay, student's own plan", rows)
    stud = []
    for r in L(D / "train_corrections.jsonl"):
        if r.get("role") != "corr_plan": continue
        i, t = r["id"].split("__c")[0], -r["t"] - 1; e = traj[i]; stud.append((e["depth"], e["steps"][t]["plan"], e["task"]))
    show("kept corrections, the student's ORIGINAL plan at the same states", stud)


def cmd_eval(a):
    for path in a.files:
        E = L(ROOT / path); rows = [(e["depth"], e["steps"][0]["plan"], e["task"]) for e in E]; show(f"{path} t=0", rows)
        for ref, k in (("recipe tree", 0), ("oracle gets", 1)):
            m = [e for e in E if parse_need(e["steps"][0]["plan"]) == refs(e["task"])[k]]; nm = [e for e in E if parse_need(e["steps"][0]["plan"]) != refs(e["task"])[k]]
            print(f"   vs {ref}: match {len(m)}/{len(E)}; success when match {sum(e['won'] for e in m)}/{len(m)} = {sum(e['won'] for e in m) / max(1, len(m)):.2f}, "
                  f"when not {sum(e['won'] for e in nm)}/{len(nm)} = {sum(e['won'] for e in nm) / max(1, len(nm)):.2f}")
        byd = collections.defaultdict(list)
        for e in E: byd[e["depth"]].append(e["won"])
        print("   success by depth: " + ", ".join(f"d{d} {sum(v)}/{len(v)}" for d, v in sorted(byd.items())))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("plans"); s.add_argument("--round_dir", required=True); s.add_argument("--traj", required=True)
    s = sub.add_parser("eval"); s.add_argument("files", nargs="+")
    a = ap.parse_args(); {"plans": cmd_plans, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
