"""Where do step pointers land? Rule-based action categories read off the trajectory log only (no planner, no rollout).
  A explore      -- still searching for the goal object; first visit/open of a location (A_found: X appears; A_miss: not)
  B redundant    -- revisit of an already-seen location with nothing to do there, re-open, close, look/inventory/examine,
                    "Nothing happens", toggling the lamp without the object, wandering while holding the right object
  C regress      -- takes a wrong object, puts the goal object somewhere that is not the goal receptacle, transforms the
                    wrong object, places an untransformed object; C_unrec = the trajectory never recovers afterwards
  D progress     -- everything goal-directed: go to/open the goal receptacle or the needed appliance, go back to where the
                    goal object was seen, take the goal object, transform it, place it, put a wrong object back down
For each round-1 arm (same 319 failures, same teacher and recipe) report the category mix of its pointers and the
verification success rate per category; baselines = all steps of the failures and all steps of won trajectories.
"""
import argparse, collections, json, random, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from alfworld_relaxed_verify import parse_goal

DEVICE = {"clean": "sinkbasin", "heat": "microwave", "cool": "fridge"}
ARMS = [("const0_nofb", "always step 0"), ("const1_nofb", "always step 1"), ("progress_nofb", "M5 progress est."),
        ("attr_nofb", "Model A"), ("oracle_nofb", "planner t_first_regress"), ("tstar_nofb", "planner t*")]


def typ(x):
    return re.sub(r"\s*\d+$", "", x.strip())


def seen_objects(o):
    m = re.search(r"you see (.*?)\.?$", o.lower())
    if not m or "nothing" in m.group(1): return set()
    return {re.sub(r"^(a|an|the)\s+", "", p.strip()) for p in re.split(r",\s*(?:and\s+)?|\s+and\s+", m.group(1)) if p.strip()}


def classify(ep):
    """Per-step category and fine label for one trajectory."""
    return classify_full(ep)[0]


def classify_full(ep):
    """(per-step [cat, label, (recovered)], per-step (milestone level before, after))."""
    G = parse_goal(ep["goal"]); obj, rec, tf, lamp = G["obj"], G["rec"], G["tf"], G["lamp"]
    need = 2 if G["two"] else 1; per = 3 if (tf or lamp) else 2
    visited, opened, seen_at = set(), set(), collections.defaultdict(set)   # loc -> object instances seen there
    holding = None; transformed = set(); placed = set(); out = []; m_hist = []

    def m_now():
        cur = 0
        if holding and typ(holding) == obj and holding not in placed:
            cur = 1 + (1 if ((tf and holding in transformed) or (lamp and holding in transformed)) else 0)
        return min(len(placed), need) * per + (0 if len(placed) >= need else cur)

    for s in ep["steps"]:
        a = s["action"].strip().lower(); o = s["next_obs"].strip(); ol = o.lower(); verb = a.split()[0] if a else ""
        holding_right = bool(holding and typ(holding) == obj and holding not in placed)
        ready = holding_right and (not tf or holding in transformed) and (not lamp)
        phase = "find" if not holding_right else ("transform" if ((tf and holding not in transformed) or (lamp and holding not in transformed)) else "place")
        m_before = m_now(); cat, lab = "D", "other"
        goal_here = lambda loc: any(typ(x) == obj and x not in placed and x != holding for x in seen_at[loc])
        if ol.startswith("nothing happens") or verb in ("look", "inventory", "examine", "help"):
            cat, lab = "B", "noop"
        elif verb == "go":
            loc = a[len("go to "):]; t = typ(loc); first = loc not in visited
            if phase == "find":
                if goal_here(loc) and not first: cat, lab = "D", "back_to_seen_obj"
                elif lamp and t in ("desklamp", "floorlamp"): cat, lab = "D", "to_lamp"
                elif first: cat, lab = "A", "explore_found" if any(typ(x) == obj for x in seen_objects(o)) else "explore_miss"
                else: cat, lab = "B", "revisit"
            elif phase == "transform":
                dev = DEVICE.get(tf) if tf else None
                if (dev and t == dev) or (lamp and "lamp" in t): cat, lab = "D", "to_device"
                elif lamp and first: cat, lab = "A", "explore_lamp"
                else: cat, lab = "B", "detour_holding"
            else:
                if t == rec: cat, lab = "D", "to_receptacle"
                else: cat, lab = "B", "detour_holding"
            visited.add(loc); seen_at[loc] |= seen_objects(o)
        elif verb == "open":
            loc = a[len("open "):]; t = typ(loc)
            if loc in opened: cat, lab = "B", "reopen"
            elif phase == "find": cat, lab = "A", "explore_found" if any(typ(x) == obj for x in seen_objects(o)) else "explore_miss"
            elif (phase == "place" and t == rec) or (phase == "transform" and tf and t == DEVICE.get(tf)): cat, lab = "D", "open_needed"
            else: cat, lab = "B", "open_unneeded"
            opened.add(loc); seen_at[loc] |= seen_objects(o)
        elif verb == "close":
            cat, lab = "B", "close"
        elif verb == "take":
            mm = re.match(r"take (.+?) from (.+)$", a); x = mm.group(1) if mm else a[5:]
            if ol.startswith("you pick up"):
                if x in placed: cat, lab = "C", "unplace_goal_obj"; placed.discard(x)
                elif typ(x) == obj: cat, lab = "D", "take_goal_obj"
                else: cat, lab = "C", "take_wrong_obj"
                holding = x
        elif verb == "move" or verb == "put":
            mm = re.match(r"(?:move|put) (.+?) (?:to|in/on) (.+)$", a); x, loc = (mm.group(1), mm.group(2)) if mm else (holding, "")
            if ol.startswith("you move") or ol.startswith("you put"):
                if x and typ(x) == obj and x not in placed:
                    if typ(loc) == rec and (not tf or x in transformed): cat, lab = "D", "place_goal"; placed.add(x)
                    elif typ(loc) == rec: cat, lab = "C", "place_untransformed"
                    else: cat, lab = "C", "misplace_goal_obj"
                else: cat, lab = "D", "put_back_wrong_obj"
                holding = None; seen_at[loc].add(x or "")
        elif verb in ("clean", "heat", "cool"):
            mm = re.match(rf"{verb} (.+?) with (.+)$", a); x = mm.group(1) if mm else ""
            if ol.startswith(f"you {verb}"):
                if typ(x) == obj and tf == verb: cat, lab = "D", "transform_goal"; transformed.add(x)
                else: cat, lab = "C", "transform_wrong"
        elif verb == "use":
            if lamp and holding_right: cat, lab = "D", "use_lamp_with_obj"; transformed.add(holding)
            else: cat, lab = "B", "use_lamp_empty"
        out.append([cat, lab]); m_hist.append((m_before, m_now()))
    # recovery for regressions: take_wrong -> later picks up the goal object; other C -> progress later exceeds the level
    for i, (cat, lab) in enumerate(out):
        if cat != "C": continue
        later = [m for (_, m) in m_hist[i + 1:]]
        target = m_hist[i][0] + (1 if lab == "take_wrong_obj" else 0)
        out[i].append("recovered" if later and max(later) >= max(target, 1) else "unrecovered")
    return out, m_hist


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="reports/alfworld/student3b_low100_train_b50.jsonl")
    ap.add_argument("--arms_dir", default="reports/alfworld/opd/student3b_low100")
    ap.add_argument("--out", default="reports/alfworld/pointer_action_types")
    ap.add_argument("--audit_n", type=int, default=60)
    a = ap.parse_args(); out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    eps = {e["id"]: e for e in load_jsonl(ROOT / a.traj)}
    lab = {i: classify(e) for i, e in eps.items()}
    ids = [json.loads(l)["id"] for l in open(ROOT / a.arms_dir / "attr_nofb/steps.jsonl")]
    CATS = ["A", "B", "C", "D"]
    def mix(pairs):
        c = collections.Counter(); cu = 0
        for i, t in pairs:
            x = lab[i][t]; c[x[0]] += 1; cu += (x[0] == "C" and x[-1] == "unrecovered")
        n = sum(c.values()); return n, {k: c[k] / n for k in CATS}, cu / n
    rows = []
    base_fail = [(i, t) for i in ids for t in range(len(eps[i]["steps"]))]
    base_won = [(i, t) for i, e in eps.items() if e["won"] for t in range(len(e["steps"]))]
    for name, pairs in [("all steps of the 319 failures", base_fail), ("all steps of won trajectories", base_won)]:
        n, m, cu = mix(pairs); rows.append({"arm": name, "n": n, **m, "C_unrec": cu})
    per_cat_verify = {}
    for arm, desc in ARMS:
        d = ROOT / a.arms_dir / arm
        if not (d / "round1/examples.jsonl").exists(): continue
        ver = {(r["id"], r["t"]) for r in load_jsonl(d / "round1/examples.jsonl")}
        pairs = [(r["id"], t) for r in load_jsonl(d / "steps.jsonl") for t in r["steps"] if r["id"] in lab and t < len(lab[r["id"]])]
        n, m, cu = mix(pairs)
        fine = collections.Counter(lab[i][t][1] for i, t in pairs)
        vc = collections.defaultdict(lambda: [0, 0])
        for i, t in pairs:
            k = lab[i][t][0] + ("_unrec" if lab[i][t][0] == "C" and lab[i][t][-1] == "unrecovered" else ""); vc[k][1] += 1; vc[k][0] += (i, t) in ver
            vc["all"][1] += 1; vc["all"][0] += (i, t) in ver
        per_cat_verify[desc] = {k: v for k, v in vc.items()}
        rows.append({"arm": desc, "n": n, **m, "C_unrec": cu, "verified": len(ver & set(pairs)), "top_fine": fine.most_common(6)})
    # pooled verify rate by category over all arms (distinct states)
    pooled = collections.defaultdict(lambda: [0, 0]); seen = set()
    for arm, desc in ARMS:
        d = ROOT / a.arms_dir / arm
        if not (d / "round1/examples.jsonl").exists(): continue
        ver = {(r["id"], r["t"]) for r in load_jsonl(d / "round1/examples.jsonl")}
        for r in load_jsonl(d / "steps.jsonl"):
            for t in r["steps"]:
                if (r["id"], t) in seen or t >= len(lab[r["id"]]): continue
                seen.add((r["id"], t)); x = lab[r["id"]][t]; k = x[1]; pooled[k][1] += 1; pooled[k][0] += (r["id"], t) in ver
    json.dump({"rows": rows, "verify_by_cat": per_cat_verify, "verify_by_fine_label_pooled": pooled}, open(out / "summary.json", "w"), indent=1)
    print(f"{'pointer':32s} {'n':>5s} " + " ".join(f"{c:>6s}" for c in CATS) + f" {'C_unrec':>7s}")
    for r in rows:
        print(f"{r['arm']:32s} {r['n']:5d} " + " ".join(f"{100 * r[c]:5.1f}%" for c in CATS) + f" {100 * r['C_unrec']:6.1f}%")
    print("\nverification success by category (verified/n)")
    for desc, vc in per_cat_verify.items():
        print(f"{desc:32s} " + "  ".join(f"{k}:{v[0]}/{v[1]}={100 * v[0] / v[1]:.0f}%" for k, v in sorted(vc.items())))
    print("\npooled over arms, distinct states, by fine label")
    for k, v in sorted(pooled.items(), key=lambda kv: -kv[1][1]):
        print(f"  {k:22s} {v[0]:4d}/{v[1]:4d} = {100 * v[0] / v[1]:.0f}%")
    # audit sample: stratified by category
    rng = random.Random(0); allp = [(i, t) for i in ids for t in range(len(eps[i]["steps"]))]; rng.shuffle(allp)
    by = collections.defaultdict(list)
    for i, t in allp: by[lab[i][t][0]].append((i, t))
    with open(out / "audit_sample.jsonl", "w") as f:
        for c in CATS:
            for i, t in by[c][: a.audit_n // 4]:
                e = eps[i]; f.write(json.dumps({"id": i, "t": t, "goal": e["goal"], "cat": lab[i][t], "prev": [(s["action"], s["next_obs"][:150]) for s in e["steps"][max(0, t - 3):t]],
                                                "action": e["steps"][t]["action"], "next_obs": e["steps"][t]["next_obs"][:200]}) + "\n")
    print(f"\naudit sample -> {out / 'audit_sample.jsonl'}")


if __name__ == "__main__":
    main()
