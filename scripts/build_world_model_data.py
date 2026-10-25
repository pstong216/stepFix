"""Phase-0 data for an action-conditioned world model of the student (ALFWorld). Three heads, one row = (trajectory
source, id, t, action, label); histories are rebuilt at training time with history_at(ep, t) and the action at t.
  dyn  -- next observation after (h_t, a_t): every logged transition (student-family trajectories, any round)
  sub  -- did the student complete its current subgoal within H steps after (h_t, a_t)?  Milestones are read off the
          environment feedback (pick up the goal object -> transform / lamp -> place), on-trajectory actions only
  find -- during search (go to / open while not holding the goal object): will the goal object be visible next?
  q    -- does the round's start student win with a greedy continuation (cap 20) after (h_t, a)? From verification
          logs (teacher alternatives) and counterfactual logs (the student's original action), i.e. mostly OFF-trajectory
Split by game file (hash), so test games never appear in any training row of any head.
"""
import argparse, collections, glob, hashlib, json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_pointer_action_types import classify_full, seen_objects, typ
from alfworld_relaxed_verify import parse_goal


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def is_test(game_file, frac):
    return int(hashlib.md5(game_file.split("json_2.1.1/")[-1].encode()).hexdigest(), 16) % 1000 < frac * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_glob", default="reports/alfworld/student3b_low100*train_b50.jsonl")
    ap.add_argument("--arms_dir", default="reports/alfworld/opd/student3b_low100")
    ap.add_argument("--out", default="data/world_model")
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--H", type=int, default=10)
    ap.add_argument("--max_dyn", type=int, default=40000)
    ap.add_argument("--max_sub", type=int, default=30000)
    a = ap.parse_args(); rng = random.Random(0); out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    src_of, eps = {}, {}
    for f in sorted(glob.glob(str(ROOT / a.traj_glob))):
        if f.endswith(".release.jsonl"): continue
        rel = str(Path(f).relative_to(ROOT))
        for e in load_jsonl(f):
            if e["id"] in eps: continue
            eps[e["id"]] = e; src_of[e["id"]] = rel
    print(f"{len(eps)} trajectories from {len(set(src_of.values()))} files", flush=True)
    rows = collections.defaultdict(list)
    for i, e in eps.items():
        sp = "test" if is_test(e["game_file"], a.test_frac) else "train"
        lab, mh = classify_full(e); n = len(e["steps"]); obj = parse_goal(e["goal"])["obj"]
        for t, s in enumerate(e["steps"]):
            rows[("dyn", sp)].append({"src": src_of[i], "id": i, "t": t, "action": s["action"], "target": s["next_obs"][:400]})
            if lab[t][1] in ("explore_found", "explore_miss", "revisit", "back_to_seen_obj", "explore_lamp"):
                vis = int(any(typ(x) == obj for x in seen_objects(s["next_obs"])))
                rows[("find", sp)].append({"src": src_of[i], "id": i, "t": t, "action": s["action"], "label": vis})
            m0 = mh[t][0]; later = [m for (_, m) in mh[t: t + a.H]]
            done = bool(later) and max(later) > m0
            if not done and t + a.H > n and e["won"]: done = True          # won before the horizon ran out
            if not done and t + a.H > n and not e["won"] and n < e["budget"]: continue   # censored: episode cut early
            rows[("sub", sp)].append({"src": src_of[i], "id": i, "t": t, "action": s["action"], "label": int(done), "won": e["won"]})
    seen = set()
    for f in sorted(glob.glob(str(ROOT / a.arms_dir / "*/round*/*.jsonl"))):
        name = Path(f).name; rnd = Path(f).parent.name; arm = Path(f).parent.parent.name
        if name in ("examples.jsonl", "skipped.jsonl"):
            for r in load_jsonl(f):
                for act, won in (r.get("alternatives") or {}).items():
                    k = (r["id"], r["t"], act, rnd)
                    if r["id"] not in eps or k in seen: continue
                    seen.add(k); e = eps[r["id"]]; sp = "test" if is_test(e["game_file"], a.test_frac) else "train"
                    rows[("q", sp)].append({"src": src_of[r["id"]], "id": r["id"], "t": r["t"], "action": act, "label": int(bool(won)),
                                            "orig": act == e["steps"][r["t"]]["action"], "round": rnd, "arm": arm, "kind": "teacher_alt"})
        elif name == "counterfactual.jsonl":
            for r in load_jsonl(f):
                k = (r["id"], r["t"], r["a_minus"], rnd)
                if r["id"] not in eps or k in seen: continue
                seen.add(k); e = eps[r["id"]]; sp = "test" if is_test(e["game_file"], a.test_frac) else "train"
                rows[("q", sp)].append({"src": src_of[r["id"]], "id": r["id"], "t": r["t"], "action": r["a_minus"], "label": int(bool(r["a_minus_won"])),
                                        "orig": True, "round": rnd, "arm": arm, "kind": "student_orig"})
    caps = {"dyn": a.max_dyn, "sub": a.max_sub, "find": a.max_sub}
    for (head, sp), rs in sorted(rows.items()):
        rng.shuffle(rs)
        if head in caps and sp == "train": rs = rs[: caps[head]]
        if head in caps and sp == "test": rs = rs[: caps[head] // 5]
        with open(out / f"{head}_{sp}.jsonl", "w") as f:
            for r in rs: f.write(json.dumps(r) + "\n")
        pos = sum(r.get("label", 0) for r in rs)
        print(f"{head:4s} {sp:5s} {len(rs):6d} rows" + (f"  positive {pos / len(rs):.1%}" if head != "dyn" else ""), flush=True)


if __name__ == "__main__":
    main()
