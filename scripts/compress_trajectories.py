"""Detour pruning of the student's own won ALFWorld trajectories, verified by exact environment replay (the env is deterministic).
Per won episode:
  pruned -- keep only progress steps (classifier cat D, plus the first-visit step at which the goal object was found); replay in
            the env; if the pruned action list does not win, greedy fallback: drop removable steps one at a time, keeping a drop
            only if the replayed sequence still wins.
  keep1  -- the pruned path plus the first removed detour block (contiguous dropped steps) restored in place, i.e. one detour and
            the exit from it; also replay-verified.
Rows use the b50 schema (steps obs/admissible/action/next_obs, observations from the replay) so alfworld_sft.py and
alfworld_opd_plus.py (kind sft) can consume them.  Usage:
    python scripts/compress_trajectories.py --traj reports/alfworld/student3b_low100_train_b50.jsonl --out reports/alfworld/compress_traj/wins.jsonl --workers 8
"""
import argparse, json, os, signal, sys, time
from multiprocessing import Pool
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("ERROR_ATTR_ROOT", str(ROOT)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from analyze_pointer_action_types import classify_full

KEEP_LABELS = {"explore_found"}            # first visit/open where the goal object was revealed: needed even though it is "exploration"
DROP_LABELS = {"put_back_wrong_obj"}       # classified D but only undoes a removable mistake


def removable(cat, lab):
    return lab in DROP_LABELS or (cat in ("A", "B", "C") and lab not in KEEP_LABELS)


def replay(game_file, actions, budget):
    from error_attr.env.alfworld_env import make_env
    env = make_env(game_file, budget + 1)
    try:
        obs, info = env.reset(); steps = []
        for act in actions:
            adm = list(info["admissible_commands"][0])
            if act not in adm: return False, steps
            nobs, rw, dn, info = env.step([act])
            steps.append({"obs": obs[0], "admissible": adm, "action": act, "next_obs": nobs[0]}); obs = nobs
            if dn[0]: return bool(info["won"][0]), steps
        return False, steps
    finally:
        env.close()


def seq(acts, mask):
    """Kept actions; a 'go to X' is dropped when the agent is already at X (not admissible in ALFWorld after pruning the detour)."""
    out = []; loc = None
    for a, m in zip(acts, mask):
        if not m: continue
        if a.startswith("go to "):
            if a[6:] == loc: continue
            loc = a[6:]
        out.append(a)
    return out


def process(e):
    def _alarm(s, f): raise TimeoutError()
    signal.signal(signal.SIGALRM, _alarm); signal.alarm(300)
    try:
        labs, _ = classify_full(e); acts = [s["action"] for s in e["steps"]]; n = len(acts)
        rem = [removable(c, l) for c, l, *_ in labs]
        # re-taking the goal object right after the agent itself misplaced it is part of that detour
        misplaced = None
        for i, (c, l, *_) in enumerate(labs):
            a = acts[i].lower()
            if l in ("misplace_goal_obj", "place_untransformed"): misplaced = a.split(" to ")[0].split(" in/on ")[0].replace("move ", "").replace("put ", "")
            elif l == "take_goal_obj" and misplaced and a.startswith(f"take {misplaced} from"): rem[i] = True; misplaced = None
            elif l == "take_goal_obj": misplaced = None
        mask = [not r for r in rem]; n_replays = 0; fallback = False
        won, steps = replay(e["game_file"], seq(acts, mask), e["budget"]); n_replays += 1
        if not won:
            # greedy fallback: first whole contiguous removable blocks, then single steps; keep a drop only if the replay still wins
            fallback = True; mask = [True] * n; won, steps = replay(e["game_file"], acts, e["budget"]); n_replays += 1
            blocks = []; i = 0
            while i < n:
                if rem[i]:
                    j = i
                    while j < n and rem[j]: j += 1
                    blocks.append(list(range(i, j))); i = j
                else: i += 1
            trials = [b for b in blocks if len(b) > 1] + [[i] for i in range(n) if rem[i]]
            for idx in trials:
                if not all(mask[i] for i in idx): continue
                for i in idx: mask[i] = False
                w, st = replay(e["game_file"], seq(acts, mask), e["budget"]); n_replays += 1
                if w: won, steps = w, st
                else:
                    for i in idx: mask[i] = True
        base = {k: e[k] for k in ("game_file", "goal", "initial_obs", "budget", "split") if k in e}
        rows = [{**base, "id": e["id"] + "__pruned", "src_id": e["id"], "variant": "pruned", "won": won, "n_steps": len(steps), "steps": steps,
                 "orig_len": n, "n_removed": n - sum(mask), "fallback": fallback, "labels": [l for _, l, *_ in labs], "kept": mask}]
        # keep1: restore the first dropped block
        blocks = []; i = 0
        while i < n:
            if not mask[i]:
                j = i
                while j < n and not mask[j]: j += 1
                blocks.append((i, j)); i = j
            else: i += 1
        k1 = None
        for (i, j) in blocks:
            m2 = list(mask); m2[i:j] = [True] * (j - i)
            w, st = replay(e["game_file"], seq(acts, m2), e["budget"]); n_replays += 1
            if w: k1 = {**base, "id": e["id"] + "__keep1", "src_id": e["id"], "variant": "keep1", "won": True, "n_steps": len(st), "steps": st,
                        "orig_len": n, "n_removed": n - sum(m2), "detour_block": [i, j], "kept": m2}; break
        if k1: rows.append(k1)
        rows[0]["n_replays"] = n_replays
        return rows
    except TimeoutError:
        return [{"id": e["id"] + "__pruned", "src_id": e["id"], "variant": "pruned", "won": False, "timeout": True, "steps": [], "n_steps": 0, "orig_len": len(e["steps"])}]
    finally:
        signal.alarm(0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--traj", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=0); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    eps = [json.loads(l) for l in open(ROOT / a.traj)]; eps = [e for e in eps if e["won"]]
    if a.max: eps = eps[: a.max]
    done = {json.loads(l)["src_id"] for l in open(out)} if out.exists() else set()
    todo = [e for e in eps if e["id"] not in done]; print(f"{len(eps)} won episodes, {len(todo)} to do", flush=True)
    t0 = time.time(); k = 0
    with open(out, "a") as f, Pool(a.workers) as pool:
        for rows in pool.imap_unordered(process, todo, chunksize=1):
            for r in rows: f.write(json.dumps(r) + "\n")
            f.flush(); k += 1
            if k % 25 == 0: print(f"[{k}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
    rows = [json.loads(l) for l in open(out)]; pr = [r for r in rows if r["variant"] == "pruned"]; k1 = [r for r in rows if r["variant"] == "keep1"]
    ok = [r for r in pr if r["won"]]
    print(f"pruned: {len(ok)}/{len(pr)} win on replay | fallback {sum(r.get('fallback', False) for r in pr)} | timeouts {sum(r.get('timeout', False) for r in pr)}")
    if ok: print(f"mean length orig {sum(r['orig_len'] for r in ok) / len(ok):.1f} -> pruned {sum(r['n_steps'] for r in ok) / len(ok):.1f}; keep1 {len(k1)} episodes, mean length {sum(r['n_steps'] for r in k1) / max(1, len(k1)):.1f}")
    print("DONE compress", flush=True)


if __name__ == "__main__":
    main()
