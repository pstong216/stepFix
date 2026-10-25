"""Oracle critical step t* for failed ALFWorld trajectories (PDDL-planner recoverability), in parallel.
Binary search over prefix length (recoverability is monotone: validated in test_oracle_attribution.py); --linear verifies.
    python scripts/alfworld_oracle_label.py --traj reports/alfworld/student3b_train.jsonl --n 200 --procs 16 --out reports/alfworld/oracle_student3b_train.jsonl
"""
import argparse, json, os, random, signal, sys, time
from multiprocessing import Pool
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("ERROR_ATTR_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def label(args):
    ep_id, game_file, actions, budget, linear = args
    from error_attr.attribution.oracle_alfworld import is_recoverable_from, find_critical_step, progress_trace
    t0 = time.time()
    def _alarm(signum, frame): raise TimeoutError("oracle timeout")
    signal.signal(signal.SIGALRM, _alarm); signal.alarm(900)
    try:
        if linear == "progress":
            r = progress_trace(game_file, actions, budget)
            return {"id": ep_id, **r, "n_actions": len(actions), "secs": round(time.time() - t0), "mode": "progress"}
        if linear:
            r = find_critical_step(game_file, actions, total_step_budget=budget, verify_monotonicity=True)
            return {"id": ep_id, "t_star": r["t_star"], "trace": r["recoverable_prefix_lengths"], "n_actions": len(actions), "secs": round(time.time() - t0), "mode": "linear"}
        n = len(actions)
        if is_recoverable_from(game_file, actions, budget):
            return {"id": ep_id, "t_star": None, "n_actions": n, "secs": round(time.time() - t0), "mode": "binary", "note": "recoverable after last action (budget exhaustion)"}
        lo, hi = 0, n  # invariant: rec(lo) True (empty prefix; checked below), rec(hi) False
        if not is_recoverable_from(game_file, actions[:0], budget):
            return {"id": ep_id, "t_star": -1, "n_actions": n, "secs": round(time.time() - t0), "mode": "binary", "note": "unsolvable from start within budget"}
        checks = 2
        while hi - lo > 1:
            mid = (lo + hi) // 2; checks += 1
            if is_recoverable_from(game_file, actions[:mid], budget): lo = mid
            else: hi = mid
        return {"id": ep_id, "t_star": lo, "n_actions": n, "secs": round(time.time() - t0), "mode": "binary", "checks": checks}
    except Exception as e:  # noqa: BLE001
        return {"id": ep_id, "error": repr(e)[:300], "n_actions": len(actions), "secs": round(time.time() - t0)}
    finally:
        signal.alarm(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--procs", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--linear", action="store_true"); ap.add_argument("--ids", default=None)
    ap.add_argument("--progress", action="store_true", help="progress oracle (plan-length trace, first wasted action) instead of t*")
    a = ap.parse_args()
    eps = [json.loads(l) for l in open(a.traj)]
    fails = [e for e in eps if not e["won"]]
    if a.ids: keep = set(json.load(open(a.ids))); fails = [e for e in fails if e["id"] in keep]
    done = set()
    if Path(a.out).exists(): done = {json.loads(l)["id"] for l in open(a.out)}
    random.Random(a.seed).shuffle(fails)
    todo = [e for e in fails if e["id"] not in done][: max(0, a.n - len(done))]
    print(f"{len(eps)} episodes, {len(fails)} failures, {len(done)} labelled, {len(todo)} to do, procs {a.procs}", flush=True)
    jobs = [(e["id"], e["game_file"], [s["action"] for s in e["steps"]], e["budget"], "progress" if a.progress else a.linear) for e in todo]
    t0 = time.time()
    with Pool(a.procs) as pool, open(a.out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(label, jobs)):
            f.write(json.dumps(r) + "\n"); f.flush()
            if (i + 1) % 10 == 0: print(f"[{i + 1}/{len(jobs)}] {(time.time() - t0) / (i + 1):.0f}s/traj (wall/n)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
