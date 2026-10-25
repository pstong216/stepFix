"""Planner-oracle progress traces (plan length after every action prefix; t_first_regress) for a set of trajectories,
in parallel on CPU.  Same format as reports/alfworld/progress_*.jsonl (t_first_regress = index of the first STATE whose plan
length exceeds the previous one, i.e. the regressing ACTION is t_first_regress - 1).
    python scripts/oracle_progress.py --traj T.jsonl --ids ids.jsonl --out out.jsonl --workers 8
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("ERROR_ATTR_ROOT", str(ROOT)); sys.path.insert(0, str(ROOT / "src"))


def work(args):
    i, game_file, actions, budget = args
    from error_attr.attribution.oracle_alfworld import progress_trace
    t0 = time.time()
    try:
        r = progress_trace(game_file, actions, budget); r["id"] = i; r["secs"] = int(time.time() - t0); r["mode"] = "progress"; return r
    except Exception as e:
        return {"id": i, "error": str(e)[:200]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--traj", required=True); ap.add_argument("--ids", required=True); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); eps = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / a.traj)}; ids = [json.loads(l)["id"] for l in open(ROOT / a.ids)]
    out = ROOT / a.out; done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    jobs = [(i, eps[i]["game_file"], [s["action"] for s in eps[i]["steps"]], eps[i]["budget"]) for i in ids if i in eps and i not in done]
    print(f"{len(jobs)} trajectories, {a.workers} workers", flush=True)
    with open(out, "a") as f, Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(work, jobs)):
            f.write(json.dumps(r) + "\n"); f.flush()
            if k % 25 == 0: print(f"[{k + 1}/{len(jobs)}]", flush=True)
    rows = [json.loads(l) for l in open(out)]; ok = [r for r in rows if r.get("t_first_regress") is not None]
    print(f"DONE oracle: {len(ok)} with t_first_regress of {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
