"""Teacher quality at the attributed state: does the privileged teacher's top action make progress (shorten the planner's
shortest completion) where the student's action lengthened it? Compares hint vs no-hint teacher, the student's original
action, and a random admissible action. CPU (PDDL planner) -- run after alfworld_attr_opd.py teacher.
    python scripts/alfworld_teacher_check.py --traj reports/alfworld/student3b_train.jsonl --targets A=hint.jsonl B=nohint.jsonl --out report.json
"""
import argparse, json, os, random, sys
from multiprocessing import Pool
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("ERROR_ATTR_ROOT", str(ROOT)); sys.path.insert(0, str(ROOT / "src"))


def plan_len_after(game_file, actions, budget):
    """Planner shortest-completion length after executing `actions` (None if the episode ended)."""
    from error_attr.attribution.oracle_alfworld import _make_single_env
    env = _make_single_env(game_file, max_episode_steps=budget + 2)
    try:
        obs, info = env.reset()
        for a in actions:
            obs, r, d, info = env.step([a])
            if d[0]: return 0 if info["won"][0] else None
        plan = info["extra.expert_plan"][0]
        return len(plan) if isinstance(plan, list) else (1 if plan else 0)
    finally:
        env.close()


def one(job):
    ep, t, cands = job
    prefix = [s["action"] for s in ep["steps"][:t]]
    base = plan_len_after(ep["game_file"], prefix, ep["budget"])
    out = {"id": ep["id"], "t": t, "before": base}
    for name, act in cands.items():
        after = plan_len_after(ep["game_file"], prefix + [act], ep["budget"])
        out[name] = {"action": act, "after": after, "progress": (after is not None and base is not None and after < base), "regress": (after is not None and base is not None and after > base)}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--traj", required=True); ap.add_argument("--targets", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    eps = {json.loads(l)["id"]: json.loads(l) for l in open(a.traj)}
    tg = {}
    for spec in a.targets:
        name, path = spec.split("=", 1)
        for r in map(json.loads, open(path)): tg.setdefault((r["id"], r["t"]), {})[name] = r
    rng = random.Random(0); jobs = []
    for (i, t), d in tg.items():
        ep = eps[i]; cands = {"student": ep["steps"][t]["action"]}
        adm = ep["steps"][t]["admissible"]; cands["random"] = rng.choice(adm)
        for name, r in d.items():
            cands[f"teacher_{name}"] = max(r["admissible"], key=lambda c: r["teacher_logprobs"][c])
        jobs.append((ep, t, cands))
    with Pool(a.procs) as pool: res = list(pool.imap_unordered(one, jobs))
    names = sorted({k for r in res for k in r if isinstance(r[k], dict)})
    summary = {}
    for n in names:
        rows = [r[n] for r in res if n in r and r["before"] is not None]
        summary[n] = {"n": len(rows), "progress": sum(x["progress"] for x in rows) / max(1, len(rows)), "regress": sum(x["regress"] for x in rows) / max(1, len(rows))}
    json.dump({"summary": summary, "rows": res}, open(a.out, "w"), indent=1)
    for n, s in summary.items(): print(f"{n:22s} n={s['n']:4d} progress {s['progress']:.1%} regress {s['regress']:.1%}")


if __name__ == "__main__":
    main()
