"""Self-generated error-step labels for failed trajectories (no planner, no human, no external model):
for each of the first --k steps of a failure, score the student's own top-2 alternatives (plain prompt, a- excluded) and execute
each alternative once in the environment; the taken action's one-step evidence comes from the logged next observation.
  label t* = first step where (best alternative env_score) - (taken action env_score) >= --margin   (a clearly better move existed)
  fallback  = the step with the largest margin (if no step reaches the threshold).
  score   -- GPU: student top-2 at each (id, t) -> <out_dir>/cands.jsonl          (resumable)
  onestep -- CPU: one-step execution of every candidate -> <out_dir>/onestep.jsonl (resumable, --limit chunks)
  label   -- CPU: labels.jsonl + steps.jsonl (pointer schema) + a summary
"""
import argparse, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from alfworld_attr_opd import history_at, load_jsonl


def ap_common(s):
    s.add_argument("--traj", required=True); s.add_argument("--ids", required=True, help="jsonl of {id} (failures to label)")
    s.add_argument("--out_dir", required=True); s.add_argument("--k", type=int, default=8)


def cmd_score(a):
    import torch
    from alfworld_opd_plus import load_policy
    from self_evolve import softmax
    eps = {e["id"]: e for e in load_jsonl(ROOT / a.traj)}; ids = [r["id"] for r in load_jsonl(ROOT / a.ids)]
    out = ROOT / a.out_dir / "cands.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
    done = {(json.loads(l)["id"], json.loads(l)["t"]) for l in open(out)} if out.exists() else set()
    todo = [(i, t) for i in ids for t in range(min(a.k, len(eps[i]["steps"]))) if (i, t) not in done]; print(f"{len(todo)} (id, t) states", flush=True)
    tok, model, pol = load_policy(a.student, str(ROOT / a.adapter) if a.adapter else None); model.eval()
    with open(out, "a") as f, torch.no_grad():
        for n, (i, t) in enumerate(todo):
            e = eps[i]; lp = pol.score_admissible_commands_cached(history_at(e, t)); am = e["steps"][t]["action"]
            cands = [c for c in sorted(lp, key=lambda c: -lp[c]) if c != am][:2]
            f.write(json.dumps({"id": i, "t": t, "a_minus": am, "cands": cands, "p": {c: softmax(lp)[c] for c in cands}}) + "\n")
            if n % 100 == 0: f.flush(); print(f"[{n + 1}/{len(todo)}]", flush=True)
    print("DONE score", flush=True)


def cmd_onestep(a):
    from error_attr.env.alfworld_env import make_env
    eps = {e["id"]: e for e in load_jsonl(ROOT / a.traj)}; rows = load_jsonl(ROOT / a.out_dir / "cands.jsonl")
    out = ROOT / a.out_dir / "onestep.jsonl"; done = {(json.loads(l)["id"], json.loads(l)["t"], json.loads(l)["cand"]) for l in open(out)} if out.exists() else set()
    todo = [(r, c) for r in rows for c in r["cands"] if (r["id"], r["t"], c) not in done]
    if a.shard: todo = todo[a.shard[0]::a.shard[1]]
    print(f"{len(todo)} candidate executions", flush=True)
    if not todo: print("DONE onestep", flush=True); return
    with open(out, "a") as f:
        for n, (r, c) in enumerate(todo):
            if a.limit and n >= a.limit: print(f"chunk limit {a.limit}", flush=True); return
            e = eps[r["id"]]; env = make_env(e["game_file"], e["budget"] + 2)
            try:
                obs, info = env.reset()
                for s in e["steps"][: r["t"]]: obs, _, _, info = env.step([s["action"]])
                o, _, dn, info = env.step([c])
                f.write(json.dumps({"id": r["id"], "t": r["t"], "cand": c, "obs_plus": o[0], "won_now": bool(info["won"][0])}) + "\n"); f.flush()
            finally:
                env.close()
    print("DONE onestep", flush=True)


def cmd_label(a):
    from env_evidence_filter import evidence
    from alfworld_relaxed_verify import parse_goal
    eps = {e["id"]: e for e in load_jsonl(ROOT / a.traj)}; rows = load_jsonl(ROOT / a.out_dir / "cands.jsonl")
    one = {(r["id"], r["t"], r["cand"]): r for r in load_jsonl(ROOT / a.out_dir / "onestep.jsonl")}
    sc = lambda ev: 2.0 * ev["goal_revealed"] + 2.0 * ev["subgoal"] + min(ev["n_new"], 5) / 5.0 - 2.0 * ev["no_info"]
    per = {}
    for r in rows:
        e = eps[r["id"]]; obj = parse_goal(e["goal"])["obj"]; t = r["t"]
        taken = sc(evidence(e, t, e["steps"][t]["action"], e["steps"][t]["next_obs"], obj))
        alts = [(sc(evidence(e, t, c, one[(r["id"], t, c)]["obs_plus"], obj)), c) for c in r["cands"] if (r["id"], t, c) in one]
        if not alts: continue
        best = max(alts); per.setdefault(r["id"], {})[t] = {"taken": taken, "best": best[0], "best_cand": best[1], "margin": best[0] - taken}
    n_thr = 0
    with open(ROOT / a.out_dir / "labels.jsonl", "w") as f, open(ROOT / a.out_dir / "steps.jsonl", "w") as g:
        for i, d in per.items():
            ts = sorted(d); hit = [t for t in ts if d[t]["margin"] >= a.margin]
            t_star = hit[0] if hit else max(ts, key=lambda t: d[t]["margin"]); n_thr += bool(hit)
            taken = eps[i]["steps"][t_star]["action"]; best = d[t_star]["best_cand"]; ft, bt = taken.split()[0], best.split()[0]
            typ = "wrong_target" if ft == bt and ft in ("go", "take", "open", "close", "examine") else "wrong_value" if ft in ("put", "move") and bt in ("put", "move") else "skip" if ft in ("look", "inventory", "help", "examine") else "wrong_op"
            f.write(json.dumps({"id": i, "t_star": t_star, "type": typ, "reached_threshold": bool(hit), "margin": d[t_star]["margin"], "best_cand": best, "per_step": {str(t): d[t] for t in ts}}) + "\n")
            g.write(json.dumps({"id": i, "steps": [t_star], "mode": None}) + "\n")
    import statistics as st
    ls = [json.loads(l)["t_star"] for l in open(ROOT / a.out_dir / "labels.jsonl")]
    print(f"=== env labels: {len(ls)} failures | threshold reached {n_thr} ({n_thr / len(ls):.0%}) | t* mean {st.mean(ls):.1f} median {st.median(ls):.0f} | t0 {sum(t == 0 for t in ls) / len(ls):.2f} t<=2 {sum(t <= 2 for t in ls) / len(ls):.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); ap_common(s); s.add_argument("--student", default="Qwen/Qwen2.5-3B-Instruct"); s.add_argument("--adapter", default="reports/alfworld/student3b_low100/final")
    o = sub.add_parser("onestep"); ap_common(o); o.add_argument("--limit", type=int, default=0); o.add_argument("--shard", type=int, nargs=2, default=None)
    l = sub.add_parser("label"); ap_common(l); l.add_argument("--margin", type=float, default=2.0)
    a = ap.parse_args(); {"score": cmd_score, "onestep": cmd_onestep, "label": cmd_label}[a.cmd](a)


if __name__ == "__main__":
    main()
