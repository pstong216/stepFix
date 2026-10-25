"""Localisability read-out for one policy's ALFWorld failures: budget exhaustion, both oracles (point of no return t*,
first harmful action), and Model A vs each. Usage:
    python scripts/alfworld_localizability_report.py --traj X.jsonl --progress P.jsonl [--oracle O.jsonl] [--attr name=A.jsonl ...]
"""
import argparse, json, random
from collections import Counter
from pathlib import Path


def load(p): return {json.loads(l)["id"]: json.loads(l) for l in open(p)} if p and Path(p).exists() else {}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--traj", required=True); ap.add_argument("--progress", default=None); ap.add_argument("--oracle", default=None); ap.add_argument("--attr", nargs="*", default=[])
    a = ap.parse_args()
    eps = {json.loads(l)["id"]: json.loads(l) for l in open(a.traj)}
    fails = {i: e for i, e in eps.items() if not e["won"]}
    n = len(eps); nf = len(fails)
    print(f"{n} episodes, success {1 - nf / n:.3f}, {nf} failures; failures hitting the budget {sum(e['n_steps'] >= e['budget'] for e in fails.values())}/{nf}; mean failure length {sum(e['n_steps'] for e in fails.values()) / max(1, nf):.1f}")
    prog = {i: r for i, r in load(a.progress).items() if "error" not in r}; orc = {i: r for i, r in load(a.oracle).items() if "error" not in r}
    if prog:
        reg = [r["t_first_regress"] for r in prog.values() if r.get("t_first_regress") is not None]
        waste = [r["t_first_waste"] for r in prog.values() if r.get("t_first_waste") is not None]
        rel = [r["t_first_regress"] / max(1, r["n_actions"]) for r in prog.values() if r.get("t_first_regress") is not None]
        print(f"progress oracle on {len(prog)} failures: first harmful action defined {len(reg)} ({len(reg) / len(prog):.1%}), mean step {sum(reg) / max(1, len(reg)):.1f} (relative {sum(rel) / max(1, len(rel)):.2f}); "
              f"first wasted action mean {sum(waste) / max(1, len(waste)):.1f}; initial plan length mean {sum(r['plan_len'][0] for r in prog.values()) / len(prog):.1f}; "
              f"final distance mean {sum(r['plan_len'][-1] for r in prog.values()) / len(prog):.1f}")
        print("  first-harmful histogram:", sorted(Counter(min(t, 30) for t in reg).items())[:16])
    if orc:
        ts = [r["t_star"] for r in orc.values() if isinstance(r.get("t_star"), int) and r["t_star"] >= 0]
        print(f"point-of-no-return oracle on {len(orc)} failures: t* defined {len(ts)}, mean step {sum(ts) / max(1, len(ts)):.1f} (relative {sum(r['t_star'] / max(1, r['n_actions']) for r in orc.values() if isinstance(r.get('t_star'), int) and r['t_star'] >= 0) / max(1, len(ts)):.2f}); "
              f"recoverable-throughout {sum(1 for r in orc.values() if r.get('t_star') is None)}")
        both = [(prog[i]["t_first_regress"], orc[i]["t_star"]) for i in orc if i in prog and prog[i].get("t_first_regress") is not None and isinstance(orc[i].get("t_star"), int) and orc[i]["t_star"] >= 0]
        if both: print(f"  gap point-of-no-return - first harmful: mean {sum(s - h for h, s in both) / len(both):.1f} steps (n={len(both)})")
    rng = random.Random(0)
    for spec in a.attr:
        name, path = spec.split("=", 1); A = load(path)
        if not A: print(f"Model A [{name}]: no predictions"); continue
        th = {i: r["t_hat"] for i, r in A.items() if r.get("t_hat") is not None}
        print(f"Model A [{name}]: {len(A)} predictions, parseable {len(th) / len(A):.1%}, mean relative position {sum(th[i] / max(1, A[i]['n_actions']) for i in th) / max(1, len(th)):.2f}, modes {Counter((r['pred'] or {}).get('error_mode') for r in A.values()).most_common(4)}")
        for key, src in [("first harmful (t_first_regress)", {i: r["t_first_regress"] for i, r in prog.items() if r.get("t_first_regress") is not None}), ("point of no return (t*)", {i: r["t_star"] for i, r in orc.items() if isinstance(r.get("t_star"), int) and r["t_star"] >= 0})]:
            pairs = [(th[i], src[i]) for i in th if i in src]
            if not pairs: continue
            rb = [(rng.randrange(A[i]["n_actions"]), src[i]) for i in th if i in src]
            f = lambda P, d: sum(abs(h - t) <= d for h, t in P) / len(P)
            print(f"  vs {key}: n={len(pairs)} exact {f(pairs, 0):.1%} within1 {f(pairs, 1):.1%} within2 {f(pairs, 2):.1%} | random exact {f(rb, 0):.1%} within2 {f(rb, 2):.1%} | mean |diff| {sum(abs(h - t) for h, t in pairs) / len(pairs):.1f}; t_hat<=target {sum(h <= t for h, t in pairs) / len(pairs):.0%}")


if __name__ == "__main__":
    main()
