"""Step A read-out: student success, failure structure (oracle t*), and Model A localisation vs oracle on OUR student's failures.
    python scripts/alfworld_step_a_report.py --traj reports/alfworld/student3b_train.jsonl --oracle reports/alfworld/oracle_student3b_train.jsonl \
        --attr full=reports/alfworld/attr7bfull_student3b_train.jsonl no_obs=reports/alfworld/attr7bnoobs_student3b_train.jsonl
"""
import argparse, json
from collections import Counter
from pathlib import Path


def load(p): return [json.loads(l) for l in open(p)] if p and Path(p).exists() else []


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--traj", required=True); ap.add_argument("--oracle", default=None); ap.add_argument("--attr", nargs="*", default=[])
    a = ap.parse_args()
    eps = load(a.traj); n = len(eps); won = sum(e["won"] for e in eps)
    print(f"episodes {n}, success {won}/{n} = {won / max(1, n):.3f}; mean steps {sum(e['n_steps'] for e in eps) / max(1, n):.1f}")
    by_type = Counter(); by_type_w = Counter()
    for e in eps:
        tt = Path(e["game_file"]).parent.parent.name.split("-")[0]; by_type[tt] += 1; by_type_w[tt] += e["won"]
    print("  by task type: " + ", ".join(f"{k} {by_type_w[k]}/{v}={by_type_w[k] / v:.2f}" for k, v in sorted(by_type.items())))
    fails = {e["id"]: e for e in eps if not e["won"]}
    print(f"failures {len(fails)}: steps used {sum(e['n_steps'] for e in fails.values()) / max(1, len(fails)):.1f} (budget {eps[0]['budget'] if eps else '?'}), "
          f"hit-budget share {sum(e['n_steps'] >= e['budget'] for e in fails.values()) / max(1, len(fails)):.2f}")
    orc = {r["id"]: r for r in load(a.oracle)}
    if orc:
        ok = [r for r in orc.values() if "error" not in r]; err = len(orc) - len(ok)
        none = sum(1 for r in ok if r["t_star"] is None); neg = sum(1 for r in ok if r["t_star"] == -1)
        ts = [r["t_star"] for r in ok if isinstance(r["t_star"], int) and r["t_star"] >= 0]
        print(f"\noracle: {len(ok)} labelled ({err} errors); recoverable-throughout (pure budget exhaustion) {none} ({none / max(1, len(ok)):.1%}); unsolvable-from-start {neg}; "
              f"decisive step exists {len(ts)} ({len(ts) / max(1, len(ok)):.1%}); mean secs/traj {sum(r['secs'] for r in ok) / max(1, len(ok)):.0f}")
        if ts:
            rel = [r["t_star"] / max(1, r["n_actions"]) for r in ok if isinstance(r["t_star"], int) and r["t_star"] >= 0]
            print(f"  t* position: mean {sum(ts) / len(ts):.1f} (0-based), relative {sum(rel) / len(rel):.2f} of trajectory; histogram by tercile "
                  f"early {sum(x < 1 / 3 for x in rel)} / mid {sum(1 / 3 <= x < 2 / 3 for x in rel)} / late {sum(x >= 2 / 3 for x in rel)}")
    for spec in a.attr:
        name, path = spec.split("=", 1); preds = {r["id"]: r for r in load(path)}
        if not preds: print(f"\nModel A [{name}]: no predictions yet"); continue
        parsed = sum(1 for r in preds.values() if r["t_hat"] is not None)
        print(f"\nModel A [{name}]: {len(preds)} predictions, parseable {parsed / len(preds):.1%}")
        pos = [r["t_hat"] / max(1, r["n_actions"]) for r in preds.values() if r["t_hat"] is not None]
        if pos: print(f"  t_hat relative position mean {sum(pos) / len(pos):.2f}; terciles early {sum(x < 1 / 3 for x in pos)} / mid {sum(1 / 3 <= x < 2 / 3 for x in pos)} / late {sum(x >= 2 / 3 for x in pos)}")
        modes = Counter((r["pred"] or {}).get("error_mode") for r in preds.values()); print(f"  modes: {modes.most_common(6)}")
        if orc:
            both = [(preds[i]["t_hat"], orc[i]["t_star"]) for i in preds if i in orc and isinstance(orc[i].get("t_star"), int) and orc[i]["t_star"] >= 0 and preds[i]["t_hat"] is not None]
            if both:
                ex = sum(h == s for h, s in both); w1 = sum(abs(h - s) <= 1 for h, s in both); w2 = sum(abs(h - s) <= 2 for h, s in both)
                before = sum(h < s for h, s in both); after = sum(h > s for h, s in both)
                print(f"  vs oracle (n={len(both)}): exact {ex / len(both):.1%}, within 1 {w1 / len(both):.1%}, within 2 {w2 / len(both):.1%}; t_hat before t* {before}, after {after}; "
                      f"mean |diff| {sum(abs(h - s) for h, s in both) / len(both):.1f}")
                import random
                rng = random.Random(0); base = [abs(rng.randrange(preds[i]['n_actions']) - orc[i]['t_star']) for i in preds if i in orc and isinstance(orc[i].get('t_star'), int) and orc[i]['t_star'] >= 0]
                print(f"  random-step baseline: exact {sum(d == 0 for d in base) / len(base):.1%}, within 1 {sum(d <= 1 for d in base) / len(base):.1%}, within 2 {sum(d <= 2 for d in base) / len(base):.1%}")


if __name__ == "__main__":
    main()
