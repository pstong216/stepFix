"""Round read-out for ALFWorld Attr-OPD: per arm, unseen success vs the student's own baseline (paired McNemar), plus cost
(teacher calls, env steps, verified share). Usage:
    python scripts/alfworld_opd_report.py --tag student3b_low100 --round round1 [--arms attr oracle random last uniform]
"""
import argparse, json
from pathlib import Path
from scipy.stats import binomtest


def load_eval(p):
    return {json.loads(l)["game_file"]: json.loads(l)["won"] for l in open(p)} if Path(p).exists() else None


def mcnemar(base, arm):
    ids = sorted(set(base) & set(arm)); b = sum(1 for i in ids if arm[i] and not base[i]); a = sum(1 for i in ids if base[i] and not arm[i])
    return b, a, (binomtest(b, a + b, 0.5).pvalue if a + b else 1.0), len(ids)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="student3b_low100"); ap.add_argument("--round", default="round1")
    ap.add_argument("--arms", nargs="+", default=["attr", "oracle", "random", "last", "uniform"]); ap.add_argument("--teacher_eval", default="reports/alfworld/eval_unseen_sft3b_b50.jsonl")
    a = ap.parse_args(); R = Path("reports/alfworld")
    base = load_eval(R / f"eval_unseen_{a.tag}.jsonl"); teacher = load_eval(a.teacher_eval)
    print(f"student baseline ({a.tag}): {sum(base.values())}/{len(base)} = {sum(base.values()) / len(base):.3f}" + (f" | teacher: {sum(teacher.values()) / len(teacher):.3f}" if teacher else ""))
    print(f"{'arm':8s} {'unseen':>7s} {'vs base b/a p':>18s} {'examples':>8s} {'verified':>8s} {'teacher calls':>13s} {'env steps':>9s} {'hours':>5s}")
    rows = {}
    for arm in a.arms:
        d = R / "opd" / a.tag / arm; ev = load_eval(d / f"eval_unseen_{a.round}.jsonl")
        lg = [json.loads(l) for l in open(d / a.round / "distill_log.jsonl")] if (d / a.round / "distill_log.jsonl").exists() else []
        fin = next((r["final"] for r in lg if "final" in r), None) or (lg[-1] if lg else {})
        if not ev:
            print(f"{arm:8s} {'(running)':>7s} {'':18s} {fin.get('examples', ''):>8} {fin.get('verified_used', ''):>8} {fin.get('teacher_calls', ''):>13} {fin.get('env_steps', ''):>9}"); continue
        b, aa, p, n = mcnemar(base, ev); acc = sum(ev.values()) / len(ev); rows[arm] = ev
        flag = "" if len(ev) >= len(base) else f" (PARTIAL {len(ev)}/{len(base)})"
        print(f"{arm:8s} {acc:7.3f}{flag} {f'{b}/{aa} p={p:.3g}':>18s} {fin.get('examples', ''):>8} {fin.get('verified_used', ''):>8} {fin.get('teacher_calls', ''):>13} {fin.get('env_steps', ''):>9} {fin.get('elapsed_s', 0) / 3600:5.1f}")
    if "attr" in rows:
        for other in [x for x in ("random", "oracle", "uniform", "last") if x in rows]:
            b, aa, p, n = mcnemar(rows[other], rows["attr"]); print(f"attr vs {other}: attr-only {b} / {other}-only {aa}, p={p:.3g}")


if __name__ == "__main__":
    main()
