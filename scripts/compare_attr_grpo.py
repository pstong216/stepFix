"""Compare Attr-GRPO arms (reports/attr_grpo_3b/<arm>/) on the 238 held-out test tasks.
Greedy: exact paired McNemar vs base and vs uniform. Sampled mean@k: paired Wilcoxon + mean diff.
    python scripts/compare_attr_grpo.py [--root reports/attr_grpo_3b] [--arms uniform attr random last]
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
from scipy.stats import binomtest, wilcoxon


def load(p):
    return {r["id"]: r for r in map(json.loads, open(p))} if Path(p).exists() else None


def mcnemar(a, b):
    """a, b: {id: bool}. Returns (n_b_only, n_a_only, p) for b better than a."""
    ids = sorted(set(a) & set(b)); x = sum(1 for i in ids if b[i] and not a[i]); y = sum(1 for i in ids if a[i] and not b[i])
    p = binomtest(x, x + y, 0.5).pvalue if x + y else 1.0
    return x, y, p, len(ids)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="reports/attr_grpo_3b"); ap.add_argument("--arms", nargs="+", default=["uniform", "attr", "random", "last", "attr_committed", "attr_soft", "random_soft"])
    a = ap.parse_args(); root = Path(a.root)
    base_g = base_s = None
    for arm in a.arms:  # the arm that ran --full_eval_at_start holds the base records
        base_g = base_g or load(root / arm / "eval_step0_full_greedy.jsonl"); base_s = base_s or load(root / arm / "eval_step0_full_sampled.jsonl")
    rows = {}
    for arm in a.arms:
        g = load(root / arm / "eval_final_greedy.jsonl"); s = load(root / arm / "eval_final_sampled.jsonl")
        log = [json.loads(l) for l in open(root / arm / "train_log.jsonl")] if (root / arm / "train_log.jsonl").exists() else []
        steps = list({r["step"]: r for r in log if isinstance(r.get("step"), int) and "loss" in r}.values())  # last record per step (resumes)
        rows[arm] = {"greedy": g, "sampled": s, "curve": sorted({r["step"]: (r["step"], r["eval_acc"]) for r in log if "eval_acc" in r}.values()),
                     "n_steps": len(steps), "mean_kl": sum(r["kl"] for r in steps) / max(1, len(steps)),
                     "masked_share": sum(r["masked_share"] for r in steps) / max(1, len(steps)),
                     "groups": sum(r["informative_groups"] for r in steps), "fallback": sum(r.get("attr_fallback", 0) for r in steps),
                     "hours": (steps[-1]["elapsed_s"] / 3600) if steps else 0}
    print(f"{'arm':8s} {'steps':>5s} {'groups':>6s} {'hours':>5s} {'KL':>7s} {'mask':>5s} | {'greedy':>6s} {'vs base (b/a,p)':>20s} {'vs unif (b/a,p)':>20s} | {'mean@k':>6s} {'vs base':>16s}")
    if base_g: print(f"{'base':8s} {'':5s} {'':6s} {'':5s} {'':7s} {'':5s} | {sum(r['acc'] for r in base_g.values())/len(base_g):6.3f} {'':20s} {'':20s} | " + (f"{sum(r['acc'] for r in base_s.values())/len(base_s):6.3f}" if base_s else ""))
    unif = rows.get("uniform", {}).get("greedy")
    for arm, r in rows.items():
        g = r["greedy"]
        if not g: print(f"{arm:8s} {r['n_steps']:5d} {r['groups']:6d} {r['hours']:5.1f} {r['mean_kl']:7.4f} {r['masked_share']:5.2f} | (no final eval yet) curve={r['curve']}"); continue
        gb = {i: rr["acc"] > 0.5 for i, rr in g.items()}; acc = sum(rr["acc"] for rr in g.values()) / len(g)
        vb = mcnemar({i: rr["acc"] > 0.5 for i, rr in base_g.items()}, gb) if base_g else None
        vu = mcnemar({i: rr["acc"] > 0.5 for i, rr in unif.items()}, gb) if unif and arm != "uniform" else None
        s = r["sampled"]; sacc = sum(rr["acc"] for rr in s.values()) / len(s) if s else None
        if s and base_s:
            ids = sorted(set(s) & set(base_s)); d = [s[i]["acc"] - base_s[i]["acc"] for i in ids]
            pw = wilcoxon(d).pvalue if any(d) else 1.0; svb = f"{sum(d)/len(d):+.3f} p={pw:.3g}"
        else: svb = ""
        print(f"{arm:8s} {r['n_steps']:5d} {r['groups']:6d} {r['hours']:5.1f} {r['mean_kl']:7.4f} {r['masked_share']:5.2f} | {acc:6.3f} "
              f"{(f'{vb[0]}/{vb[1]} p={vb[2]:.3g}' if vb else ''):>20s} {(f'{vu[0]}/{vu[1]} p={vu[2]:.3g}' if vu else ''):>20s} | {(f'{sacc:6.3f}' if sacc is not None else ''):>6s} {svb:>16s}")
    # no-answer shares (team final None) -- format/length failures vs committed wrong answers
    print("\nfailure composition on the 238 test tasks: no-answer share (greedy | sampled) and committed-wrong share")
    def comp(g):
        fin = [f for r in g.values() for f in r["finals"]]; cor = [c for r in g.values() for c in r["correct"]]
        n = len(fin); na = sum(1 for f in fin if f is None); cw = sum(1 for f, c in zip(fin, cor) if f is not None and not c)
        c = sum(cor)
        return f"noans {na/n:.3f} wrong {cw/n:.3f} correct {c/n:.3f} | P(correct|answered) {c/max(1,c+cw):.3f} (n_ans={c+cw})"
    if base_g: print(f"  {'base':8s} greedy: {comp(base_g)}" + (f" | sampled: {comp(base_s)}" if base_s else ""))
    for arm, r in rows.items():
        if r["greedy"]: print(f"  {arm:8s} greedy: {comp(r['greedy'])}" + (f" | sampled: {comp(r['sampled'])}" if r["sampled"] else ""))
    # per-benchmark greedy
    print("\nper-benchmark greedy accuracy")
    bms = sorted({rr["benchmark"] for r in rows.values() if r["greedy"] for rr in r["greedy"].values()} | ({rr["benchmark"] for rr in base_g.values()} if base_g else set()))
    hdr = f"{'bench':10s}" + "".join(f"{k:>9s}" for k in (["base"] if base_g else []) + [k for k in rows if rows[k]["greedy"]]); print(hdr)
    for bm in bms:
        line = f"{bm:10s}"
        for g in ([base_g] if base_g else []) + [rows[k]["greedy"] for k in rows if rows[k]["greedy"]]:
            v = [rr["acc"] for rr in g.values() if rr["benchmark"] == bm]; line += f"{(sum(v)/len(v) if v else float('nan')):9.3f}"
        print(line + f"   (n={sum(1 for rr in (base_g or next(r['greedy'] for r in rows.values() if r['greedy'])).values() if rr['benchmark']==bm)})")
    print("\ntraining curves (step, subset eval acc):")
    for arm, r in rows.items(): print(f"  {arm:8s} {r['curve']}")
    print("\ntraining dynamics per 12-step window: mean reward in informative groups | informative groups per task tried | mean KL | mask share")
    for arm in a.arms:
        lp = root / arm / "train_log.jsonl"
        if not lp.exists(): continue
        steps = list({r["step"]: r for r in map(json.loads, open(lp)) if isinstance(r.get("step"), int) and "loss" in r}.values())
        cells = []
        for w0 in range(1, 49, 12):
            w = [r for r in steps if w0 <= r["step"] < w0 + 12]
            if not w: continue
            n_roll = sum(r["informative_groups"] for r in w)
            mr = sum(r["mean_reward"] * r["informative_groups"] for r in w) / max(1, n_roll)
            inf = sum(r["informative_groups"] for r in w) / max(1, sum(r["tasks_tried"] for r in w))
            cells.append(f"[{w0}-{w0+11}] R={mr:.3f} inf={inf:.2f} KL={sum(r['kl'] for r in w)/len(w):.4f} mask={sum(r['masked_share'] for r in w)/len(w):.2f}")
        print(f"  {arm:8s} " + " | ".join(cells))


if __name__ == "__main__":
    main()
