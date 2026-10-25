"""Pooled read-out for ALFWorld Attr-OPD: every (arm, round) with an unseen eval, plus the extended splits
(valid_seen 140, held-out train games 1500-1999 = 500) when present. Paired McNemar (exact binomial) vs the
low-data student on the same games, and vs a chosen reference adapter. Also prints cost from distill_log.
Usage:
    python scripts/alfworld_pooled_report.py [--tag student3b_low100] [--ref attr_nofb:round1] [--markdown]
"""
import argparse, json
from pathlib import Path
from scipy.stats import binomtest

R = Path("reports/alfworld")
# (arm, round) -> extended-eval tag under reports/alfworld/ext/
EXT = {("attr", "round1"): "opd_attr", ("attr", "round2"): "opd_attr_r2",
       ("oracle", "round1"): "opd_oracle", ("oracle", "round2"): "opd_oracle_r2", ("oracle", "round3"): "opd_oracle_r3",
       ("random", "round1"): "opd_random", ("last", "round1"): "opd_last", ("uniform", "round1"): "opd_uniform",
       ("attr_nofb", "round1"): "opd_attr_nofb", ("attr_nofb", "round2"): "opd_attr_nofb_r2", ("attr_nofb", "round3"): "opd_attr_nofb_r3",
       ("attr_nofb", "round1_k3"): "opd_attr_nofb_k3", ("attr_nofb", "round1_seed1"): "opd_attr_nofb_seed1", ("attr_nofb", "round1_cv"): "opd_attr_nofb_cv",
       ("oracle_nofb", "round1"): "opd_oracle_nofb", ("oracle_nofb", "round2"): "opd_oracle_nofb_r2",
       ("attralf_nofb", "round1"): "opd_attralf_nofb", ("const1_nofb", "round1"): "opd_const1_nofb", ("oracle_nofb_p1", "round1"): "opd_oracle_nofb_p1", ("oracle_nofb_p2", "round1"): "opd_oracle_nofb_p2", ("oracle_nofb_m2", "round1"): "opd_oracle_nofb_m2", ("oracle_nofb_p5", "round1"): "opd_oracle_nofb_p5", ("oracle_nofb_m5", "round1"): "opd_oracle_nofb_m5", ("const1_r2", "round2"): "opd_const1_r2", ("tstar_nofb", "round1"): "opd_tstar_nofb", ("tstar_nofb_p1", "round1"): "opd_tstar_nofb_p1", ("tstar_nofb_p2", "round1"): "opd_tstar_nofb_p2", ("attr_nofb_p1", "round1"): "opd_attr_nofb_p1", ("attr_nofb_p2", "round1"): "opd_attr_nofb_p2", ("attralf_nofb", "round2"): "opd_attralf_nofb_r2", ("attralf_nofb", "round3"): "opd_attralf_nofb_r3"}
ORDER = ["attr", "oracle", "random", "last", "uniform", "attr_w1", "attr_nofb", "oracle_nofb", "attralf_nofb", "const1_nofb", "oracle_nofb_p1", "oracle_nofb_m2", "oracle_nofb_p2", "oracle_nofb_p5", "oracle_nofb_m5", "const1_r2", "tstar_nofb", "tstar_nofb_p1", "tstar_nofb_p2", "attr_nofb_p1", "attr_nofb_p2"]


def load(p):
    p = Path(p)
    if not p.exists():
        return {}
    return {(r["game_file"]): bool(r["won"]) for r in map(json.loads, open(p))}


def mcnemar(base, arm):
    ids = sorted(set(base) & set(arm)); b = sum(1 for i in ids if arm[i] and not base[i]); a = sum(1 for i in ids if base[i] and not arm[i])
    return b, a, (binomtest(b, a + b, 0.5).pvalue if a + b else 1.0), len(ids)


def splits(tag, arm=None, rnd=None):
    """dict split -> {game: won}. Student when arm is None."""
    if arm is None:
        return {"unseen": load(R / f"eval_unseen_{tag}.jsonl"), "seen": load(R / "ext" / "eval_seen_student_low100.jsonl"),
                "heldout": load(R / "ext" / "eval_heldout_student_low100.jsonl")}
    d = R / "opd" / tag / arm; et = EXT.get((arm, rnd))
    return {"unseen": load(d / f"eval_unseen_{rnd}.jsonl"),
            "seen": load(R / "ext" / f"eval_seen_{et}.jsonl") if et else {}, "heldout": load(R / "ext" / f"eval_heldout_{et}.jsonl") if et else {}}


def pooled(s):
    out = {}
    for k, v in s.items():
        for g, w in v.items():
            out[(k, g)] = w
    return out


def cost(tag, arm, rnd):
    p = R / "opd" / tag / arm / rnd / "distill_log.jsonl"
    if not p.exists():
        return {}
    lg = [json.loads(l) for l in open(p)]
    fins = [r["final"] for r in lg if "final" in r]  # one per chunk (chunked distillation): sum the counts, keep max teacher_calls
    if not fins:
        return lg[-1] if lg else {}
    out = {k: sum(f.get(k, 0) for f in fins) for k in ("examples", "verified_used", "unverified_used", "skipped_no_success", "env_steps", "elapsed_s")}
    out["teacher_calls"] = max(f.get("teacher_calls", 0) for f in fins); return out


def fmt(s):
    return f"{sum(s.values())}/{len(s)} = {100 * sum(s.values()) / len(s):.1f}%" if s else "--"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="student3b_low100"); ap.add_argument("--ref", default="attr_nofb:round1")
    ap.add_argument("--markdown", action="store_true"); a = ap.parse_args()
    stu = splits(a.tag); stu_p = pooled(stu)
    rows = []
    for arm in ORDER:
        d = R / "opd" / a.tag / arm
        if not d.exists():
            continue
        for ev in sorted(d.glob("eval_unseen_*.jsonl")):
            if ev.name.endswith(".release.jsonl") or "partial" in ev.name:
                continue
            rnd = ev.name[len("eval_unseen_"):-len(".jsonl")]
            s = splits(a.tag, arm, rnd); c = cost(a.tag, arm, rnd)
            rows.append((arm, rnd, s, c))
    ref_arm, ref_rnd = a.ref.split(":"); ref = pooled(splits(a.tag, ref_arm, ref_rnd))
    hdr = ["arm", "round", "unseen(134)", "seen(140)", "heldout(500)", "pooled", "vs student b/a p", f"vs {a.ref} b/a p", "verified ex", "teacher calls"]
    lines = []
    lines.append(["student", "--", fmt(stu["unseen"]), fmt(stu["seen"]), fmt(stu["heldout"]), fmt(stu_p), "--", "--", "--", "--"])
    for arm, rnd, s, c in rows:
        p = pooled(s); b, aa, pv, n = mcnemar(stu_p, p); b2, a2, pv2, n2 = mcnemar(ref, p)
        lines.append([arm, rnd, fmt(s["unseen"]), fmt(s["seen"]), fmt(s["heldout"]), fmt(p) + ("" if n == len(stu_p) else f" (n={n})"),
                      f"{b}/{aa} p={pv:.2g}", f"{b2}/{a2} p={pv2:.2g}" if n2 else "--", str(c.get("verified_used", "")), str(c.get("teacher_calls", ""))])
    if a.markdown:
        print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
        for l in lines:
            print("| " + " | ".join(l) + " |")
    else:
        w = [max(len(x[i]) for x in [hdr] + lines) for i in range(len(hdr))]
        for l in [hdr] + lines:
            print("  ".join(x.ljust(w[i]) for i, x in enumerate(l)))


if __name__ == "__main__":
    main()
