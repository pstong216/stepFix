"""Oct 10 2026: summary of the EXTENDED ScienceWorld eval (test variations beyond the first 30 per task, seed-0 shuffle; ~1,168 episodes)
alone and combined with the 848 main eval.  Wins, mean score (negatives clipped to 0), score<0 count, exact McNemar (two-sided binomial).
   python scripts/sciworld_eval_ext_summary.py [--pairs A:B ...] [models ...]
"""
import argparse, glob, json, math
from pathlib import Path

S = "reports/sciworld"
MAIN = {  # model -> (848 test file, 848 dev file)
    "R0": (f"{S}/stepfix/r0/eval_test.jsonl", f"{S}/refix/eval_dev_safe.jsonl"),
    "r1b": (f"{S}/stepfix/r1b/eval_stepfix/eval_test.jsonl", f"{S}/stepfix/r1b/eval_stepfix/eval_dev.jsonl"),
    "r1b_random": (f"{S}/stepfix/r1b/eval_random/eval_test.jsonl", f"{S}/stepfix/r1b/eval_random/eval_dev.jsonl"),
    "r2": (f"{S}/stepfix/r2/eval_stepfix/eval_test.jsonl", f"{S}/stepfix/r2/eval_stepfix/eval_dev.jsonl"),
    "r3": (f"{S}/stepfix/r3/eval_stepfix/eval_test.jsonl", f"{S}/stepfix/r3/eval_stepfix/eval_dev.jsonl"),
    **{m: (f"{S}/stepfix/{m}/eval/eval_test.jsonl", f"{S}/stepfix/{m}/eval/eval_dev.jsonl") for m in ["r2h", "r3h", "r2f", "r3f"]},
    **{f"{b}_r{k}": (f"{S}/baselines/{b}/r{k}/eval/eval_test.jsonl", f"{S}/baselines/{b}/r{k}/eval/eval_dev.jsonl")
       for b in ["rftm", "rftfull"] for k in (1, 2, 3)},
}
ORDER = ["R0", "r1b", "r2h", "r3h", "rftm_r1", "rftm_r2", "rftm_r3", "rftfull_r1", "rftfull_r2", "rftfull_r3", "r1b_random", "r2", "r3", "r2f", "r3f"]


def load(files):
    d = {}
    for f in files:
        for l in open(f):
            try: e = json.loads(l)
            except Exception: continue
            d.setdefault((e["split"], e["task"], e["variation"]), (bool(e["won"]), float(e["final_score"])))
    return d


def ext(m): return load(sorted(glob.glob(f"{S}/eval_ext/{m}/eval_test_ext_s*.jsonl")))


def stats(d):
    n = len(d); w = sum(v[0] for v in d.values())
    return f"{w}/{n} ({100 * w / max(n, 1):.1f}%) | mean {sum(max(v[1], 0) for v in d.values()) / max(n, 1):.2f} | score<0 {sum(v[1] < 0 for v in d.values())}"


def mcnemar(a, b):
    ks = sorted(set(a) & set(b)); oa = sum(a[k][0] and not b[k][0] for k in ks); ob = sum(b[k][0] and not a[k][0] for k in ks); n = oa + ob
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(oa, ob) + 1)) / 2 ** n) if n else 1.0
    return f"only-A {oa} / only-B {ob}, p = {p:.3g} (n paired {len(ks)})"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("models", nargs="*"); ap.add_argument("--pairs", nargs="*", default=[]); a = ap.parse_args()
    E, C = {}, {}
    for m in a.models or ORDER:
        E[m] = ext(m)
        if not E[m]: print(f"{m}: no extended results yet"); continue
        C[m] = {**load(MAIN[m]), **E[m]} if m in MAIN else None
        print(f"{m:11s} EXT  {stats(E[m])}")
        if C[m] is not None: print(f"{'':11s} 848  {stats(load(MAIN[m]))}\n{'':11s} ALL  {stats(C[m])}")
    for pr in a.pairs:
        x, y = pr.split(":")
        for k in (x, y):
            if k not in E: E[k] = ext(k); C[k] = {**load(MAIN[k]), **E[k]}
        print(f"McNemar A={x} vs B={y}:  EXT {mcnemar(E[x], E[y])}  |  ALL {mcnemar(C[x], C[y])}")
