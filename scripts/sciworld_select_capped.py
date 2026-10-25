"""Oct 10 23:00 (user-approved): R2 re-selection on an existing round dir (teacher rows already computed), training data only.
  corrections : env_top exactly as sciworld_stepfix.py select (best candidate with S > 0 and not no-info, global order by (S, teacher prob)),
                but each task type keeps at most cap = ceil(cap_mult x K / #types-with-failures) corrections; the freed slots go to the next
                states in the same global order (total stays K = round(rho x failures)).
  replay      : RETENTION replay -- n = round(replay_mult x K) steps split over task types in proportion to the round's training WINS
                (= what the start policy can already do; largest remainder), each quota filled with complete shortest wins of that type
                (skip look around / inventory / task), as replay_fw but weighted by wins instead of failures.
  usage: python scripts/sciworld_select_capped.py --dir reports/sciworld/stepfix/x2 --out reports/sciworld/stepfix/x2c --replay_mult 1.2"""
import argparse, glob, json, math, collections
from pathlib import Path

ROOT = Path("/users/8/pan00389/multi-agent/error-attribution")
L = lambda p: [json.loads(l) for l in open(p)]
ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--rho", type=float, default=0.2); ap.add_argument("--cap_mult", type=float, default=2.0); ap.add_argument("--replay_mult", type=float, default=1.2)
# Oct 10 23:40 x2d filters (user-approved): --min_s (confirmed-progress evidence), --skip_info (no look around / inventory / task / wait as a
# correction target, the replay SKIP rule), --commit_guard (an irreversible "focus on" commit is accepted only when it replaces a FATAL wrong
# commit, i.e. a- ended the episode with score < 0 at t; otherwise one-step success cannot tell an informed answer from a lucky guess)
ap.add_argument("--min_s", type=float, default=0.0); ap.add_argument("--skip_info", action="store_true"); ap.add_argument("--commit_guard", action="store_true")
a = ap.parse_args(); D = ROOT / a.dir; O = ROOT / a.out; O.mkdir(parents=True, exist_ok=True)
eps = {e["id"]: e for e in L(D / "traj.jsonl")}; fails = [x["id"] for x in L(D / "ids_fail.jsonl")]; nf = len(fails); K = max(1, round(a.rho * nf))
rows = {}
for p in sorted(glob.glob(str(D / "teacher" / "rows_s*.jsonl"))):
    for r in L(p): rows.setdefault(r["id"], r)
picks = []
for r in rows.values():
    if "error" in r: continue
    e_ = eps[r["id"]]; fatal = e_["steps"][r["t"]]["score"] < 0 and r["t"] == len(e_["steps"]) - 1
    def okc(c):
        if c["no_info"] or c["score"] <= 0 or c["score"] < a.min_s: return False
        w0 = c["cand"].strip().split()[0] if c["cand"].strip() else ""
        if a.skip_info and (c["cand"].strip() in ("look around", "inventory", "task") or w0.startswith("wait")): return False
        if a.commit_guard and c["cand"].strip().startswith("focus on") and not fatal:
            # Oct 11 00:30: mode file (read at run time) -- "strict" (x2d): no non-fatal commit; "confirmed": a non-fatal commit is kept
            # only if the environment's own score rose after it (score_after > score_before)
            mode = (open(ROOT / "reports/sciworld/stepfix/.commit_guard_mode").read().strip() if (ROOT / "reports/sciworld/stepfix/.commit_guard_mode").exists() else "strict")
            if mode == "strict" or c["score_after"] <= c["score_before"]: return False
        return True
    good = [c for c in r.get("cands", []) if okc(c)]
    if good: c = max(good, key=lambda c: (c["score"], c["prob"])); picks.append((r, c, (c["score"], c["prob"])))
picks.sort(key=lambda x: x[2], reverse=True)
types_f = sorted({eps[i]["task"] for i in fails}); cap = math.ceil(a.cap_mult * K / len(types_f))
sel, cnt, skipped = [], collections.Counter(), collections.Counter()
for r, c, _ in picks:
    if len(sel) >= K: break
    if cnt[r["task"]] >= cap: skipped[r["task"]] += 1; continue
    sel.append((r, c)); cnt[r["task"]] += 1
src = str((D / "traj.jsonl").relative_to(ROOT))
open(O / "targets_cap.jsonl", "w").writelines(json.dumps({"kind": "corr", "src": src, "id": r["id"], "t": r["t"], "action": c["cand"], "a_minus": r["a_minus"],
                                                         "S": c["score"], "prob": c["prob"]}) + "\n" for r, c in sel)
# retention replay
n = round(a.replay_mult * K); SKIP = ("look around", "inventory", "task")
wins = sorted([e for e in eps.values() if e["won"]], key=lambda e: (len(e["steps"]), e["id"]))
steps_of = lambda e: [t for t, s_ in enumerate(e["steps"]) if s_["action"].strip() not in SKIP]
capr = collections.Counter(); nw = collections.Counter()
for e in wins: capr[e["task"]] += len(steps_of(e)); nw[e["task"]] += 1
tot = sum(nw.values()); raw = {k: n * v / tot for k, v in nw.items()}; quota = {k: int(math.floor(v)) for k, v in raw.items()}
for k in sorted(raw, key=lambda k: (-(raw[k] - quota[k]), k))[: n - sum(quota.values())]: quota[k] += 1
eff = {k: min(quota[k], capr[k]) for k in quota}
rep = []
for k in sorted(eff):
    got = 0
    for e in [w for w in wins if w["task"] == k]:
        for t in steps_of(e):
            if got >= eff[k]: break
            rep.append({"kind": "replay", "src": src, "id": e["id"], "t": t, "action": e["steps"][t]["action"]}); got += 1
        if got >= eff[k]: break
open(O / "replay_ret.jsonl", "w").writelines(json.dumps(x) + "\n" for x in rep)
print(json.dumps({"K": K, "pool": len(picks), "types_with_failures": len(types_f), "cap_per_type": cap, "selected": len(sel),
                  "mean_S": round(sum(c["score"] for _, c in sel) / max(1, len(sel)), 3), "by_task": dict(sorted(cnt.items())),
                  "capped_out": dict(sorted(skipped.items())), "replay_n": len(rep), "replay_quota": dict(sorted(eff.items()))}))
