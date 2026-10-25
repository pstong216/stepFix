"""Self-evolving local correction, round 1 (3B student = its own teacher; the teacher reads the full failed trajectory).
  hindsight -- for every step of the given failures: the model's action distribution with the normal prompt (plain) and
               with the rest of the failed episode shown (hindsight); stores both (KV-cached scoring)
  choose    -- supervised step per failure for each step-selection rule:
               rand_early (random among the first 3 steps) | entropy (max plain entropy) | attr (Model A) |
               hkl (max KL(hindsight || plain): where knowing what happened later changes the model's choice most)
  targets   -- candidates at the chosen step = top-2 of the hindsight distribution excluding the logged action a-
  select    -- no rollout: one-step environment evidence + log contradiction, scored by the FROZEN round-1 env filter;
               best candidate per state, keep the top half of states -> offline training rows (kind=kl, one-hot)
"""
import argparse, json, math, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import os
D = ROOT / os.environ.get("SE_DIR", "reports/alfworld/self_evolve")
TRAJ = os.environ.get("SE_TRAJ", "reports/alfworld/student3b_low100_train_b50.jsonl")
MODEL = os.environ.get("SE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
STUDENT = os.environ.get("SE_ADAPTER", str(ROOT / "reports/alfworld/student3b_low100/final")) or None
ATTR = os.environ.get("SE_ATTR", "reports/alfworld/opd/student3b_low100/attr_nofb/steps.jsonl")


def L(p):
    return [json.loads(l) for l in open(p)]


class HindsightHistory:
    """Wraps a policy history: same prompt, plus what actually happened after step t in the failed episode."""
    def __init__(self, h, ep, t, max_future=25, obs_chars=160):
        self.steps = h.steps; self._h = h
        fut = ep["steps"][t:t + max_future]
        self._block = "\n".join(f"- step {t + i}: {s['action']} -> {s['next_obs'][:obs_chars]}" for i, s in enumerate(fut))
        self._n = len(ep["steps"])
    def render_prompt(self, max_history_steps=10):
        p = self._h.render_prompt(max_history_steps)
        note = ("Hindsight: this attempt FAILED. After this point the agent did the following "
                f"(episode length {self._n} steps, budget exhausted):\n{self._block}\n"
                "Using everything above, choose the best action for the CURRENT step.\n")
        return p.replace("Admissible actions:", note + "Admissible actions:", 1)


def softmax(lp):
    z = max(lp.values()); Z = z + math.log(sum(math.exp(v - z) for v in lp.values())); return {c: math.exp(v - Z) for c, v in lp.items()}


def cmd_hindsight(a):
    import torch
    from alfworld_attr_opd import history_at
    from alfworld_opd_plus import load_policy
    ids = [json.loads(l)["id"] for l in open(ROOT / a.ids)]; eps = {e["id"]: e for e in L(ROOT / TRAJ)}
    out = D / a.out; done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    tok, model, pol = load_policy(MODEL, STUDENT)
    with open(out, "a") as f, torch.no_grad():
        for k, i in enumerate(ids):
            if i in done: continue
            e = eps[i]; rec = []
            for t in range(len(e["steps"])):
                h = history_at(e, t)
                lp = pol.score_admissible_commands_cached(h); lh = pol.score_admissible_commands_cached(HindsightHistory(h, e, t))
                p, q = softmax(lp), softmax(lh)
                rec.append({"t": t, "plain": lp, "hind": lh, "H": -sum(v * math.log(v + 1e-12) for v in p.values()),
                            "KL_hp": sum(q[c] * (math.log(q[c] + 1e-12) - math.log(p.get(c, 1e-12) + 1e-12)) for c in q)})
            f.write(json.dumps({"id": i, "steps": rec}) + "\n"); f.flush(); print(f"[{k + 1}/{len(ids)}]", flush=True)
    print("DONE hindsight", flush=True)


SELFATTR_PROMPT = ("Below is the full log of an agent that FAILED a household task. Steps are numbered from 0.\n\n"
                   "Task: {goal}\n\n{log}\n\nAt which step did the agent make the first decisive mistake, i.e. the earliest action "
                   "after which the task went wrong? Answer with only the step number.")


def cmd_selfattr(a):
    """Self-attribution: the same model reads the whole failed log and names the first decisive mistake (no training)."""
    import re, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    ids = [json.loads(l)["id"] for l in open(ROOT / a.ids)]; eps = {e["id"]: e for e in L(ROOT / TRAJ)}
    tok = AutoTokenizer.from_pretrained(MODEL); model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0})
    if STUDENT: model = PeftModel.from_pretrained(model, STUDENT)
    model.eval()
    with open(D / "selfattr.jsonl", "w") as f, torch.no_grad():
        for i in ids:
            e = eps[i]; log = "\n".join(f"Step {t}: action: {s['action']} | observation: {s['next_obs'][:160]}" for t, s in enumerate(e["steps"]))
            msg = tok.apply_chat_template([{"role": "user", "content": SELFATTR_PROMPT.format(goal=e["goal"], log=log)}], tokenize=False, add_generation_prompt=True)
            x = tok(msg, return_tensors="pt", add_special_tokens=False).to(model.device)
            out = tok.decode(model.generate(**x, max_new_tokens=12, do_sample=False)[0, x.input_ids.shape[1]:], skip_special_tokens=True)
            m = re.search(r"\d+", out); t = min(int(m.group()), len(e["steps"]) - 1) if m else 0
            f.write(json.dumps({"id": i, "t_hat": t, "raw": out}) + "\n"); f.flush()
    print("DONE selfattr", flush=True)


def _arms(a, default):
    return a.arms.split(",") if getattr(a, "arms", None) else default


def cmd_hindsight_steps(a):
    """Plain + hindsight distributions only at given (id, step) pairs -> targets.jsonl in the steps file's directory."""
    import torch
    from alfworld_attr_opd import history_at
    from alfworld_opd_plus import load_policy
    eps = {e["id"]: e for e in L(ROOT / TRAJ)}; steps = L(ROOT / a.steps); out = (ROOT / a.steps).parent / "targets.jsonl"
    done = {(json.loads(l)["id"], json.loads(l)["t"]) for l in open(out)} if out.exists() else set()
    tok, model, pol = load_policy(MODEL, STUDENT)
    with open(out, "a") as f, torch.no_grad():
        for k, r in enumerate(steps):
            i, t = r["id"], r["steps"][0]
            if (i, t) in done: continue
            e = eps[i]; h = history_at(e, t)
            lp = pol.score_admissible_commands_cached(h); lh = pol.score_admissible_commands_cached(HindsightHistory(h, e, t))
            f.write(json.dumps({"id": i, "t": t, "a_minus": e["steps"][t]["action"], "admissible": list(lh), "teacher_logprobs": lh, "plain_logprobs": lp, "mode": None}) + "\n"); f.flush()
            if k % 50 == 0: print(f"[{k}/{len(steps)}]", flush=True)
    print("DONE hindsight_steps", flush=True)


def cmd_order(a):
    """Priority order for verification: states whose candidates survive the env veto first (by hindsight-teacher prob)."""
    from alfworld_relaxed_verify import parse_goal
    import env_evidence_filter as E
    d = (ROOT / a.steps).parent; eps = {e["id"]: e for e in L(ROOT / TRAJ)}
    one = {(r["id"], r["t"], r["cand"]): r["obs_plus"] for r in L(d / "onestep.jsonl")}; rows = []
    for T in L(d / "targets.jsonl"):
        e = eps[T["id"]]; t = T["t"]; obj = parse_goal(e["goal"])["obj"]; lp = T["teacher_logprobs"]
        cands = [c for c in sorted(lp, key=lambda c: -lp[c]) if c != T["a_minus"]][:2]; best = -1e9
        for c in cands:
            if (T["id"], t, c) in one:
                ev = E.evidence(e, t, c, one[(T["id"], t, c)], obj)
                if not ev["no_info"] and E.hindsight(e, t + 1, c, obj) != "contradicts": best = max(best, lp[c])
        rows.append((best, T))
    rows.sort(key=lambda x: -x[0])
    with open(d / "targets_ordered.jsonl", "w") as f:
        for b, T in rows: f.write(json.dumps({**T, "proxy_ok": b > -1e9}) + "\n")
    print(f"ordered {len(rows)} states; {sum(b > -1e9 for b, _ in rows)} pass the veto", flush=True)


def cmd_choose(a):
    H = {r["id"]: r["steps"] for p in sorted(D.glob("hindsight*.jsonl")) for r in L(p)}
    attr = {r["id"]: (r["steps"][0] if "steps" in r else r["t_hat"]) for r in L(ROOT / ATTR)}
    sa = {r["id"]: r["t_hat"] for r in L(D / "selfattr.jsonl")} if (D / "selfattr.jsonl").exists() else {}
    sp = {r["id"]: r["t_hat"] for r in L(D / "selfprog_pointer.jsonl")} if (D / "selfprog_pointer.jsonl").exists() else {}
    rng = random.Random(0)
    rules = {"rand_early": lambda i, s: rng.randrange(min(3, len(s))), "entropy": lambda i, s: max(range(len(s)), key=lambda t: s[t]["H"]),
             "attr": lambda i, s: attr[i], "hkl": lambda i, s: max(range(len(s)), key=lambda t: s[t]["KL_hp"])}
    if sa: rules["selfattr"] = lambda i, s: min(sa[i], len(s) - 1)
    if sp: rules["selfprog"] = lambda i, s: min(sp[i], len(s) - 1)   # self-trained attributor: progress estimator on own wins
    for name, fn in rules.items():
        if getattr(a, "arms", None) and name not in a.arms.split(","): continue
        (D / name / "round1").mkdir(parents=True, exist_ok=True); (D / name / "round1/examples.jsonl").touch()
        with open(D / name / "steps.jsonl", "w") as f:
            for i, s in H.items(): f.write(json.dumps({"id": i, "steps": [fn(i, s)], "mode": None}) + "\n")
    print("steps written for", list(rules))


def cmd_targets(a):
    H = {r["id"]: r["steps"] for p in sorted(D.glob("hindsight*.jsonl")) for r in L(p)}; eps = {e["id"]: e for e in L(ROOT / TRAJ)}
    for name in _arms(a, [n for n in ["rand_early", "entropy", "attr", "hkl", "selfattr", "selfprog"] if (D / n / "steps.jsonl").exists()]):
        with open(D / name / "targets.jsonl", "w") as f:
            for r in L(D / name / "steps.jsonl"):
                i, t = r["id"], r["steps"][0]; st = H[i][t]
                f.write(json.dumps({"id": i, "t": t, "a_minus": eps[i]["steps"][t]["action"], "admissible": list(st["hind"]),
                                    "teacher_logprobs": st["hind"], "plain_logprobs": st["plain"], "mode": None}) + "\n")
    print("targets written")


def cmd_select(a):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from alfworld_relaxed_verify import parse_goal
    import env_evidence_filter as E
    HS = ["contradicts", "silent", "supports", "n/a"]
    envf = lambda d: [d["plus_goal_revealed"], math.log1p(d["plus_n_new"]), d["plus_no_info"], d["gain_new"]] + [d["hindsight"] == h for h in HS]
    F1 = json.load(open(ROOT / "reports/alfworld/env_evidence/features.json"))
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(np.array([envf(f) for f in F1], float), [f["won"] for f in F1])
    eps = {e["id"]: e for e in L(ROOT / TRAJ)}
    for name in _arms(a, [n for n in ["rand_early", "entropy", "attr", "hkl", "selfattr", "selfprog"] if (D / n / "onestep.log").exists() and "DONE onestep" in open(D / n / "onestep.log").read()]):
        one = {(r["id"], r["t"], r["cand"]): r["obs_plus"] for r in L(D / name / "onestep.jsonl")}
        rows = []
        for T in L(D / name / "targets.jsonl"):
            e = eps[T["id"]]; t = T["t"]; obj = parse_goal(e["goal"])["obj"]; lp = T["teacher_logprobs"]
            cands = [c for c in sorted(lp, key=lambda c: -lp[c]) if c != T["a_minus"]][:2]; best = None
            for c in cands:
                if (T["id"], t, c) not in one: continue
                ep_ = E.evidence(e, t, c, one[(T["id"], t, c)], obj); em = E.evidence(e, t, T["a_minus"], e["steps"][t]["next_obs"], obj)
                d = {"plus_goal_revealed": ep_["goal_revealed"], "plus_n_new": ep_["n_new"], "plus_no_info": ep_["no_info"], "gain_new": ep_["n_new"] - em["n_new"],
                     "hindsight": E.hindsight(e, t + 1, c, obj)}
                s = float(m.predict_proba(np.array([envf(d)], float))[0, 1])
                if best is None or s > best[0]: best = (s, c)
            if best: rows.append({"id": T["id"], "t": t, "cand": best[1], "score": best[0], "cmds": T["admissible"]})
        rows.sort(key=lambda r: -r["score"]); keep = rows[: len(rows) // 2]
        with open(D / name / "train_proxy.jsonl", "w") as f:
            for r in keep:
                f.write(json.dumps({"kind": "kl", "src": TRAJ, "id": r["id"], "t": r["t"], "cmds": r["cmds"], "target": [1.0 if c == r["cand"] else 0.0 for c in r["cmds"]]}) + "\n")
        json.dump(rows, open(D / name / "proxy_scores.json", "w"))
        print(f"{name}: {len(rows)} states scored, kept {len(keep)} (score >= {keep[-1]['score']:.3f})")


def cmd_select_v2(a):
    """v4 selection: environment evidence only VETOES (no new information, or contradicted by the failed log); among the
    surviving candidates pick the one the hindsight teacher (the model itself, full log) prefers; cap any single verb at 60%.
    Emits DPO pairs (chosen = selected candidate, rejected = the logged action a-). Arm 'placebo' = attr states with a random
    admissible action as 'chosen' (measures the effect of the update itself)."""
    import collections
    from alfworld_relaxed_verify import parse_goal
    import env_evidence_filter as E
    eps = {e["id"]: e for e in L(ROOT / TRAJ)}; rng = random.Random(0)
    for name in _arms(a, ["rand_early", "entropy", "attr", "hkl", "selfattr", "selfprog"]):
        one = {(r["id"], r["t"], r["cand"]): r["obs_plus"] for r in L(D / name / "onestep.jsonl")}
        rows = []; nveto = 0
        for T in L(D / name / "targets.jsonl"):
            e = eps[T["id"]]; t = T["t"]; obj = parse_goal(e["goal"])["obj"]; lp = T["teacher_logprobs"]
            cands = [c for c in sorted(lp, key=lambda c: -lp[c]) if c != T["a_minus"]][:2]; ok = []
            for c in cands:
                if (T["id"], t, c) not in one: continue
                ev = E.evidence(e, t, c, one[(T["id"], t, c)], obj)
                if ev["no_info"] or E.hindsight(e, t + 1, c, obj) == "contradicts": nveto += 1; continue
                ok.append(c)
            if ok: rows.append({"id": T["id"], "t": t, "chosen": ok[0], "rejected": T["a_minus"], "p": lp[ok[0]]})
        rows.sort(key=lambda r: -r["p"]); cap = collections.Counter(); keep = []
        n_max = max(1, int(0.6 * len(rows)))
        for r in rows:
            v = r["chosen"].split()[0]
            if cap[v] >= n_max: continue
            cap[v] += 1; keep.append(r)
        with open(D / name / "dpo_pairs.jsonl", "w") as f:
            for r in keep: f.write(json.dumps({"kind": "dpo", "src": TRAJ, "id": r["id"], "t": r["t"], "chosen": r["chosen"], "rejected": r["rejected"]}) + "\n")
        print(f"{name}: {len(rows)} states survive the veto ({nveto} candidates vetoed), kept {len(keep)} pairs; verbs {dict(cap)}")
    # placebo: same states as attr, random admissible 'chosen'
    n_attr = sum(1 for _ in open(D / "attr" / "dpo_pairs.jsonl")); (D / "placebo").mkdir(exist_ok=True)
    T_attr = L(D / "attr" / "targets.jsonl"); rng.shuffle(T_attr)
    with open(D / "placebo" / "dpo_pairs.jsonl", "w") as f:
        for T in T_attr[:n_attr]:
            c = rng.choice([x for x in T["admissible"] if x != T["a_minus"]])
            f.write(json.dumps({"kind": "dpo", "src": TRAJ, "id": T["id"], "t": T["t"], "chosen": c, "rejected": T["a_minus"]}) + "\n")
    print(f"placebo: {n_attr} pairs")


def cmd_refs(a):
    """Reference log-probs for DPO with the SAME scorer used in training (student_logprobs_grad), base model, no grad."""
    import torch
    from alfworld_attr_opd import history_at, student_logprobs_grad
    from alfworld_opd_plus import load_policy
    eps = {e["id"]: e for e in L(ROOT / TRAJ)}; tok, model, pol = load_policy(MODEL, STUDENT)
    for name in _arms(a, ["placebo", "rand_early", "entropy", "attr", "hkl", "selfattr", "selfprog"]):
        rows = L(D / name / "dpo_pairs.jsonl")
        with torch.no_grad():
            for r in rows:
                lp = student_logprobs_grad(pol, tok, history_at(eps[r["id"]], r["t"]), [r["chosen"], r["rejected"]])
                r["ref_c"], r["ref_r"] = float(lp[0]), float(lp[1])
        with open(D / name / "dpo_pairs.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        print(name, "refs", len(rows), flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hindsight"); h.add_argument("--ids", required=True); h.add_argument("--out", default="hindsight.jsonl")
    sa = sub.add_parser("selfattr"); sa.add_argument("--ids", required=True)
    for n in ("hindsight_steps", "order"): sub.add_parser(n).add_argument("--steps", required=True)
    for n in ("choose", "targets", "select", "select_v2", "refs"): sub.add_parser(n).add_argument("--arms", default=None)
    a = ap.parse_args(); D.mkdir(parents=True, exist_ok=True)
    {"hindsight": cmd_hindsight, "selfattr": cmd_selfattr, "choose": cmd_choose, "targets": cmd_targets, "select": cmd_select, "select_v2": cmd_select_v2, "refs": cmd_refs, "hindsight_steps": cmd_hindsight_steps, "order": cmd_order}[a.cmd](a)


if __name__ == "__main__":
    main()
