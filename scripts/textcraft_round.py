"""TextCraft self-evolution round with the ALFWorld FINAL recipe (user-approved Oct 8: R1 only).
Stages (each sharded stage shards its work list BEFORE the resume filter and writes one file per shard; `cover` checks coverage):
  fails    -- merged R0 collection (train_r1 order) -> ids_fail.jsonl (failures among the FIRST 1000 tasks), K = round(0.2 x #fails),
              Model A release rows for exactly those failures, split into --n_shards input files (integer step_number, reasoning None,
              is_final_answer False, no reward/score)
  pointer  -- Model A predictions -> pointer.jsonl {id, t_hat clipped to the episode} and diag_modelA.raw.jsonl {id, raw t_hat, type}
              (same code path as run_a3b_final_chain.sh modelA_pointer)
  teacher  -- informed self-teacher = scripts/self_teacher_informed.py: hindsight block (self_evolve.HindsightHistory) + Model A diagnosis
              + the student's own analysis (teacher_reason_auc.NoteHistory / ANALYSIS_Q, greedy, 96 tokens), then the student scores the
              admissible set; writes {id, t, a_minus, admissible, teacher_logprobs, plain_logprobs, analysis, diag}
  onestep  -- (CPU) candidates = top-2 of the teacher distribution excluding a- (proxy_eval_se3b.candidates); each executed ONE step from
              the saved state; evidence F from scripts/textcraft_evidence.py (env_score = F, no_info as defined there)
  select   -- rule env_top exactly as scripts/proxy_arms_se3b.py: per state veto no_info, keep candidates with env_score > 0, pick max
              (env_score, hind_p); rank states by (env_score, hind_p); keep top K; one-hot KL target over the admissible list
  train    -- scripts/alfworld_opd_plus.py cmd_train UNCHANGED (one-hot KL over the admissible set for corrections, SFT CE for replay,
              per-example non-finite guard, AdamW lr 1e-5, batch 8, 1 epoch, seed 0, continuing the start LoRA); only the history object
              and the policy prompt are TextCraft's (textcraft_agent.history_at / SYSTEM)
  cover    -- unique ids of a set of shard files == expected
  compare  -- R0 vs R1 on TEST: success overall / by depth, steps, failure types, paired McNemar, goal-level sign test
"""
import argparse, collections, glob, json, math, os, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("ERROR_ATTR_ROOT", str(ROOT)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from textcraft_env import TextCraftEnv, splits


def L(p): return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def rel(p):
    p = Path(p); return str(p.relative_to(ROOT)) if p.is_absolute() else str(p)


def softmax_d(lp):
    z = max(lp.values()); Z = z + math.log(sum(math.exp(v - z) for v in lp.values())); return {c: math.exp(v - Z) for c, v in lp.items()}


# ----------------------------------------------------------------------------- merge / coverage
def merge(files, key):
    rows = {}
    for f in files:
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            rows.setdefault(key(r), r)
    return rows


def cmd_cover(a):
    files = [f for f in sorted(glob.glob(str(ROOT / a.glob))) if not f.endswith(".release.jsonl")]; keyf = (lambda r: r["task"]) if a.key == "task" else (lambda r: r["id"]) if a.key == "id" else (lambda r: (r["id"], r["t"]))
    got = merge(files, keyf)
    if a.split:
        exp = set(splits()[a.split][a.offset: a.offset + a.n] if a.n else splits()[a.split][a.offset:])
    else:
        exp = {r["id"] for r in L(ROOT / a.ids)}
        if a.key == "idt": got = {k[0]: v for k, v in got.items()}
    miss = exp - set(got); extra = set(got) - exp
    print(f"COVER {a.glob}: {len(files)} files, {len(got)} unique, expected {len(exp)}, missing {len(miss)}, extra {len(extra)} -> {'OK' if not miss and not extra else 'INCOMPLETE'}")
    if a.out and not miss:
        order = splits()[a.split] if a.split else [r["id"] for r in L(ROOT / a.ids)]
        pos = {k: i for i, k in enumerate(order)}
        rows = sorted((v for k, v in got.items() if k in exp), key=lambda r: pos[r["task"] if a.key == "task" else r["id"]])
        open(ROOT / a.out, "w").writelines(json.dumps(r) + "\n" for r in rows)
        print(f"merged -> {a.out} ({len(rows)} rows)")
    sys.exit(0 if not miss and not extra else 1)


# ----------------------------------------------------------------------------- failures / release / pointer
def cmd_fails(a):
    from textcraft_agent import to_release
    eps = L(ROOT / a.traj); first = eps[: a.first]; fails = [e for e in first if not e["won"]]
    D = ROOT / a.out_dir; D.mkdir(parents=True, exist_ok=True)
    open(D / "ids_fail.jsonl", "w").writelines(json.dumps({"id": e["id"]}) + "\n" for e in fails)
    K = max(1, round(a.k_frac * len(fails))); json.dump({"n_first": len(first), "n_fail_first": len(fails), "k_frac": a.k_frac, "K": K,
                                                        "n_total": len(eps), "n_won_total": sum(e["won"] for e in eps)}, open(D / "K.json", "w"), indent=1)
    rows = [to_release(e) for e in fails]
    if a.query_full:   # v3 (Oct 9, user): Model A sees the complete task statement (all crafting commands + goal); the user turn is not rendered
        for r, e in zip(rows, fails): r["task"]["query"] = e["initial_obs"]
    for i in range(a.n_shards):
        open(D / f"release_s{i}.jsonl", "w").writelines(json.dumps(r) + "\n" for r in rows[i::a.n_shards])
    print(f"fails: {len(eps)} episodes ({sum(e['won'] for e in eps)} won); first {len(first)}: {len(fails)} failures -> K = round({a.k_frac} x {len(fails)}) = {K}; "
          f"release in {a.n_shards} shard files")


def cmd_pointer(a):
    D = ROOT / a.out_dir; ids = [r["id"] for r in L(D / "ids_fail.jsonl")]; eps = {e["id"]: e for e in L(ROOT / a.traj)}
    pred = merge(sorted(glob.glob(str(D / "modelA_pred_s*.jsonl"))), lambda r: r["id"])
    miss = [i for i in ids if i not in pred]
    if miss: raise SystemExit(f"Model A predictions missing for {len(miss)} failures")
    rows = []
    for i in ids:
        th = pred[i]["t_hat"]; t = max(0, min(int(th) if th is not None else 0, len(eps[i]["steps"]) - 1))
        rows.append({"id": i, "t_hat": t, "type": None, "raw_t": th, "n": len(eps[i]["steps"]), "mode": (pred[i].get("pred") or {}).get("error_mode")})
    open(D / "pointer.jsonl", "w").writelines(json.dumps({"id": r["id"], "steps": [r["t_hat"]], "mode": None}) + "\n" for r in rows)
    open(D / "pointer.raw.jsonl", "w").writelines(json.dumps({"id": r["id"], "t_hat": r["t_hat"], "type": None}) + "\n" for r in rows)
    open(D / "diag_modelA.raw.jsonl", "w").writelines(json.dumps({"id": r["id"], "t_hat": r["raw_t"] if r["raw_t"] is not None else 0, "type": "Model-A mode " + str(r["mode"])}) + "\n" for r in rows)
    ts = [r["t_hat"] for r in rows]; rl = [r["t_hat"] / max(1, r["n"] - 1) for r in rows]
    print(f"Model A pointer: {len(rows)} states, unparsed {sum(r['raw_t'] is None for r in rows)}, mean t {sum(ts) / len(ts):.2f} (mean length {sum(r['n'] for r in rows) / len(rows):.1f}), "
          f"t0 {sum(t == 0 for t in ts) / len(ts):.2f}, relative position mean {sum(rl) / len(rl):.2f}, quartiles {sorted(collections.Counter(min(3, int(4 * x)) for x in rl).items())}, "
          f"modes {dict(collections.Counter(r['mode'] for r in rows).most_common(5))}")


# ----------------------------------------------------------------------------- policy (TextCraft prompt) usable by the shared trainer
def tc_policy_class():
    from error_attr.policy.alfworld_agent import LLMPolicy
    from textcraft_agent import SYSTEM

    class TCPolicy(LLMPolicy):
        def _build_chat_prompt(self, history):
            return self.tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": history.render_prompt()}],
                                                      tokenize=False, add_generation_prompt=True)
    return TCPolicy


def load_tc_policy(student, adapter, trainable=False):
    import alfworld_opd_plus as O
    tok, model, _ = O._orig_load_policy(student, adapter, trainable) if hasattr(O, "_orig_load_policy") else O.load_policy(student, adapter, trainable)
    return tok, model, tc_policy_class()(model, tok)


# ----------------------------------------------------------------------------- informed self-teacher (= self_teacher_informed.py)
class FullHindsight:
    """v3 teacher view (Oct 9, user): nothing truncated -- the FULL prefix (no 10-step window), EVERY remaining step of the failed episode
    with its FULL observation (self_evolve.HindsightHistory caps the future at 25 steps and 160 characters), the outcome (FAILED,
    episode length, step budget) and the full task statement (the policy prompt header: all crafting commands + the goal)."""
    def __init__(self, e, t):
        from textcraft_agent import TCHistory
        self._h = TCHistory(e["initial_obs"], [(s_["action"], s_["next_obs"]) for s_ in e["steps"][:t]], e["steps"][t]["admissible"], history=10 ** 9)
        self.steps = self._h.steps
        self._block = "\n".join(f"- step {t + i}: {s_['action']} -> {s_['next_obs']}" for i, s_ in enumerate(e["steps"][t:]))
        self._n = len(e["steps"]); self._b = e["budget"]
    def render_prompt(self, max_history_steps=None):
        p = self._h.render_prompt()
        note = ("Hindsight: this attempt FAILED (the goal item was never crafted). After this point the agent did the following "
                f"(episode length {self._n} steps, step budget {self._b}, budget exhausted):\n{self._block}\n"
                "Using everything above, choose the best action for the CURRENT step.\n")
        return p.replace("Admissible actions:", note + "Admissible actions:", 1)



TYPE_WORDS = {"wrong_location": "went to the wrong place", "wrong_object": "took the wrong object", "wrong_target": "acted on the wrong place or object",
              "misorder": "did a later step too early (e.g. placed before cleaning/heating)", "misplace": "put the object in the wrong place",
              "wrong_value": "put the object in the wrong place", "wrong_op": "used the wrong action", "skip": "skipped a required step", "random": "an unproductive action"}
GEN_SYSTEM = "You are a crafting agent reviewing a failed attempt."   # ALFWorld: "You are an embodied household agent reviewing a failed attempt."


def cmd_teacher(a):
    import torch
    from textcraft_agent import history_at
    from self_evolve import HindsightHistory
    from teacher_reason_auc import NoteHistory, ANALYSIS_Q
    D = ROOT / a.out_dir; eps = {e["id"]: e for e in L(ROOT / a.traj)}
    steps = [(r["id"], r["steps"][0]) for r in L(D / "pointer.jsonl")]; diag = {r["id"]: r for r in L(D / "diag_modelA.raw.jsonl")}
    steps = steps[a.shard[0]::a.shard[1]]                                    # shard BEFORE the resume filter
    out = D / f"targets_s{a.shard[0]}.jsonl"; done = set()
    if out.exists():
        keep = []; own = set(steps)
        for l in open(out):
            try: r = json.loads(l)
            except Exception: continue
            k = (r["id"], r["t"])
            if k in own and k not in done: done.add(k); keep.append(json.dumps(r) + "\n")
        open(out, "w").writelines(keep)
    todo = [s for s in steps if s not in done]; print(f"shard {a.shard}: {len(steps)} states, {len(done)} done, {len(todo)} to do", flush=True)
    tok, model, pol = load_tc_policy(a.model, str(ROOT / a.adapter) if not os.path.isabs(a.adapter) else a.adapter); model.eval()

    def generate(history, question, max_new_tokens=96):
        user = history.render_prompt().replace("Next action:", question)
        msg = tok.apply_chat_template([{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        x = tok(msg, return_tensors="pt", add_special_tokens=False).to(model.device)
        o = model.generate(**x, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        return tok.decode(o[0, x.input_ids.shape[1]:], skip_special_tokens=True).strip()
    with open(out, "a") as f, torch.no_grad():
        for k, (i, t) in enumerate(todo):
            e = eps[i]; h = history_at(e, t); lp_plain = pol.score_admissible_commands_cached(h)   # plain = the policy's own view when it acted
            if a.full_view: hh = FullHindsight(e, t); h = hh                                    # v3: the teacher sees the whole failed trajectory
            else: hh = HindsightHistory(h, e, t)
            parts = []; d = diag.get(i)
            if d:
                typ = d.get("type"); parts.append(f"Diagnosis (an error attributor trained on this agent's own trajectories): the first decisive error was at step {d['t_hat']}"
                                                  + (f", type: {typ} ({TYPE_WORDS.get(typ, typ)})" if typ else "") + ".")
            an = generate(NoteHistory(h, "\n".join(parts) if parts else "Note: this attempt FAILED.", hind=hh), ANALYSIS_Q)
            parts.append("Analysis: " + an)
            lp = pol.score_admissible_commands_cached(NoteHistory(h, "\n".join(parts), hind=hh))
            f.write(json.dumps({"id": i, "t": t, "a_minus": e["steps"][t]["action"], "admissible": list(lp), "teacher_logprobs": lp, "plain_logprobs": lp_plain,
                                "mode": "informed", "analysis": an, "diag": d}) + "\n"); f.flush()
            if k % 20 == 0: print(f"[{k + 1}/{len(todo)}]", flush=True)
    print("DONE informed_teacher", flush=True)


# ----------------------------------------------------------------------------- one-step evidence of the 2 candidates
def cmd_onestep(a):
    from textcraft_evidence import req_tree, replay, one_step
    D = ROOT / a.out_dir; eps = {e["id"]: e for e in L(ROOT / a.traj)}
    tg = sorted(merge(sorted(glob.glob(str(D / "targets_s*.jsonl"))), lambda r: (r["id"], r["t"])).values(), key=lambda r: (r["id"], r["t"]))
    tg = tg[a.shard[0]::a.shard[1]]                                           # shard BEFORE the resume filter
    out = D / f"onestep_s{a.shard[0]}.jsonl"; done = {(r["id"], r["t"]) for r in L(out)}
    env = TextCraftEnv(); n = 0
    with open(out, "a") as f:
        for T in tg:
            if (T["id"], T["t"]) in done: continue
            e = eps[T["id"]]; t = T["t"]; lp = T["teacher_logprobs"]; pl = T["plain_logprobs"]; ph, pp = softmax_d(lp), softmax_d(pl)
            alts = [c for c in sorted(lp, key=lambda c: -lp[c]) if c != T["a_minus"]][:2]
            seen = replay(env, e, t); inv_t = env.get_state(); nodes = req_tree(env)
            orac = env.expert_actions(); orac = orac[0] if orac else None
            cands = []
            for rank, c in enumerate(alts):
                ev = one_step(env, e, t, nodes, inv_t, seen, c)
                cands.append({"cand": c, "rank": rank, "hind_p": ph[c], "plain_p": pp[c], "env_score": ev["F"], "no_info": ev["no_info"], "dphi": ev["dphi"],
                              "regress": ev["regress"], "dL": ev["dL"], "done": ev["done"], "obs_plus": ev["obs"]})
            taken = one_step(env, e, t, nodes, inv_t, seen, T["a_minus"])
            f.write(json.dumps({"id": T["id"], "t": t, "a_minus": T["a_minus"], "inv_t": inv_t, "oracle_next": orac, "taken": {k: taken[k] for k in ("F", "no_info", "dL", "done")},
                                "cands": cands}) + "\n"); n += 1
    print(f"DONE onestep {n} states (shard {a.shard})", flush=True)


# ----------------------------------------------------------------------------- env_top selection
def ckind(c):
    k = c.split()[0]; return k if k in ("get", "craft") else "inventory"


def taken_scores(traj, keys):
    """{(id, t): F of the logged action at t}: inventory before = replay of steps[:t], after = replay of steps[:t+1], observation = the
    LOGGED next_obs (asserted equal to the deterministic re-execution)."""
    from textcraft_evidence import req_tree, replay, evidence
    eps = {e["id"]: e for e in L(ROOT / traj)}; env = TextCraftEnv(); out = {}
    for (i, t) in keys:
        e = eps[i]; seen = replay(env, e, t); inv0 = env.get_state(); nodes = req_tree(env); s_ = e["steps"][t]
        o, r, d = env.step(s_["action"]); assert o == s_["next_obs"], (i, t, o, s_["next_obs"])
        out[(i, t)] = evidence(env, nodes, inv0, s_["action"], s_["next_obs"], env.get_state(), d, seen)["F"]
    return out


def cmd_select(a):
    D = ROOT / a.out_dir; K = json.load(open(D / "K.json"))["K"] if a.k is None else a.k
    tg = merge(sorted(glob.glob(str(D / "targets_s*.jsonl"))), lambda r: (r["id"], r["t"]))
    one = merge(sorted(glob.glob(str(D / "onestep_s*.jsonl"))), lambda r: (r["id"], r["t"]))
    picks = []; ft = {}
    if a.rule == "env_top_margin":   # Oct 9 (user): F of the action actually taken, from its OBSERVED next observation (same evidence function)
        ft = taken_scores(a.traj, list(one))
    for key, r in one.items():
        ok = [c for c in r["cands"] if not c["no_info"] and c["env_score"] > 0]
        if a.rule == "env_top_margin": ok = [c for c in ok if c["env_score"] > ft[key]]   # STRICT improvement over the taken action
        if not ok: continue
        c = max(ok, key=lambda c: (c["env_score"], c["hind_p"])); picks.append((key, c, (c["env_score"], c["hind_p"]), r))
    picks = sorted(picks, key=lambda x: x[2], reverse=True)[:K]
    out = (ROOT / a.dest if a.dest else D) / f"targets_{a.rule}"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.jsonl", "w") as f:
        for key, c, _, r in picks:
            cmds = tg[key]["admissible"]
            f.write(json.dumps({"kind": "kl", "src": rel(a.traj), "id": key[0], "t": key[1], "cmds": cmds, "target": [1.0 if x == c["cand"] else 0.0 for x in cmds], "cand": c["cand"]}) + "\n")
    good = lambda c: c["dL"] <= -1 or c["done"]
    kinds = collections.Counter((ckind(c["cand"]), "a-:" + ckind(r["a_minus"])) for _, c, _, r in picks)
    st = {"K": K, "n_states": len(one), "n_with_positive": sum(1 for r in one.values() if any(not c["no_info"] and c["env_score"] > 0 for c in r["cands"])),
          "n_targets": len(picks), "cand_good_oracle": sum(good(c) for _, c, _, _ in picks), "taken_bad_and_cand_good": sum(1 for _, c, _, r in picks if good(c) and not (r["taken"]["dL"] <= -1 or r["taken"]["done"])),
          "cand_is_oracle_next": sum(1 for _, c, _, r in picks if c["cand"] == r["oracle_next"]), "mean_F": sum(c["env_score"] for _, c, _, _ in picks) / max(1, len(picks)),
          "mean_hind_p": sum(c["hind_p"] for _, c, _, _ in picks) / max(1, len(picks)), "kinds(cand, a-)": {f"{k[0]}|{k[1]}": v for k, v in kinds.most_common()},
          "a_minus_failed": sum(1 for _, c, _, r in picks if r["taken"]["no_info"])}
    if ft:
        st["n_states_margin_ok"] = sum(1 for key, r in one.items() if any(not c["no_info"] and c["env_score"] > 0 and c["env_score"] > ft[key] for c in r["cands"]))
        st["taken_F_mean_kept"] = sum(ft[key] for key, _, _, _ in picks) / max(1, len(picks))
        st["stored_taken_F_agrees"] = sum(abs(ft[key] - r["taken"]["F"]) < 1e-9 for key, r in one.items())
        print(f"margin rule: {st['n_states_margin_ok']} of {len(one)} states have a candidate with F > F(taken); stored re-executed taken F agrees on {st['stored_taken_F_agrees']}/{len(one)}")
    json.dump(st, open(out / "stats.json", "w"), indent=1)
    print(f"=== textcraft {a.rule}: {len(one)} scanned states, {st['n_with_positive']} with a positive non-null candidate, K = {K}, kept {len(picks)} | "
          f"kept candidates that shorten the oracle plan {st['cand_good_oracle']}/{len(picks)}; true corrections (taken not progress -> cand progress) {st['taken_bad_and_cand_good']}; "
          f"cand == oracle next {st['cand_is_oracle_next']}; taken action had failed {st['a_minus_failed']} | kinds {st['kinds(cand, a-)']}")


def cmd_replay_fw(a):
    """make_replay.py --mode fail_weighted, TextCraft category = crafting depth: --n steps split over depths in proportion to the round's
    OWN training-collection failures per depth (round per depth), each quota filled with the non-inventory steps of that depth's
    shortest wins, trajectory by trajectory (as in make_replay.py, the last trajectory is cut when the quota is reached)."""
    ALL = L(ROOT / a.traj); src = rel(a.traj); wins = sorted([e for e in ALL if e["won"]], key=lambda e: len(e["steps"])); rows = []
    TY = lambda e: e["depth"]; fails = {}
    for e in ALL:
        if not e["won"]: fails[TY(e)] = fails.get(TY(e), 0) + 1
    tot = sum(fails.values()); quota = {k: round(a.n * v / tot) for k, v in fails.items()}; eff = dict(quota); carry = 0
    def fill(k, q):
        got = 0
        for e in [w for w in wins if TY(w) == k]:
            if got >= q: break
            for t, s_ in enumerate(e["steps"]):
                if s_["action"].split()[0] in ("look", "inventory", "help", "examine") or got >= q: continue
                rows.append({"kind": "sft", "src": src, "id": e["id"], "t": t, "action": s_["action"]}); got += 1
        return got
    if a.carry:   # v3 (Oct 9, user): deepest depth first; a depth's unfilled quota moves to the next shallower depth that has wins
        for k in sorted(quota, reverse=True):
            eff[k] = quota[k] + carry; carry = eff[k] - fill(k, eff[k])
        if carry: print(f"WARNING: {carry} replay steps could not be placed (no shallower depth with enough wins)")
        print(f"carry-over: effective quota by depth {dict(sorted(eff.items()))}")
    else:
        for k in sorted(quota, key=lambda k: -quota[k]): fill(k, quota[k])
    Path(ROOT / a.out).parent.mkdir(parents=True, exist_ok=True); open(ROOT / a.out, "w").writelines(json.dumps(r) + "\n" for r in rows)
    ids = {r["id"] for r in rows}; byd = collections.Counter(next(e["depth"] for e in wins if e["id"] == r["id"]) for r in rows)
    full = sum(1 for i in ids if sum(1 for r in rows if r["id"] == i) == sum(1 for s_ in next(e for e in wins if e["id"] == i)["steps"] if s_["action"].split()[0] != "inventory"))
    print(f"fail_weighted quota by depth (failures {dict(sorted(fails.items()))}): {dict(sorted(quota.items()))}; rows by depth {dict(sorted(byd.items()))}; "
          f"{len(rows)} steps from {len(ids)} wins ({full} complete); verbs {dict(collections.Counter(r['action'].split()[0] for r in rows))}")


# ----------------------------------------------------------------------------- trainer (shared code, TextCraft prompt)
def cmd_train(a):
    import alfworld_opd_plus as O
    from textcraft_agent import history_at
    O._orig_load_policy = O.load_policy
    O.load_policy = lambda student, adapter, trainable=False: load_tc_policy(student, adapter, trainable)
    O.history_at = lambda ep, t: history_at(ep, t)
    # Oct 9 01:45 (R1_v3, after 6 OOMs on shared GPUs): score the admissible commands in chunks of LP_CHUNK=2 instead of 8 inside
    # student_logprobs_grad.  Memory only: every command's log-prob is an independent right-padded forward, so loss and gradients are
    # the same (up to bf16 kernel noise); the 8-command chunk materialised 8 x prompt x vocab logits (~9 GB for TextCraft prompts).
    import functools
    O.student_logprobs_grad = functools.partial(O.student_logprobs_grad, batch_size=int(os.environ.get("TC_LP_CHUNK", "2")))
    O.cmd_train(a)


# ----------------------------------------------------------------------------- R0 vs R1
def binom_two_sided(k, n):
    if n == 0: return 1.0
    p = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * p)


def cmd_compare(a):
    from textcraft_env import parse_task
    A = {e["task"]: e for e in L(ROOT / a.r0)}; B = {e["task"]: e for e in L(ROOT / a.r1)}; ks = sorted(set(A) & set(B))
    print(f"paired tasks {len(ks)} (r0 {len(A)}, r1 {len(B)})")
    n0, n1 = a.names.split(",")
    for name, X in ((n0, A), (n1, B)):
        byd = collections.defaultdict(list)
        for k in ks: byd[X[k]["depth"]].append(X[k]["won"])
        w = sum(X[k]["won"] for k in ks)
        print(f"  {name}: {w}/{len(ks)} = {w / len(ks):.3f}; mean steps {sum(X[k]['n_steps'] for k in ks) / len(ks):.2f}; by depth " + ", ".join(f"d{d} {sum(v)}/{len(v)}" for d, v in sorted(byd.items())))
    b = sum(1 for k in ks if B[k]["won"] and not A[k]["won"]); c = sum(1 for k in ks if A[k]["won"] and not B[k]["won"])
    print(f"  McNemar (task level): {n1} only {b}, {n0} only {c}, exact two-sided p = {binom_two_sided(min(b, c), b + c):.4g}")
    g = collections.defaultdict(lambda: [0, 0])
    for k in ks:
        gi = parse_task(k)[0]; g[gi][0] += A[k]["won"]; g[gi][1] += B[k]["won"]
    up = sum(1 for v in g.values() if v[1] > v[0]); dn = sum(1 for v in g.values() if v[1] < v[0]); eq = len(g) - up - dn
    print(f"  goal-level sign test over {len(g)} goals (wins over the 5 seeds): {n1} better {up}, {n0} better {dn}, ties {eq}, exact two-sided p = {binom_two_sided(min(up, dn), up + dn):.4g}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("cover"); s.add_argument("--glob", required=True); s.add_argument("--key", default="task", choices=["task", "id", "idt"]); s.add_argument("--split", default=None)
    s.add_argument("--offset", type=int, default=0); s.add_argument("--n", type=int, default=0); s.add_argument("--ids", default=None); s.add_argument("--out", default=None)
    s = sub.add_parser("fails"); s.add_argument("--traj", required=True); s.add_argument("--out_dir", required=True); s.add_argument("--first", type=int, default=1000)
    s.add_argument("--k_frac", type=float, default=0.2); s.add_argument("--n_shards", type=int, default=2); s.add_argument("--query_full", action="store_true")
    s = sub.add_parser("pointer"); s.add_argument("--traj", required=True); s.add_argument("--out_dir", required=True)
    s = sub.add_parser("teacher"); s.add_argument("--traj", required=True); s.add_argument("--out_dir", required=True); s.add_argument("--adapter", required=True)
    s.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); s.add_argument("--shard", type=int, nargs=2, default=[0, 1]); s.add_argument("--full_view", action="store_true")
    s = sub.add_parser("onestep"); s.add_argument("--traj", required=True); s.add_argument("--out_dir", required=True); s.add_argument("--shard", type=int, nargs=2, default=[0, 1])
    s = sub.add_parser("select"); s.add_argument("--traj", required=True); s.add_argument("--out_dir", required=True); s.add_argument("--k", type=int, default=None)
    s.add_argument("--rule", default="env_top", choices=["env_top", "env_top_margin"]); s.add_argument("--dest", default=None, help="output dir (default --out_dir)")
    s = sub.add_parser("replay_fw"); s.add_argument("--traj", required=True); s.add_argument("--out", required=True); s.add_argument("--n", type=int, default=100); s.add_argument("--carry", action="store_true")
    t = sub.add_parser("train"); t.add_argument("--data", nargs="+", required=True); t.add_argument("--student", default="Qwen/Qwen2.5-3B-Instruct"); t.add_argument("--adapter", required=True)
    t.add_argument("--out", required=True); t.add_argument("--lr", type=float, default=1e-5); t.add_argument("--batch", type=int, default=8)
    t.add_argument("--epochs", type=int, default=1); t.add_argument("--beta", type=float, default=0.1); t.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("compare"); s.add_argument("--r0", required=True); s.add_argument("--r1", required=True); s.add_argument("--names", default="R0,R1")
    a = ap.parse_args()
    {"cover": cmd_cover, "fails": cmd_fails, "pointer": cmd_pointer, "teacher": cmd_teacher, "onestep": cmd_onestep, "select": cmd_select, "train": cmd_train, "compare": cmd_compare, "replay_fw": cmd_replay_fw}[a.cmd](a)


if __name__ == "__main__":
    main()
