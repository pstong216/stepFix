"""StepFix on ScienceWorld (Oct 9 2026, user-approved) -- the ALFWorld recipe (scripts/run_k20_round.sh) with a GENERATIVE adaptation of
the hindsight self-teacher (ScienceWorld has ~400+ valid actions per step and the student generates its action; nothing is scored over an
admissible list).  Stages (all resumable, all write under the round dir D):
  prep     merge collection shards (dedupe by id) -> D/traj.jsonl, D/ids_fail.jsonl, D/traj.release.jsonl (failures, alfagent rows for Model A)
  pointer  Model A predictions (scripts/alfworld_attribute.py --variant no_observations) -> D/pointer.jsonl (t* = clip(t_hat))
  teacher  [GPU] per failure: replay the recorded prefix (logs mismatches), hindsight self-teacher prompt (same construction as
           scripts/self_teacher_informed.py, rendered with the ScienceWorld student's render/chat), own <=3-sentence analysis, 8 samples at
           T=1.0 mapped with the SAFE snap rule, valid & != a- kept, ranked by the teacher's log-prob of the action text (+EOS), top 2;
           each candidate executed ONCE after replaying the prefix from reset; one-step generic evidence (scripts/generic_evidence.py, native
           score) -> D/teacher/rows_s*.jsonl
  select   env_top (exactly as ALFWorld: best candidate with S>0 and not no-info, tie by teacher prob; top K = round(0.2 x failures) states
           by (S, prob)) -> D/targets_env_top.jsonl ; random control -> D/targets_rand.jsonl ; statistics -> D/select_stats.json
  replay   100 replay steps (2 random steps of each of the 50 shortest wins; skip look around / inventory / task) -> D/replay.jsonl
  train    [GPU] LoRA continued from the current adapter, 1 epoch, lr 1e-5, batch 8 (per-example backward, clip 1.0, AdamW -- the ALFWorld
           trainer's optimisation), token-level CE on the action text given the student's own prompt (10-step history); checkpoint every
           --ckpt_every optimizer steps; resumes from the checkpoint after preemption.
The student input NEVER contains hindsight, diagnosis or analysis (training rows are rendered with sciworld_agent.render only).
"""
import argparse, difflib, glob, json, math, os, random, sys, time
import os as _os_umask; _os_umask.umask(0o027)   # Oct 12: some srun steps inherit umask 0117 (dirs created without x); force 0027
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("ERROR_ATTR_ROOT", str(ROOT)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from sciworld_agent import render, chat, SYSTEM
from generic_evidence import generic_evidence

ANALYSIS_Q = ("Before acting: this attempt FAILED. In at most 3 sentences, say why it failed and what the agent should do at the "
              "CURRENT step (do not name a specific action yet).")   # identical to teacher_reason_auc.ANALYSIS_Q (ALFWorld)
REVIEW_SYSTEM = "You are an agent reviewing a failed attempt at a science experiment task in a simulated text environment."   # ALFWorld: "...embodied household agent reviewing a failed attempt."
NOINFO_PREFIX = ("no known action matches that input", "ambiguous request", "unknown action")   # Oct 9 23:40: + "Unknown action. Type 'help'..." (0 R1 candidates affected)


def L(p):
    out = []
    for l in open(p):
        try: out.append(json.loads(l))
        except Exception: pass
    return out


def snap_safe(raw, va):
    """The SAFE snap rule of sciworld_agent.py collect --snap safe; returns (action or None if not mappable to a valid action, mapped?)."""
    if raw in va: return raw, False
    vb = raw.split(" ")[0] if raw else ""
    pool = [x for x in va if x.split(" ")[0] == vb and not (x.startswith("focus on") and not raw.startswith("focus on"))]
    m = difflib.get_close_matches(raw, pool, n=1, cutoff=0.85)
    return (m[0], True) if m else (None, False)


# ----------------------------------------------------------------------------- one-step evidence (NEW; ALFWorld behaviour untouched)
def sw_body(o, goal):
    """ScienceWorld 'observe' stub: strip the task description (the 'task' command repeats it), like WebShop strips its instruction."""
    g = goal.strip(); g2 = g.replace("Task Description:", "").strip()
    return o.replace(g, " ").replace(g2, " ").strip()


def sciworld_evidence(goal, hist_obs, cur_obs, obs_after, score_before, score_after, done):
    b = lambda o: sw_body(o, goal)
    ev = generic_evidence(goal, [b(o) for o in hist_obs], b(cur_obs), b(obs_after), env="scienceworld", score_before=score_before,
                          score_after=score_after, done=done, success=score_after >= 100)
    if obs_after.strip().lower().startswith(NOINFO_PREFIX) and not ev["no_info"]:   # parser rejection / ambiguity prompt = no information
        ev["no_info"] = True; ev["score"] -= 2.0
    ev["parser_reject"] = obs_after.strip().lower().startswith(NOINFO_PREFIX)
    return ev


def hist_obs_at(e, t):   # every observation seen before steps[t].obs (initial obs = steps[0].obs), as alfworld_step builds it
    return [s["obs"] for s in e["steps"][:t]] + [s["next_obs"] for s in e["steps"][:t]]


# ----------------------------------------------------------------------------- prep / pointer
def to_release(e):
    traj = [{"kind": "user", "content": e["steps"][0]["obs"] if e["steps"] else ""}]
    for i, st in enumerate(e["steps"]):
        traj.append({"kind": "action", "step_number": i + 1, "reasoning": None, "action": st["action"], "observation": st["next_obs"], "is_final_answer": False})
    return {"id": e["id"], "framework": "alfagent", "benchmark": "scienceworld", "modality": "text", "task": {"query": e["goal"], "answer": ""},
            "trajectory": traj, "ground_truth": {}, "extras": {"agents": [{"id": "agent", "role": "agent"}]}}


def cmd_prep(a):
    D = ROOT / a.dir; D.mkdir(parents=True, exist_ok=True); rows = {}
    for pat in a.inputs:
        for p in sorted(glob.glob(str(ROOT / pat))):
            for e in L(p): rows.setdefault(e["id"], e)
    eps = sorted(rows.values(), key=lambda e: (e["task"], e["variation"])); fails = [e for e in eps if not e["won"]]
    open(D / "traj.jsonl", "w").writelines(json.dumps(e) + "\n" for e in eps)
    open(D / "ids_fail.jsonl", "w").writelines(json.dumps({"id": e["id"]}) + "\n" for e in fails)
    open(D / "traj.release.jsonl", "w").writelines(json.dumps(to_release(e)) + "\n" for e in fails)
    neg = sum(e["final_score"] < 0 for e in fails); early = sum(e["n_steps"] < e["budget"] for e in fails)
    print(f"prep: {len(eps)} episodes ({a.expect} expected), {len(eps) - len(fails)} won, {len(fails)} failed ({neg} ended with score<0, {early} ended before the budget); "
          f"mean score {sum(max(e['final_score'], 0) for e in eps) / max(1, len(eps)):.1f}")
    if a.expect and len(eps) != a.expect: print(f"WARNING: {len(eps)} episodes != expected {a.expect}")


def cmd_pointer(a):
    D = ROOT / a.dir; eps = {e["id"]: e for e in L(D / "traj.jsonl")}; ids = [r["id"] for r in L(D / "ids_fail.jsonl")]
    pred = {}
    for p in sorted(glob.glob(str(D / "modelA_pred*.jsonl"))):
        for r in L(p): pred.setdefault(r["id"], r)
    rows, miss, nonum, clip = [], 0, 0, 0
    for i in ids:
        if i not in pred: miss += 1; continue
        th = pred[i]["t_hat"]
        if th is None: nonum += 1; continue
        n = len(eps[i]["steps"]); t = max(0, min(int(th), n - 1)); clip += t != th
        rows.append({"id": i, "t": t, "t_hat": th, "mode": (pred[i].get("pred") or {}).get("error_mode"), "n_steps": n})
    open(D / "pointer.jsonl", "w").writelines(json.dumps(r) + "\n" for r in rows)
    ts = [r["t"] for r in rows]; rel = [r["t"] / max(1, r["n_steps"] - 1) for r in rows]
    st = {"failures": len(ids), "pointer_states": len(rows), "no_prediction": miss, "unparsed_step": nonum, "clipped": clip,
          "mean_t": sum(ts) / max(1, len(ts)), "frac_t0": sum(t == 0 for t in ts) / max(1, len(ts)), "mean_rel_pos": sum(rel) / max(1, len(rel)),
          "modes": {m: sum(r["mode"] == m for r in rows) for m in sorted({str(r["mode"]) for r in rows})}}
    json.dump(st, open(D / "pointer_stats.json", "w"), indent=1); print("pointer:", json.dumps(st))


# ----------------------------------------------------------------------------- teacher + one-step evidence
def hind_note(e, t, max_future=25, obs_chars=160, full=False):
    fut = e["steps"][t:] if full else e["steps"][t:t + max_future]; n = len(e["steps"])
    block = "\n".join(f"- step {t + i}: {s['action']} -> {s['next_obs'] if full else s['next_obs'][:obs_chars]}" for i, s in enumerate(fut))
    end = "budget exhausted" if n >= e.get("budget", 50) else "the episode ended without success"   # ALFWorld failures always exhaust the budget
    return (f"Hindsight: this attempt FAILED. After this point the agent did the following (episode length {n} steps, {end}{'; complete record' if full else ''}):\n{block}\n"
            "Using everything above, choose the best action for the CURRENT step.\n")


def render_full(goal, steps):   # Oct 10 (full-view teacher, as scripts/full_view.py): every prefix step, complete observations
    lines = [f"Task: {goal}", ""]
    for s in steps:
        lines.append(f"Observation: {s['obs']}")
        if s.get("action") is not None: lines.append(f"Action: {s['action']}")
    lines += ["", "Next action:"]
    return "\n".join(lines)


def diag_line(p):   # as self_teacher_informed.py with run_k20_round.sh's diag rows (type = "Model-A mode <mode>"; TYPE_WORDS falls back to the type)
    typ = "Model-A mode " + str(p["mode"])
    return f"Diagnosis (an error attributor trained on this agent's own trajectories): the first decisive error was at step {p['t_hat']}, type: {typ} ({typ})."


def insert_before_next(prompt, note):
    head, sep, tail = prompt.rpartition("Next action:")
    return head + note.rstrip() + "\n" + sep + tail


def replay_to(env, e, t, reset, step):
    """reset + replay steps[:t]; returns (obs, score, n_mismatch, first_mismatch)."""
    obs, info = reset(env, e["task"], e["variation"]); score = info.get("score", 0) if isinstance(info, dict) else 0; mm = 0; first = None
    for k in range(t):
        s = e["steps"][k]; obs, score, dn, _ = step(env, s["action"])
        if obs.strip() != s["next_obs"].strip() or score != s["score"]:
            mm += 1; first = first if first is not None else k
    if t > 0 and obs.strip() != e["steps"][t]["obs"].strip() and first is None: mm += 1; first = t
    return obs, score, mm, first


def cmd_teacher(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from error_attr.env.scienceworld_env import make, reset, step, valid_actions
    D = ROOT / a.dir; eps = {e["id"]: e for e in L(D / "traj.jsonl")}; ptr = [p for p in L(D / "pointer.jsonl") if p["id"] in eps]
    if a.shard: ptr = ptr[a.shard[0]::a.shard[1]]
    if a.limit: ptr = ptr[: a.limit]
    out = D / "teacher" / f"rows_s{a.shard[0] if a.shard else 0}.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():   # Oct 10 00:05 fix: rows that failed with CUDA OOM (GPU contention) are NOT final -- drop them so they are redone
        keep = [r for r in L(out) if "OutOfMemory" not in r.get("error", "")]; open(out, "w").writelines(json.dumps(r) + "\n" for r in keep)
    done = {r["id"] for r in L(out)} if out.exists() else set()
    todo = [p for p in ptr if p["id"] not in done]; print(f"{len(todo)} states to do ({len(done)} done) in shard {a.shard}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model); tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, str(ROOT / a.adapter)).eval()
    torch.manual_seed(a.seed + (a.shard[0] if a.shard else 0))
    env = make(55)

    @torch.no_grad()
    def gen(msg, n, sample, max_new):
        ids = tok(msg, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        if sample:   # exactly the student's own sampling call in sciworld_agent.py collect (T=1.0, top_p=1.0; the rest from the model's generation config)
            g = model.generate(ids, max_new_tokens=max_new, do_sample=True, temperature=1.0, top_p=1.0, num_return_sequences=n, pad_token_id=tok.pad_token_id)
        else:        # greedy, as teacher_reason_auc.generate (ALFWorld analysis)
            g = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        return [tok.decode(x[ids.shape[1]:], skip_special_tokens=True).strip() for x in g]

    @torch.no_grad()
    def logprob(msg, act):
        p = tok(msg, add_special_tokens=False).input_ids; c = tok(act, add_special_tokens=False).input_ids + [tok.eos_token_id]
        ids = torch.tensor([p + c], device=model.device); lg = model(input_ids=ids, logits_to_keep=len(c) + 1).logits[0, :-1].float()
        return torch.log_softmax(lg, -1).gather(1, torch.tensor(c, device=model.device)[:, None]).sum().item()

    f = open(out, "a"); t0 = time.time()
    for k, p in enumerate(todo):
        e = eps[p["id"]]; t = p["t"]; goal = e["goal"]; am = e["steps"][t]["action"]
        hist = [{"obs": s["obs"], "action": s["action"]} for s in e["steps"][:t]] + [{"obs": e["steps"][t]["obs"], "action": None}]
        base = render_full(goal, hist) if a.full_view else render(goal, hist, 10); hn = hind_note(e, t, full=a.full_view); dl = diag_line(p)
        rec = {"id": p["id"], "t": t, "task": e["task"], "variation": e["variation"], "a_minus": am, "t_hat": p["t_hat"], "mode": p["mode"]}
        try:
            obs_t, sc_t, mm, first = replay_to(env, e, t, reset, step); va = valid_actions(env)
            rec.update(replay_mismatch=mm, replay_first_mismatch=first, n_valid=len(va), score_before=sc_t)
            # analysis: hindsight + diagnosis, then the analysis question (greedy, <=96 tokens), as self_teacher_informed.py
            an_user = insert_before_next(base, hn + dl).replace("Next action:", ANALYSIS_Q)
            an = gen(tok.apply_chat_template([{"role": "system", "content": REVIEW_SYSTEM}, {"role": "user", "content": an_user}], tokenize=False, add_generation_prompt=True), 1, False, 96)[0]
            tmsg = chat(tok, insert_before_next(base, hn + dl + "\nAnalysis: " + an))
            samples = []; acc_cache = {}; n_draw = 0
            def env_accepts(txt):   # --accept_env (v2 proposal): an unlisted command counts as valid if the parser accepts it from the replayed state
                if txt not in acc_cache:
                    replay_to(env, e, t, reset, step); o2, _, _, _ = step(env, txt); acc_cache[txt] = not o2.strip().lower().startswith(NOINFO_PREFIX)
                return acc_cache[txt]
            while True:
                raws = [r.split("\n")[0].strip() for r in gen(tmsg, a.n_samples, True, 24)]; n_draw += len(raws)
                for r in raws:
                    m, mp = snap_safe(r, va)
                    if m is None and a.accept_env and r and env_accepts(r): m = r; samples.append({"raw": r, "action": m, "mapped": False, "env_accepted": True}); continue
                    samples.append({"raw": r, "action": m, "mapped": mp})
                cands = sorted({s["action"] for s in samples if s["action"] is not None and s["action"] != am})
                if len(cands) >= a.top or n_draw >= a.n_samples + a.max_extra: break
            lp = {c: logprob(tmsg, c) for c in cands}; top = sorted(cands, key=lambda c: -lp[c])[: a.top]
            rec.update(analysis=an, samples=samples, cand_logprob=lp, n_valid_samples=sum(s["action"] is not None for s in samples),
                       n_eq_aminus=sum(s["action"] == am for s in samples))
            # one-step evidence: each top candidate executed once after replaying the prefix from reset
            ho = hist_obs_at(e, t); res = []
            for c in top:
                _, sb, mmc, firstc = replay_to(env, e, t, reset, step); oa, sa, dn, _ = step(env, c)
                ev = sciworld_evidence(goal, ho, e["steps"][t]["obs"], oa, sb, sa, dn)
                res.append({"cand": c, "logprob": lp[c], "prob": math.exp(lp[c]), "obs_after": oa[:600], "score_before": sb, "score_after": sa, "done": dn,
                            "replay_mismatch": mmc, "replay_first_mismatch": firstc, **ev})
            rec["cands"] = res
            # the taken action's own one-step evidence (recorded next obs; for statistics only)
            st = e["steps"][t]; rec["taken_ev"] = sciworld_evidence(goal, ho, st["obs"], st["next_obs"], e["steps"][t - 1]["score"] if t > 0 else 0, st["score"],
                                                                    t == len(e["steps"]) - 1 and e["n_steps"] < e["budget"])
            rec["taken_score_after"] = st["score"]
        except torch.OutOfMemoryError:   # Oct 10 fix: never record an OOM as a result; exit non-zero so the driver retries this shard (resumes here)
            print("CUDA OOM -> exit for retry", flush=True); sys.exit(3)
        except Exception as ex:   # an env/java failure on one state: log it, rebuild the env, keep going (reported in select stats)
            import traceback; traceback.print_exc(); rec["error"] = repr(ex)[:300]
            try: env.close()
            except Exception: pass
            env = make(55)
        f.write(json.dumps(rec) + "\n"); f.flush()
        if k % 5 == 0: print(f"[{k + 1}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
    print("DONE teacher", flush=True)


# ----------------------------------------------------------------------------- selection
def cmd_select(a):
    D = ROOT / a.dir; eps = {e["id"]: e for e in L(D / "traj.jsonl")}; nf = len(L(D / "ids_fail.jsonl")); K = max(1, round(a.rho * nf))   # rho: Oct 10 K40 variant (default 0.2)
    rows = {}
    for p in sorted(glob.glob(str(D / "teacher" / "rows_s*.jsonl"))):
        for r in L(p): rows.setdefault(r["id"], r)
    rows = list(rows.values()); ok_rows = [r for r in rows if "error" not in r]
    if a.evidence in ("window", "window_aff", "window_aff2"):   # Oct 10 k-step variant (+ affordance: S_A): S / no-info replaced by the window evidence
        win = {}; key = "S3" if a.evidence == "window" else "SA"
        for p in sorted(glob.glob(str(D / a.evidence / "rows_s*.jsonl"))):
            for w in L(p): win.setdefault((w["id"], w["cand"]), w)
        miss = 0
        for r in ok_rows:
            for c in r.get("cands", []):
                w = win.get((r["id"], c["cand"]))
                if w is None or "error" in w: miss += 1; c["no_info"] = True; c["score"] = -9.0; continue
                c["score1"] = c["score"]; c["no_info1"] = c["no_info"]; c["score"] = w[key]; c["no_info"] = w["nul"]
        print(f"window evidence: {len(win)} rows; {miss} candidates without a window row (treated as no-info)")
    picks, top1 = [], []
    for r in ok_rows:
        cs = r.get("cands", [])
        if cs: top1.append((r, max(cs, key=lambda c: c["logprob"])))
        good = [c for c in cs if not c["no_info"] and c["score"] > 0]
        if good: c = max(good, key=lambda c: (c["score"], c["prob"])); picks.append((r, c, (c["score"], c["prob"])))
    env_top = [(r, c) for r, c, _ in sorted(picks, key=lambda x: x[2], reverse=True)[:K]]
    rng = random.Random(100 * a.round + a.seed); rnd = list(top1); rnd.sort(key=lambda x: x[0]["id"]); rng.shuffle(rnd); rnd = rnd[:K]
    src = str((D / "traj.jsonl").relative_to(ROOT))
    def dump(name, sel):
        open(D / name, "w").writelines(json.dumps({"kind": "corr", "src": src, "id": r["id"], "t": r["t"], "action": c["cand"], "a_minus": r["a_minus"],
                                                   "S": c["score"], "prob": c["prob"]}) + "\n" for r, c in sel)
    dump("targets_env_top.jsonl", env_top); dump("targets_rand.jsonl", rnd)
    # Oct 10 refined recipe: MARGIN (S > S of the taken action, its recorded next obs) + FAIL-PROPORTIONAL allocation over task types
    fails_by = {}
    for i in (x["id"] for x in L(D / "ids_fail.jsonl")): fails_by[eps[i]["task"]] = fails_by.get(eps[i]["task"], 0) + 1
    raw = {k: K * v / nf for k, v in fails_by.items()}; quota = {k: int(math.floor(v)) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - quota[k]), k))[: K - sum(quota.values())]: quota[k] += 1   # largest remainder
    mpicks = []
    for r in ok_rows:
        good = [c for c in r.get("cands", []) if not c["no_info"] and c["score"] > 0 and "taken_ev" in r and c["score"] > r["taken_ev"]["score"]]
        if good: c = max(good, key=lambda c: (c["score"], c["prob"])); mpicks.append((r, c, (c["score"], c["prob"])))
    mpicks.sort(key=lambda x: x[2], reverse=True); fp, used, filled = [], set(), {}
    for k in sorted(quota):
        mine = [x for x in mpicks if x[0]["task"] == k][: quota[k]]; filled[k] = len(mine)
        for x in mine: fp.append(x); used.add(x[0]["id"])
    short = K - len(fp); extra = [x for x in mpicks if x[0]["id"] not in used][: max(0, short)]
    fp += extra; fp.sort(key=lambda x: x[2], reverse=True); fp = [(r, c) for r, c, _ in fp]
    dump("targets_fp.jsonl", fp)
    dump("targets_all_top1.jsonl", sorted(top1, key=lambda x: x[0]["id"]))   # Oct 10 ablation C: teacher top-1 at EVERY attributed failure, no filter, no K cap
    fp_alloc = {k: {"failures": fails_by[k], "quota": quota[k], "filled_own": filled[k], "redistributed_in": sum(x[0]["task"] == k for x in extra)} for k in sorted(quota)}

    def neg100(r):   # the taken action a- ended the episode with score < 0 right at t* (a fatal wrong 'focus on')
        e = eps[r["id"]]; st = e["steps"][r["t"]]; return st["score"] < 0 and r["t"] == len(e["steps"]) - 1
    def describe(sel):
        return {"n": len(sel), "from_1cand_states": sum(len(r.get("cands", [])) == 1 for r, _ in sel), "fix_fatal_aminus": sum(neg100(r) for r, _ in sel), "raises_score": sum(c["score_after"] > c["score_before"] for _, c in sel),
                "aminus_focus": sum(r["a_minus"].startswith("focus on") for r, _ in sel), "cand_focus": sum(c["cand"].startswith("focus on") for _, c in sel),
                "mean_S": sum(c["score"] for _, c in sel) / max(1, len(sel)), "S_hist": {str(s): sum(c["score"] == s for _, c in sel) for s in sorted({c["score"] for _, c in sel})},
                "by_task": {t: sum(r["task"] == t for r, _ in sel) for t in sorted({r["task"] for r, _ in sel})},
                "examples": [(r["task"], r["t"], r["a_minus"], c["cand"], c["score"]) for r, c in sel[:12]]}
    allc = [c for r in ok_rows for c in r.get("cands", [])]
    ns = sum(len(r.get("samples", [])) for r in ok_rows)
    st = {"failures": nf, "K": K, "rho": a.rho, "env_top_pool": len(picks), "teacher_states": len(rows), "errors": len(rows) - len(ok_rows),
          "replay_mismatch_states": sum(r.get("replay_mismatch", 0) > 0 for r in ok_rows),
          "cand_replay_mismatch": sum(c.get("replay_mismatch", 0) > 0 for c in allc),
          "samples": ns, "valid_samples": sum(r.get("n_valid_samples", 0) for r in ok_rows), "samples_eq_aminus": sum(r.get("n_eq_aminus", 0) for r in ok_rows),
          "mapped_samples": sum(s["mapped"] for r in ok_rows for s in r.get("samples", []) if s["action"]),
          "states_by_ncands": {n: sum(len(r.get("cands", [])) == n for r in ok_rows) for n in range(a.top + 1)},
          "distinct_valid_samples_incl_aminus": {n: sum(len({x["action"] for x in r.get("samples", []) if x["action"]}) == n for r in ok_rows) for n in range(9)},
          "cands": len(allc), "cands_no_info": sum(c["no_info"] for c in allc), "cands_parser_reject": sum(c["parser_reject"] for c in allc),
          "cands_S_pos": sum(c["score"] > 0 and not c["no_info"] for c in allc), "cands_regress": sum(c["regress"] for c in allc), "cands_progress": sum(c["progress"] for c in allc),
          "states_with_positive_cand": len(picks),
          "S_hist_all_cands": {str(s): sum(c["score"] == s for c in allc) for s in sorted({c["score"] for c in allc})},
          "taken_S_mean_at_states": sum(r["taken_ev"]["score"] for r in ok_rows if "taken_ev" in r) / max(1, len(ok_rows)),
          "env_top": describe(env_top), "random": describe(rnd), "random_pool": len(top1),
          "fp": describe(fp), "fp_margin_pool": len(mpicks), "fp_shortfall_redistributed": max(0, short), "fp_alloc": fp_alloc}
    json.dump(st, open(D / "select_stats.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in st.items() if k not in ("env_top", "random", "fp", "fp_alloc")}))
    print("fp (margin + fail-proportional):", json.dumps({k: v for k, v in st["fp"].items() if k != "examples"}), "| margin pool", len(mpicks), "| redistributed", max(0, short))
    print("fp allocation (type: failures/quota/filled_own/+redistributed):", "; ".join(f"{k} {v['failures']}/{v['quota']}/{v['filled_own']}/+{v['redistributed_in']}" for k, v in fp_alloc.items()))
    print("env_top:", json.dumps(st["env_top"])); print("random:", json.dumps({k: v for k, v in st["random"].items() if k != "examples"}))


def cmd_replay_fw(a):
    """Oct 10 refined recipe: --n steps split over task types in proportion to the round's training failures (largest remainder), each quota
    filled with complete shortest wins of that type (trajectory by trajectory, the last one cut), skipping look around / inventory / task;
    quota a type cannot fill (no / too few wins) is carried to the other types in proportion to their failures (iterated)."""
    D = ROOT / a.dir; src = str((D / "traj.jsonl").relative_to(ROOT)); ALL = L(D / "traj.jsonl")
    SKIP = ("look around", "inventory", "task"); wins = sorted([e for e in ALL if e["won"]], key=lambda e: (len(e["steps"]), e["id"]))
    steps_of = lambda e: [t for t, s_ in enumerate(e["steps"]) if s_["action"].strip() not in SKIP]
    cap = {}
    for e in wins: cap[e["task"]] = cap.get(e["task"], 0) + len(steps_of(e))
    fails = {}
    for e in ALL:
        if not e["won"]: fails[e["task"]] = fails.get(e["task"], 0) + 1
    tot = sum(fails.values()); raw = {k: a.n * v / tot for k, v in fails.items()}; quota = {k: int(math.floor(v)) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - quota[k]), k))[: a.n - sum(quota.values())]: quota[k] += 1
    eff = {k: min(quota[k], cap.get(k, 0)) for k in quota}
    for _ in range(20):   # carry the unfilled quota to types with spare capacity, in proportion to their failures
        left = a.n - sum(eff.values()); open_ = [k for k in fails if cap.get(k, 0) > eff[k]]
        if left <= 0 or not open_: break
        w = sum(fails[k] for k in open_); add = {k: left * fails[k] / w for k in open_}; ad = {k: int(math.floor(v)) for k, v in add.items()}
        for k in sorted(add, key=lambda k: (-(add[k] - ad[k]), k))[: left - sum(ad.values())]: ad[k] += 1
        for k in open_: eff[k] = min(cap[k], eff[k] + ad[k])
    rows = []
    for k in sorted(eff):
        got = 0
        for e in [w_ for w_ in wins if w_["task"] == k]:
            for t in steps_of(e):
                if got >= eff[k]: break
                rows.append({"kind": "replay", "src": src, "id": e["id"], "t": t, "action": e["steps"][t]["action"]}); got += 1
            if got >= eff[k]: break
    open(D / "replay_fw.jsonl", "w").writelines(json.dumps(r) + "\n" for r in rows)
    print(f"replay_fw: {len(rows)} steps from {len({r['id'] for r in rows})} wins; quota {dict(sorted(quota.items()))}; effective (carry) {dict(sorted((k, v) for k, v in eff.items() if v))}"
          + (f"; WARNING {a.n - len(rows)} unplaced" if len(rows) < a.n else ""))


def cmd_replay(a):
    D = ROOT / a.dir; rng = random.Random(a.seed); src = str((D / "traj.jsonl").relative_to(ROOT))
    wins = sorted([e for e in L(D / "traj.jsonl") if e["won"]], key=lambda e: (len(e["steps"]), e["id"])); rows = []
    for e in wins:
        ok = [t for t, s in enumerate(e["steps"]) if s["action"].strip() not in ("look around", "inventory", "task")]
        for t in rng.sample(ok, min(a.per_ep, len(ok))): rows.append({"kind": "replay", "src": src, "id": e["id"], "t": t, "action": e["steps"][t]["action"]})
        if len(rows) >= a.n: break
    rows = rows[: a.n]; open(D / "replay.jsonl", "w").writelines(json.dumps(r) + "\n" for r in rows)
    print(f"replay: {len(rows)} steps from {len({r['id'] for r in rows})} of {len(wins)} wins")


# ----------------------------------------------------------------------------- rollout verification baseline (ablation D, Oct 10)
def cmd_rollout(a):
    """Each teacher candidate (D/teacher rows) is executed after replaying the prefix and the episode is continued GREEDILY by the policy
    (--adapter, the round's start policy; same render / safe snap as collect) until the episode budget (50 total steps) or done."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from error_attr.env.scienceworld_env import make, reset, step, valid_actions
    D = ROOT / a.dir; eps = {e["id"]: e for e in L(D / "traj.jsonl")}; rows = {}
    for p in sorted(glob.glob(str(D / "teacher" / "rows_s*.jsonl"))):
        for r in L(p): rows.setdefault(r["id"], r)
    jobs = [(r["id"], r["t"], c["cand"]) for r in sorted(rows.values(), key=lambda r: r["id"]) for c in r.get("cands", [])]
    if a.shard: jobs = jobs[a.shard[0]::a.shard[1]]
    out = D / "rollout" / f"rows_s{a.shard[0] if a.shard else 0}.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["id"], r["cand"]) for r in L(out)} if out.exists() else set()
    if out.exists() and out.stat().st_size and open(out, "rb").read()[-1:] != b"\n": open(out, "a").write("\n")
    todo = [j for j in jobs if (j[0], j[2]) not in done]; print(f"{len(todo)} candidate rollouts to do", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model); tok.pad_token = tok.pad_token or tok.eos_token
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}), str(ROOT / a.adapter)).eval()
    env = make(55); f = open(out, "a"); t0 = time.time()
    for k, (i, t, c) in enumerate(todo):
        e = eps[i]; goal = e["goal"]
        try:
            obs, sc, mm, first = replay_to(env, e, t, reset, step)
            hist = [{"obs": s_["obs"], "action": s_["action"]} for s_ in e["steps"][:t]]
            hist.append({"obs": obs, "action": c}); obs, sc, dn, _ = step(env, c); n = t + 1; acts = [c]
            while not dn and n < e.get("budget", 50):
                hist.append({"obs": obs, "action": None})
                ids = tok(chat(tok, render(goal, hist, 10)), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                with torch.no_grad(): g = model.generate(ids, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
                raw = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]
                act, _ = snap_safe(raw, valid_actions(env)); act = act or raw   # collect --snap safe: unmappable text is passed through
                hist[-1]["action"] = act; obs, sc, dn, _ = step(env, act); n += 1; acts.append(act)
            rec = {"id": i, "t": t, "cand": c, "won": sc >= 100, "final_score": sc, "n_steps": n, "replay_mismatch": mm, "actions": acts}
        except torch.OutOfMemoryError:
            print("CUDA OOM -> exit for retry", flush=True); sys.exit(3)
        except Exception as ex:
            import traceback; traceback.print_exc(); rec = {"id": i, "t": t, "cand": c, "won": False, "error": repr(ex)[:300]}
            try: env.close()
            except Exception: pass
            env = make(55)
        f.write(json.dumps(rec) + "\n"); f.flush()
        if k % 10 == 0: print(f"[{k + 1}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
    print("DONE rollout", flush=True)


def window_evidence(goal, hist_obs, cur_obs, window, score_before):
    """Oct 10 k-step evidence (NEW; the one-step sciworld_evidence is unchanged). window = [(obs, score, done), ...]: the candidate step
    first, then the greedy continuation steps kept (a continuation step that regresses is cut off before this call).
    loc: a new content unit overlapping the goal in any window observation; sat: the native score rises at any window step, or an
    acknowledged change on a goal-named object occurs; novelty: min(#new units over the window, 5)/5; nul: the candidate is a parser
    rejection, or no new unit and no sat over the window; vio: ONLY the candidate step regresses."""
    import re
    from generic_evidence import toks
    b = lambda o: sw_body(o, goal).strip()
    seen = [b(o) for o in hist_obs] + [b(cur_obs)]; known = set().union(*[toks(o) for o in seen]) if seen else set()
    gw = toks(goal.replace("Instruction:", "")); new_all = set(); sat = False; prev = score_before
    for k, (o, sc, dn) in enumerate(window):
        bo = b(o); new = toks(bo) - known - new_all; new_all |= new
        m = re.match(r"you ((?!see|arrive|are )[a-z]+(?: [a-z]+)?) the ([a-z]+)( \d+)?", bo.lower())
        rep = bool(m) and any(re.match(rf"you {re.escape(m.group(1))} the {re.escape(m.group(2) + (m.group(3) or ''))}\b", x.lower()) for x in seen)
        if sc > prev or (dn and sc >= 100) or (bool(m) and m.group(2) in gw and not rep): sat = True
        prev = sc; seen.append(bo)
    o1, s1, d1 = window[0]
    vio = s1 < score_before or (d1 and s1 < 100)
    loc = any(x.split()[0] in gw for x in new_all)
    reject = o1.strip().lower().startswith(NOINFO_PREFIX)
    nul = reject or (len(new_all) == 0 and not sat)
    S3 = 2.0 * loc + 2.0 * sat + min(len(new_all), 5) / 5.0 - 2.0 * nul - 4.0 * vio
    return {"S3": S3, "loc": loc, "sat": sat, "n_new": len(new_all), "nul": nul, "vio": vio, "parser_reject": reject}


class TargetedUnlock:
    """Oct 10 variant B v2 (targeted unlock): new valid actions created by window steps that did NOT move the agent (observation 'You move
    to' / 'You go to' / teleport, or the room name in the step's look text changes), counted only if they mention an inventory object or a
    goal entity phrase (environment object names that occur as a phrase in the goal, excluding generic names such as 'substance')."""
    GENERIC = {"substance", "agent", "air", "object", "thing", "inventory", "room", "door"}

    def __init__(self, env, goal, va0, obs0):
        import re
        self.re = re; self.goal = goal.lower(); self.prev_va = set(va0); self.targeted = set(); self.moved = []; self.goal_phrases = set()
        self.room = self._room(env.look()); self._update_goal(env)   # env.look() = free look (no step)

    def _room(self, look):
        m = self.re.search(r"this room is called the ([a-z ]+?)\.", (look or "").lower()); return m.group(1) if m else None

    def _update_goal(self, env):
        for o in env.get_possible_objects():
            ol = o.lower().strip()
            if ol and ol not in self.GENERIC and self.re.search(rf"\b{self.re.escape(ol)}\b", self.goal): self.goal_phrases.add(ol)

    def after(self, env, act, obs, info):
        ol = obs.strip().lower(); room = self._room(info.get("look", "")) if isinstance(info, dict) else None
        moved = ol.startswith(("you move to", "you go to")) or "teleport" in ol or (room is not None and self.room is not None and room != self.room)
        if room is not None: self.room = room
        va = set(valid_actions_of(env)); self._update_goal(env); self.moved.append(bool(moved))
        if not moved:
            inv = (info.get("inv", "") if isinstance(info, dict) else "").lower()
            inv_objs = {o.lower() for o in env.get_possible_objects() if o.lower() not in self.GENERIC and self.re.search(rf"\b{self.re.escape(o.lower())}\b", inv)}
            keys = self.goal_phrases | inv_objs
            for x in va - self.prev_va:
                if any(self.re.search(rf"\b{self.re.escape(k)}\b", x.lower()) for k in keys): self.targeted.add(x)
        self.prev_va = va


def valid_actions_of(env):
    from error_attr.env.scienceworld_env import valid_actions
    return valid_actions(env)


def cmd_window(a):
    """[GPU] For each teacher candidate: replay the prefix, execute it, then the round's start policy continues GREEDILY for --k - 1 steps
    (same render / chat / safe snap as collect, no hindsight). A continuation step that regresses (score drop, or the episode ends
    without success) ends the window BEFORE it (the student's own mistake is not charged to the candidate; logged as cut)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from error_attr.env.scienceworld_env import make, reset, step, valid_actions
    D = ROOT / a.dir; eps = {e["id"]: e for e in L(D / "traj.jsonl")}; rows = {}
    for p in sorted(glob.glob(str(D / "teacher" / "rows_s*.jsonl"))):
        for r in L(p): rows.setdefault(r["id"], r)
    jobs = [(r["id"], r["t"], c["cand"]) for r in sorted(rows.values(), key=lambda r: r["id"]) for c in r.get("cands", [])]
    if a.shard: jobs = jobs[a.shard[0]::a.shard[1]]
    if a.aff2: a.aff = False
    out = D / ("window_aff2" if a.aff2 else "window_aff" if a.aff else "window") / f"rows_s{a.shard[0] if a.shard else 0}.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["id"], r["cand"]) for r in L(out)} if out.exists() else set()
    if out.exists() and out.stat().st_size and open(out, "rb").read()[-1:] != b"\n": open(out, "a").write("\n")
    todo = [j for j in jobs if (j[0], j[2]) not in done]; print(f"{len(todo)} candidate windows to do (k={a.k}, affordance={a.aff})", flush=True)
    from generic_evidence import toks as _toks
    tok = AutoTokenizer.from_pretrained(a.model); tok.pad_token = tok.pad_token or tok.eos_token
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}), str(ROOT / a.adapter)).eval()
    env = make(55); f = open(out, "a"); t0 = time.time()
    for n_, (i, t, c) in enumerate(todo):
        e = eps[i]; goal = e["goal"]
        try:
            obs, sb, mm, first = replay_to(env, e, t, reset, step)
            hist = [{"obs": s_["obs"], "action": s_["action"]} for s_ in e["steps"][:t]] + [{"obs": obs, "action": c}]
            va0 = set(valid_actions(env)) if (a.aff or a.aff2) else None; va_new = set()   # --aff (Oct 10 variant B): valid actions before the candidate
            if a.aff2: tg = TargetedUnlock(env, goal, va0, obs)
            o1, s1, d1, info1 = step(env, c); win = [(o1, s1, d1)]; acts = [c]; cut = None; prev = s1; o = o1; dn = d1
            if a.aff: va_new |= set(valid_actions(env)) - va0
            if a.aff2: tg.after(env, c, o1, info1)
            regress1 = s1 < sb or (d1 and s1 < 100)
            if not regress1 and not d1:
                for k in range(a.k - 1):
                    if t + 1 + k >= e.get("budget", 50): break
                    hist.append({"obs": o, "action": None})
                    ids = tok(chat(tok, render(goal, hist, 10)), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                    with torch.no_grad(): g = model.generate(ids, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
                    raw = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]
                    act, _ = snap_safe(raw, valid_actions(env)); act = act or raw
                    hist[-1]["action"] = act; o, sc, dn, info_k = step(env, act); acts.append(act)
                    if sc < prev or (dn and sc < 100): cut = k + 1; break   # the student's own regress: window ends before this step
                    win.append((o, sc, dn)); prev = sc
                    if a.aff: va_new |= set(valid_actions(env)) - va0
                    if a.aff2: tg.after(env, act, o, info_k)
                    if dn: break
            ev = window_evidence(goal, hist_obs_at(e, t), e["steps"][t]["obs"], win, sb)
            rec = {"id": i, "t": t, "cand": c, **ev, "k_used": len(win), "cut_at": cut, "actions": acts, "score_before": sb, "scores": [w[1] for w in win],
                   "replay_mismatch": mm, "obs": [w[0][:300] for w in win]}
            if a.aff:   # aff = the window created NEW valid actions that mention a goal content unit (same toks() of the goal as loc); S_A = S_3 + aff
                gw = _toks(goal.replace("Instruction:", "")); unlocked = sorted(x for x in va_new if any(u.split()[0] in gw for u in _toks(x)))
                rec.update(aff=bool(unlocked), n_unlocked_goal=len(unlocked), n_new_valid=len(va_new), unlocked_examples=unlocked[:8], SA=ev["S3"] + 1.0 * bool(unlocked))
            if a.aff2:
                n = len(tg.targeted); aff2 = 1 <= n <= 50
                rec.update(aff=aff2, n_targeted=n, targeted_examples=sorted(tg.targeted)[:8], moved_steps=tg.moved, goal_phrases=sorted(tg.goal_phrases), SA=ev["S3"] + 1.0 * aff2)
        except torch.OutOfMemoryError:
            print("CUDA OOM -> exit for retry", flush=True); sys.exit(3)
        except Exception as ex:
            import traceback; traceback.print_exc(); rec = {"id": i, "t": t, "cand": c, "error": repr(ex)[:300]}
            try: env.close()
            except Exception: pass
            env = make(55)
        f.write(json.dumps(rec) + "\n"); f.flush()
        if n_ % 10 == 0: print(f"[{n_ + 1}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
    print("DONE window", flush=True)


def cmd_subgoal(a):
    """[GPU] Oct 10 'next-subgoal relative evidence' (r2g). For the taken action a- AND each teacher candidate at a state: replay the prefix,
    execute the action, then the round's start policy continues GREEDILY (render / chat / safe snap as collect) until the FIRST native score
    increase over the state's score (d = steps used, the action itself = 1), a regress / episode end without success, or H steps in total
    (d = inf). A candidate whose own step regresses is vio (rejected). The windowed generic evidence up to the stop is kept for tie-breaks."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from error_attr.env.scienceworld_env import make, reset, step, valid_actions
    D = ROOT / a.dir; eps = {e["id"]: e for e in L(D / "traj.jsonl")}; rows = {}
    for p in sorted(glob.glob(str(D / "teacher" / "rows_s*.jsonl"))):
        for r in L(p): rows.setdefault(r["id"], r)
    states = [r for r in sorted(rows.values(), key=lambda r: r["id"]) if r.get("cands")]
    if a.shard: states = states[a.shard[0]::a.shard[1]]
    out = D / "subgoal" / f"rows_s{a.shard[0] if a.shard else 0}.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in L(out)} if out.exists() else set()
    if out.exists() and out.stat().st_size and open(out, "rb").read()[-1:] != b"\n": open(out, "a").write("\n")
    todo = [r for r in states if r["id"] not in done]; print(f"{len(todo)} states to do (H={a.H})", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model); tok.pad_token = tok.pad_token or tok.eos_token
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}), str(ROOT / a.adapter)).eval()
    env = make(55); f = open(out, "a"); t0 = time.time()

    def run(e, t, x):
        goal = e["goal"]; obs, sb, mm, first = replay_to(env, e, t, reset, step)
        hist = [{"obs": s_["obs"], "action": s_["action"]} for s_ in e["steps"][:t]] + [{"obs": obs, "action": x}]
        o, sc, dn, _ = step(env, x); path = [x]; win = [(o, sc, dn)]; d = None; vio = sc < sb or (dn and sc < 100); stop = "vio" if vio else None
        if not vio and sc > sb: d = 1; stop = "subgoal"
        prev = sc
        while d is None and stop is None and not dn and len(path) < a.H and t + len(path) < e.get("budget", 50):
            hist.append({"obs": o, "action": None})
            ids = tok(chat(tok, render(goal, hist, 10)), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            with torch.no_grad(): g = model.generate(ids, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
            raw = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]
            act, _ = snap_safe(raw, valid_actions(env)); act = act or raw
            hist[-1]["action"] = act; o, sc, dn, _ = step(env, act); path.append(act)
            if sc < prev or (dn and sc < 100): stop = "regress"; break
            win.append((o, sc, dn))
            if sc > sb: d = len(path); stop = "subgoal"
            prev = sc
        stop = stop or ("done" if dn else "horizon")
        ev = window_evidence(goal, hist_obs_at(e, t), e["steps"][t]["obs"], win, sb)
        return {"action": x, "d": d, "vio": bool(vio), "stop": stop, "path": path, "score_before": sb, "replay_mismatch": mm, "S_win": ev["S3"], "nul": ev["nul"]}

    for k, r in enumerate(todo):
        e = eps[r["id"]]; t = r["t"]
        try:
            rec = {"id": r["id"], "t": t, "task": e["task"], "a_minus": run(e, t, r["a_minus"]), "cands": []}
            for c in r["cands"]:
                x = run(e, t, c["cand"]); x["prob"] = c["prob"]; rec["cands"].append(x)
        except torch.OutOfMemoryError:
            print("CUDA OOM -> exit for retry", flush=True); sys.exit(3)
        except Exception as ex:
            import traceback; traceback.print_exc(); rec = {"id": r["id"], "t": t, "error": repr(ex)[:300]}
            try: env.close()
            except Exception: pass
            env = make(55)
        f.write(json.dumps(rec) + "\n"); f.flush()
        if k % 5 == 0: print(f"[{k + 1}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
    print("DONE subgoal", flush=True)


def cmd_select_subgoal(a):
    """POSITIVE candidate iff not vio and d(cand) < d(a-) (inf = H + 1; a candidate that never reaches the subgoal is never positive).
    Per state the best positive candidate by (margin = d(a-) - d(cand), windowed evidence, teacher prob); global top K = round(rho x failures)."""
    D = ROOT / a.dir; nf = len(L(D / "ids_fail.jsonl")); K = max(1, round(a.rho * nf)); src = str((D / "traj.jsonl").relative_to(ROOT)); INF = a.H + 1
    rows = {}
    for p in sorted(glob.glob(str(D / "subgoal" / "rows_s*.jsonl"))):
        for r in L(p): rows.setdefault(r["id"], r)
    ok = [r for r in rows.values() if "error" not in r]; dd = lambda x: INF if x is None else x; picks = []
    for r in ok:
        da = dd(r["a_minus"]["d"]); pos = [c for c in r["cands"] if not c["vio"] and c["d"] is not None and c["d"] < da]
        if pos:
            c = max(pos, key=lambda c: (da - c["d"], c["S_win"], c["prob"])); picks.append((r, c, (da - c["d"], c["S_win"], c["prob"])))
    picks.sort(key=lambda x: x[2], reverse=True); sel = picks[:K]
    open(D / "targets_subgoal.jsonl", "w").writelines(json.dumps({"kind": "corr", "src": src, "id": r["id"], "t": r["t"], "action": c["action"], "a_minus": r["a_minus"]["action"],
                                                                 "margin": m[0], "d_cand": c["d"], "d_aminus": r["a_minus"]["d"], "S_win": c["S_win"], "prob": c["prob"]}) + "\n" for r, c, m in sel)
    import collections
    allc = [c for r in ok for c in r["cands"]]
    st = {"states": len(rows), "errors": len(rows) - len(ok), "K": K, "positive_states": len(picks), "selected": len(sel),
          "aminus_reaches_subgoal": sum(r["a_minus"]["d"] is not None for r in ok), "cands": len(allc), "cands_reach_subgoal": sum(c["d"] is not None for c in allc),
          "cands_vio": sum(c["vio"] for c in allc), "stops": dict(collections.Counter(c["stop"] for c in allc)),
          "margin_hist_selected": dict(sorted(collections.Counter(m[0] for _, _, m in sel).items())), "margin_hist_all_positive": dict(sorted(collections.Counter(m[0] for _, _, m in picks).items())),
          "by_task_selected": dict(collections.Counter(r["task"] for r, _, _ in sel))}
    json.dump(st, open(D / "select_subgoal_stats.json", "w"), indent=1); print("subgoal selection:", json.dumps(st))


def cmd_select_rollout(a):
    """ALFWorld rollout-20% baseline rule: states with >=1 winning candidate; K = round(0.2 x failures) random states among them (all if
    fewer; seed 100 x round + seed); target = the winning candidate with the higher teacher probability."""
    D = ROOT / a.dir; nf = len(L(D / "ids_fail.jsonl")); K = max(1, round(0.2 * nf)); src = str((D / "traj.jsonl").relative_to(ROOT))
    rows = {}
    for p in sorted(glob.glob(str(D / "teacher" / "rows_s*.jsonl"))):
        for r in L(p): rows.setdefault(r["id"], r)
    ro = {}
    for p in sorted(glob.glob(str(D / "rollout" / "rows_s*.jsonl"))):
        for r in L(p): ro.setdefault((r["id"], r["cand"]), r)
    passing = []
    for i, r in sorted(rows.items()):
        w = [c for c in r.get("cands", []) if ro.get((i, c["cand"]), {}).get("won")]
        if w: passing.append((r, max(w, key=lambda c: c["prob"])))
    rng = random.Random(100 * a.round + a.seed); sel = list(passing); rng.shuffle(sel); sel = sel[:K]
    open(D / "targets_roll.jsonl", "w").writelines(json.dumps({"kind": "corr", "src": src, "id": r["id"], "t": r["t"], "action": c["cand"], "a_minus": r["a_minus"],
                                                              "S": c["score"], "prob": c["prob"]}) + "\n" for r, c in sel)
    nro = len(ro); nw = sum(x.get("won", False) for x in ro.values()); ne = sum("error" in x for x in ro.values())
    print(f"rollout: {nro} candidate rollouts ({ne} errors), {nw} won; {len(passing)} states with a winning candidate; K = {K}; kept {len(sel)}; "
          f"kept with S>0 {sum(c['score'] > 0 for _, c in sel)}, raising score {sum(c['score_after'] > c['score_before'] for _, c in sel)}")


# ----------------------------------------------------------------------------- trainer
def cmd_train(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    rng = random.Random(a.seed); items = []
    for p in a.data:
        path, _, w = p.partition(":"); w = float(w) if w else 1.0   # Oct 10 18:15: optional per-file weight path:w (default 1.0 = unchanged)
        rows = L(ROOT / path)
        for r_ in rows: r_["_w"] = w
        items += rows; print(f"{path}: {len(rows)} rows, weight {w}", flush=True)
    rng.shuffle(items); eps = {}
    def ep(r):
        if r["src"] not in eps: eps[r["src"]] = {e["id"]: e for e in L(ROOT / r["src"])}
        return eps[r["src"]][r["id"]]
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True); ck = out / "ckpt"
    if (out / "adapter" / "adapter_config.json").exists(): print("already trained", flush=True); print("DONE train", flush=True); return
    start_k = 0; adapter = str(ROOT / a.adapter)
    if (ck / "state.json").exists():
        stt = json.load(open(ck / "state.json")); start_k = stt["k"]; adapter = str(ck / "adapter"); print(f"RESUME from checkpoint at item {start_k}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model); tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    base.gradient_checkpointing_enable(); base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, adapter, is_trainable=True); model.train()
    params = [p_ for p_ in model.parameters() if p_.requires_grad]; opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    if start_k: opt.load_state_dict(torch.load(ck / "opt.pt", map_location="cuda:0"))
    # Oct 11 03:30 --kl_replay (user-approved R3 test): replay rows are trained toward the START policy's full next-token distribution
    # (cross-entropy on the replayed action + kl_beta x forward KL(p_ref || p_theta) on every action token, top-64 + tail bucket);
    # correction rows keep cross-entropy.  p_ref is computed once with the start adapter before any update and cached in out/ref_kl.pt.
    refkl = {}
    if a.kl_replay:
        rf = out / "ref_kl.pt"
        if rf.exists(): refkl = torch.load(rf)
        else:
            assert not start_k, "ref_kl.pt missing on resume"
            model.eval()
            with torch.no_grad():
                for k_, r_ in enumerate(items):
                    if r_["kind"] != "replay": continue
                    e_ = ep(r_); t_ = r_["t"]
                    h_ = [{"obs": s_["obs"], "action": s_["action"]} for s_ in e_["steps"][:t_]] + [{"obs": e_["steps"][t_]["obs"], "action": None}]
                    pi = tok(chat(tok, render(e_["goal"], h_, 10)), add_special_tokens=False).input_ids; ti = tok(r_["action"], add_special_tokens=False).input_ids + [tok.eos_token_id]
                    ids_ = (pi + ti)[-a.max_len:]; nt = len(ti)
                    lg = model(input_ids=torch.tensor([ids_], device="cuda:0")).logits[0, -nt - 1:-1].float().log_softmax(-1)
                    v, i = lg.topk(64, dim=-1); refkl[k_] = (i.cpu(), v.cpu())
            torch.save(refkl, rf); model.train(); print(f"KL reference: {len(refkl)} replay rows", flush=True)
    log = open(out / "train_log.jsonl", "a"); gacc = [torch.zeros_like(p_, dtype=torch.float32) for p_ in params]; acc = {}; n = 0; nsteps = 0; t0 = time.time(); skipped = 0
    for k in range(start_k, len(items)):
        r = items[k]; e = ep(r); t = r["t"]
        hist = [{"obs": s["obs"], "action": s["action"]} for s in e["steps"][:t]] + [{"obs": e["steps"][t]["obs"], "action": None}]
        p_ids = tok(chat(tok, render(e["goal"], hist, 10)), add_special_tokens=False).input_ids; t_ids = tok(r["action"], add_special_tokens=False).input_ids + [tok.eos_token_id]
        ids = (p_ids + t_ids)[-a.max_len:]; lab = ([-100] * len(p_ids) + t_ids)[-a.max_len:]
        if k in refkl:   # KL toward the start policy on this replay row
            nt = len(t_ids); lg = model(input_ids=torch.tensor([ids], device="cuda:0")).logits[0, -nt - 1:-1].float().log_softmax(-1)
            ri, rv = refkl[k]; ri = ri.to(lg.device); rv = rv.to(lg.device); pr = rv.exp(); ls = lg.gather(-1, ri)
            tail_r = (1 - pr.sum(-1)).clamp_min(1e-8); tail_s = (1 - ls.exp().sum(-1)).clamp_min(1e-8)
            kl = ((pr * (rv - ls)).sum(-1) + tail_r * (tail_r.log() - tail_s.log())).mean()
            ce = -lg.gather(-1, torch.tensor(t_ids, device=lg.device)[:, None]).mean()   # keep the replayed action (as before) ...
            loss = ce + a.kl_beta * kl                                                    # ... plus the KL anchor to the start policy
        else:
            loss = model(input_ids=torch.tensor([ids], device="cuda:0"), labels=torch.tensor([lab], device="cuda:0")).loss
        for p_ in params: p_.grad = None
        if torch.isfinite(loss):
            (r.get("_w", 1.0) * loss / a.batch).backward()
            if all(p_.grad is None or torch.isfinite(p_.grad).all() for p_ in params):
                for g_, p_ in zip(gacc, params):
                    if p_.grad is not None: g_.add_(p_.grad.float())
                acc.setdefault(r["kind"], []).append(loss.item())
            else: skipped += 1
        else: skipped += 1
        n += 1
        if n == a.batch or k == len(items) - 1:
            for g_, p_ in zip(gacc, params): p_.grad = g_.to(p_.dtype).clone(); g_.zero_()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(); n = 0; nsteps += 1
            rec = {"k": k + 1, "of": len(items), **{f"loss_{kk}": sum(v) / len(v) for kk, v in acc.items()}, "elapsed_s": round(time.time() - t0)}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True); acc = {}
            if nsteps % a.ckpt_every == 0 and k < len(items) - 1:   # checkpoint at a batch boundary: adapter + optimizer + item index (atomic rename)
                tmp = out / "ckpt_tmp"; os.system(f"rm -rf {tmp}"); tmp.mkdir()
                model.save_pretrained(str(tmp / "adapter")); torch.save(opt.state_dict(), tmp / "opt.pt"); json.dump({"k": k + 1}, open(tmp / "state.json", "w"))
                os.system(f"rm -rf {ck}"); os.rename(tmp, ck)
    model.save_pretrained(str(out / "adapter_tmp")); tok.save_pretrained(str(out / "adapter_tmp")); os.rename(out / "adapter_tmp", out / "adapter")
    print(f"DONE train {len(items)} items (skipped non-finite {skipped})", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prep"); s.add_argument("--dir", required=True); s.add_argument("--inputs", nargs="+", required=True); s.add_argument("--expect", type=int, default=0)
    s = sub.add_parser("pointer"); s.add_argument("--dir", required=True)
    s = sub.add_parser("teacher"); s.add_argument("--dir", required=True); s.add_argument("--adapter", required=True); s.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    s.add_argument("--shard", type=int, nargs=2, default=None); s.add_argument("--n_samples", type=int, default=8); s.add_argument("--top", type=int, default=2)
    s.add_argument("--seed", type=int, default=0); s.add_argument("--limit", type=int, default=0)
    s.add_argument("--full_view", action="store_true", help="Oct 10 refined recipe: complete prefix, all future steps, full observations"); s.add_argument("--accept_env", action="store_true", help="v2 proposal (not default): unlisted but parser-accepted commands are valid"); s.add_argument("--max_extra", type=int, default=0, help="v2 proposal: extra samples (batches of n_samples) until top distinct candidates exist")
    s = sub.add_parser("select"); s.add_argument("--dir", required=True); s.add_argument("--round", type=int, required=True); s.add_argument("--rho", type=float, default=0.2); s.add_argument("--evidence", default="onestep", choices=["onestep", "window", "window_aff", "window_aff2"]); s.add_argument("--seed", type=int, default=0); s.add_argument("--top", type=int, default=2)
    s = sub.add_parser("replay"); s.add_argument("--dir", required=True); s.add_argument("--n", type=int, default=100); s.add_argument("--per_ep", type=int, default=2); s.add_argument("--seed", type=int, required=True)
    s = sub.add_parser("rollout"); s.add_argument("--dir", required=True); s.add_argument("--adapter", required=True); s.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); s.add_argument("--shard", type=int, nargs=2, default=None)
    s = sub.add_parser("window"); s.add_argument("--dir", required=True); s.add_argument("--adapter", required=True); s.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); s.add_argument("--shard", type=int, nargs=2, default=None); s.add_argument("--k", type=int, default=3); s.add_argument("--aff", action="store_true", help="Oct 10 variant B: + affordance gain (writes D/window_aff)"); s.add_argument("--aff2", action="store_true", help="Oct 10 variant B v2: targeted unlock (writes D/window_aff2)")
    s = sub.add_parser("subgoal"); s.add_argument("--dir", required=True); s.add_argument("--adapter", required=True); s.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); s.add_argument("--shard", type=int, nargs=2, default=None); s.add_argument("--H", type=int, default=9)
    s = sub.add_parser("select_subgoal"); s.add_argument("--dir", required=True); s.add_argument("--H", type=int, default=9); s.add_argument("--rho", type=float, default=0.2)
    s = sub.add_parser("select_rollout"); s.add_argument("--dir", required=True); s.add_argument("--round", type=int, required=True); s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("replay_fw"); s.add_argument("--dir", required=True); s.add_argument("--n", type=int, default=100)
    s = sub.add_parser("train"); s.add_argument("--data", nargs="+", required=True); s.add_argument("--adapter", required=True); s.add_argument("--out", required=True); s.add_argument("--kl_replay", action="store_true"); s.add_argument("--kl_beta", type=float, default=10.0)
    s.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); s.add_argument("--lr", type=float, default=1e-5); s.add_argument("--batch", type=int, default=8)
    s.add_argument("--seed", type=int, default=0); s.add_argument("--max_len", type=int, default=3072); s.add_argument("--ckpt_every", type=int, default=3)
    a = ap.parse_args(); globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
