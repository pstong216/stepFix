"""Assemble efficiency.json / efficiency.csv / tables/efficiency.tex / figures/efficiency_data.json from eff/parts."""
import csv, json, os
H = os.path.dirname(os.path.abspath(__file__)); P = H + "/parts"
PAPER = "/scratch.global/pan00389/error-attribution/paper/overleaf_draft"
J = lambda n: json.load(open(f"{P}/{n}.json"))
WINS = {"stepfix": {0: 267, 1: 307, 2: 336, 3: 353}, "rft": {1: 330, 2: 352}, "rftm": {1: 268}, "roll": {1: 327},
        "gigpo": {8: 272, 16: 286, 24: 328, 48: 392, 72: 391}, "serl": {8: 277, 16: 285, 24: 299, 48: 313}}


def zero(): return {k: 0 for k in ("episodes", "policy_env_steps", "extra_env_steps_replay", "extra_env_steps_new", "policy_calls", "policy_prompt_tokens",
                                    "policy_gen_tokens", "extra_calls", "extra_input_tokens", "extra_gen_tokens", "train_rows", "train_forward_tokens",
                                    "loss_tokens", "bwd_seq_tokens")}
def add(a, b):
    for k, v in b.items(): a[k] = a.get(k, 0) + v
    return a
def col(n):
    c = J(n); return {"episodes": c["episodes"], "policy_env_steps": c["env_steps"], "policy_calls": c["policy_calls"], "policy_prompt_tokens": c["prompt_tokens"], "policy_gen_tokens": c["gen_tokens"]}
def ma(n):
    c = J(n); return {"extra_calls": c["calls"], "extra_input_tokens": c["input_tokens"], "extra_gen_tokens": c["gen_tokens"]}
def te(n):
    c = J(n); return {"extra_calls": c["calls"], "extra_input_tokens": c["input_tokens"], "extra_gen_tokens": c["gen_tokens"],
                      "extra_env_steps_replay": c["env_replay_steps"], "extra_env_steps_new": c["env_new_steps"]}
def tr(n):
    c = J(n); return {"train_rows": c["rows"], "train_forward_tokens": c["forward_tokens"] + c["kl_ref_forward_tokens"], "loss_tokens": c["loss_tokens"], "bwd_seq_tokens": c["backward_seq_tokens"]}


rows = []
def emit(method, budget, cum, wins, per_round=None, notes=""):
    d = dict(cum); d["fte"] = d["policy_prompt_tokens"] + d["policy_gen_tokens"] + d["extra_input_tokens"] + d["extra_gen_tokens"] + d["train_forward_tokens"] + 2 * d["bwd_seq_tokens"]
    rows.append({"method": method, "budget": budget, "wins_848": wins, **d, "notes": notes, **({"per_round": per_round} if per_round else {})})


# StepFix main chain: R1 = x1 collection + x1 Model A + z1 teacher + w1 data; R2 = w2 collection/Model A/teacher + w2k data (KL); R3 = wk3 (KL)
cum = zero(); spec = [("R1", "col_R1_x1", "modelA_R1", "teacher_R1", "train_stepfix_R1"), ("R2", "col_R2_w2", "modelA_R2", "teacher_R2", "train_stepfix_R2"),
                      ("R3", "col_R3_wk3", "modelA_R3", "teacher_R3", "train_stepfix_R3")]
for i, (r, c, m, t, x) in enumerate(spec, 1):
    pr = add(add(add(add(zero(), col(c)), ma(m)), te(t)), tr(x)); add(cum, pr)
    emit("StepFix", r, cum, WINS["stepfix"][i], per_round=dict(pr), notes="extra = Model A + teacher (analysis, sampling with shared prompt counted once per call, logprob scoring); R2/R3 train forward incl. KL reference pass on replay rows")
cum = zero()
for i, (c, x) in enumerate([("col_R1_x1", "train_rft_R1"), ("col_x2rft", "train_rft_R2")], 1):
    pr = add(add(zero(), col(c)), tr(x)); add(cum, pr); emit("RFT (all success steps)", f"R{i}", cum, WINS["rft"][i], per_round=dict(pr))
emit("RFT (matched rows)", "R1", add(add(zero(), col("col_R1_x1")), tr("train_rftm_R1")), WINS["rftm"][1])
ro = J("rollout_roll1")
roll = add(add(add(add(zero(), col("col_R1_x1")), ma("modelA_R1")), te("teacher_R1")), tr("train_roll1"))
add(roll, {"extra_calls": ro["policy_calls"], "extra_input_tokens": ro["prompt_tokens_est"], "extra_gen_tokens": ro["gen_tokens"],
           "extra_env_steps_replay": ro["env_replay_steps"], "extra_env_steps_new": ro["env_new_steps"]})
emit("Rollout verification", "R1", roll, WINS["roll"][1], notes=f"{ro['rollouts']} candidate rollouts, {ro['env_new_steps']} post-prefix env steps, {ro['policy_calls']} greedy policy calls; rollout prompt tokens ESTIMATED from per-(task, step) mean prompt length of the R0 collection")
for name, lab in (("gigpo", "GiGPO"), ("serl", "SERL")):
    st = J(f"rl_{name}")
    for S in sorted(WINS[name]):
        sub = [r for r in st if r["step"] <= S]
        if len(sub) < S: continue
        tot = sum(r["total_tokens"] for r in sub); resp = sum(r["resp_tokens"] for r in sub); pro = sum(r["prompt_tokens"] for r in sub); turns = sum(r["turns"] for r in sub)
        d = zero(); d.update(episodes=64 * S, policy_env_steps=turns, policy_calls=turns, policy_prompt_tokens=round(pro), policy_gen_tokens=round(resp),
                             train_rows=turns, loss_tokens=round(resp), bwd_seq_tokens=round(tot))
        if name == "gigpo": d["train_forward_tokens"] = round(3 * tot)   # old-logprob + reference + update forward
        else:
            d["train_forward_tokens"] = round(2 * tot)                  # old-logprob + update forward (no KL loss)
            d["extra_calls"] = turns; d["extra_input_tokens"] = sum(r.get("teacher_fwd_tokens", 0) for r in sub)   # feedback-conditioned teacher scoring pass
        emit(lab, f"step {S}", d, WINS[name][S], notes="verl metrics: perf/total_num_tokens (exact), prompt/response_length means x turns (rollout/<step>.jsonl lines)")

json.dump({"scienceworld": rows, "r0_wins": 267, "n_eval": 848, "parts": {n[:-5]: J(n[:-5]) for n in sorted(os.listdir(P)) if n.endswith(".json") and not n.startswith("rl_") and not n.startswith("col_")}},
          open(f"{H}/efficiency.json", "w"), indent=1)
keys = [k for k in rows[0] if k not in ("per_round",)]
with open(f"{H}/efficiency.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); [w.writerow(r) for r in rows]

# figure data (self-contained copy in the paper repo)
json.dump([{k: r[k] for k in ("method", "budget", "episodes", "loss_tokens", "bwd_seq_tokens", "wins_848")} for r in rows], open(f"{PAPER}/figures/efficiency_data.json", "w"), indent=1)


def fm(x):
    if x == 0: return "0"
    if x >= 1e6: return f"{x / 1e6:.1f}M" if x < 1e8 else f"{x / 1e6:.0f}M"
    if x >= 1e3: return f"{x / 1e3:.1f}k" if x < 1e5 else f"{x / 1e3:.0f}k"
    return f"{x:,}"
order = [("StepFix", "R1"), ("StepFix", "R2"), ("StepFix", "R3"), None, ("RFT (all success steps)", "R1"), ("RFT (all success steps)", "R2"), ("RFT (matched rows)", "R1"),
         ("Rollout verification", "R1"), None, ("GiGPO", "step 24"), ("GiGPO", "step 48"), ("SERL", "step 24"), ("SERL", "step 48")]
lab = {"StepFix": r"\ours{}", "RFT (all success steps)": "RFT, all steps", "RFT (matched rows)": "RFT, matched", "Rollout verification": "Rollout verif.",
       "GiGPO": "GiGPO", "SERL": "SERL"}
R = {(r["method"], r["budget"]): r for r in rows}
L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
     r" & & & Policy & Extra & \multicolumn{2}{c}{Backprop tok.} & Train & Wins \\ \cmidrule(lr){6-7}",
     r"Method & Budget & Episodes & gen.\ tok. & infer.\ tok. & loss & seq. & rows & /848 \\", r"\midrule"]
for o in order:
    if o is None: L.append(r"\midrule"); continue
    r = R[o]; b = o[1].replace("step ", "")
    L.append(f"{lab[o[0]]} & {b} & {r['episodes']:,} & {fm(r['policy_gen_tokens'])} & {fm(r['extra_input_tokens'] + r['extra_gen_tokens'])} & {fm(r['loss_tokens'])} & {fm(r['bwd_seq_tokens'])} & {r['train_rows']:,} & {r['wins_848']} \\\\")
L += [r"\bottomrule", r"\end{tabular}}",
      r"\caption{Implementation-independent compute on ScienceWorld, cumulative to each budget (R$k$ = round $k$, 1,483 collection episodes per round; GiGPO/SERL budgets are optimizer steps of 64 episodes). RFT, all steps: every success step; RFT, matched: as many rows as \ours{} R1. "
      r"Policy gen.\ tok.: tokens decoded by the policy during collection (prompt prefill is reported in the text). "
      r"Extra infer.\ tok.: input + generated tokens of model calls outside collection: for \ours{}, the attributor and the self-teacher (analysis, sampling, log-prob scoring); "
      r"rollout verification adds 24.4k greedy policy calls; for SERL, the feedback-conditioned teacher scoring pass. "
      r"Backprop tok.: loss = tokens that receive a loss (action tokens); seq.\ = all tokens in the backward pass (prompt + action). "
      r"Training rows for RL = all generated turns, each trained on every step. Wall-clock is not compared because the engines differ.}",
      r"\label{tab:efficiency}", r"\end{table}"]
open(f"{PAPER}/tables/efficiency.tex", "w").write("\n".join(L) + "\n")

for r in rows:
    print(f"{r['method'][:22]:22s} {r['budget']:8s} ep {r['episodes']:5d} gen {fm(r['policy_gen_tokens']):>6s} pre {fm(r['policy_prompt_tokens']):>6s} extra {fm(r['extra_input_tokens'] + r['extra_gen_tokens']):>6s} "
          f"envx {r['extra_env_steps_replay']}+{r['extra_env_steps_new']} fwd {fm(r['train_forward_tokens']):>6s} loss {fm(r['loss_tokens']):>6s} seq {fm(r['bwd_seq_tokens']):>6s} rows {r['train_rows']:6d} FTE {fm(r['fte']):>6s} wins {r['wins_848']}")
