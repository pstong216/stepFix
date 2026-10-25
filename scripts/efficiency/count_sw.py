"""Implementation-independent compute counts for ScienceWorld StepFix + baselines (CPU only, tokenizer only).
Writes eff/parts/*.json (one per component) so reruns are cheap."""
import glob, json, math, os, sys, collections
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ.setdefault("HF_HOME", "/scratch.global/pan00389/hf_cache")
ROOT = "/users/8/pan00389/multi-agent/error-attribution"; os.chdir(ROOT)
sys.path.insert(0, ROOT + "/src"); sys.path.insert(0, ROOT + "/scripts")
from sciworld_agent import render, chat
import sciworld_stepfix as SFX
from transformers import AutoTokenizer
OUT = os.path.dirname(os.path.abspath(__file__)) + "/parts"; os.makedirs(OUT, exist_ok=True)
SF = "reports/sciworld/stepfix"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")


def L(p): return [json.loads(l) for l in open(p) if l.strip()]
def ntok(texts):
    if not texts: return []
    out = []
    for i in range(0, len(texts), 2000): out += [len(x) for x in tok(texts[i:i + 2000], add_special_tokens=False).input_ids]
    return out
_eps = {}
def eps_of(src):
    if src not in _eps: _eps[src] = {e["id"]: e for e in L(src)}
    return _eps[src]


def done(name): return os.path.exists(f"{OUT}/{name}.json")
def save(name, d): json.dump(d, open(f"{OUT}/{name}.json", "w"), indent=1); print(name, json.dumps({k: v for k, v in d.items() if not isinstance(v, (dict, list))}), flush=True)


def collection(name, traj):
    if done(name): return
    E = L(traj); prompts = []; gens = []; pos = []
    for e in E:
        hist = []
        for k, s in enumerate(e["steps"]):
            hist.append({"obs": s["obs"], "action": None}); prompts.append(chat(tok, render(e["goal"], hist, 10))); gens.append(s.get("raw", s["action"]) or "")
            hist[-1]["action"] = s["action"]; pos.append((e["task"], k))
    P = ntok(prompts); G = [g + 1 for g in ntok(gens)]   # + EOS
    by = collections.defaultdict(list); byk = collections.defaultdict(list)
    for (tk, k), p in zip(pos, P): by[f"{tk}|{k}"].append(p); byk[str(k)].append(p)
    save(name, {"episodes": len(E), "wins": sum(e["won"] for e in E), "env_steps": len(P), "policy_calls": len(P), "prompt_tokens": sum(P), "gen_tokens": sum(G),
                "mean_prompt": sum(P) / len(P), "mean_gen": sum(G) / len(G),
                "prompt_by_task_step": {k: sum(v) / len(v) for k, v in by.items()}, "prompt_by_step": {k: sum(v) / len(v) for k, v in byk.items()}})


def modelA(name, reldir):
    if done(name): return
    from whowhen_eval.prompts import load_taxonomy
    from whowhen_eval.render import get_renderer
    from error_attr.attribution.ablation import build_prompt
    tax = load_taxonomy(ROOT + "/data/whowhen_pro_official"); rnd = get_renderer("alfagent")
    msgs = []; raws = []
    for i in range(6):
        for r in L(f"{reldir}/rel_s{i}.jsonl"):
            rel = dict(r); rel.update(r.get("extras") or {}); text, _, _ = build_prompt("no_observations", rel, rnd(rel), tax)
            msgs.append(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True))
        raws += [p["raw"] for p in L(f"{reldir}/modelA_pred_s{i}.jsonl")]
    P = ntok(msgs); G = [min(g + 1, 96) for g in ntok(raws)]
    save(name, {"calls": len(P), "input_tokens": sum(P), "gen_tokens": sum(G), "mean_input": sum(P) / len(P)})


def teacher(name, D):
    if done(name): return
    eps = eps_of(f"{D}/traj.jsonl"); ptr = {p["id"]: p for p in L(f"{D}/pointer.jsonl")}
    rows = {}
    for p in sorted(glob.glob(f"{D}/teacher/rows_s*.jsonl")):
        for r in L(p): rows.setdefault(r["id"], r)
    an_prompts = []; an_txt = []; tmsgs = []; n_sample_calls = []; samp_raw = []; lp_cands = []; c = collections.Counter()
    for r in rows.values():
        e = eps[r["id"]]; t = r["t"]; p = ptr[r["id"]]; c["states"] += 1
        c["env_replay_steps"] += t   # initial replay
        if "analysis" not in r: c["errors"] += 1; continue
        hist = [{"obs": s["obs"], "action": s["action"]} for s in e["steps"][:t]] + [{"obs": e["steps"][t]["obs"], "action": None}]
        base = SFX.render_full(e["goal"], hist); hn = SFX.hind_note(e, t, full=True); dl = SFX.diag_line(p)
        an_user = SFX.insert_before_next(base, hn + dl).replace("Next action:", SFX.ANALYSIS_Q)
        an_prompts.append(tok.apply_chat_template([{"role": "system", "content": SFX.REVIEW_SYSTEM}, {"role": "user", "content": an_user}], tokenize=False, add_generation_prompt=True))
        an_txt.append(r["analysis"])
        tm = chat(tok, SFX.insert_before_next(base, hn + dl + "\nAnalysis: " + r["analysis"])); tmsgs.append(tm)
        ns = len(r["samples"]); n_sample_calls.append(math.ceil(ns / 8)); c["samples"] += ns; samp_raw += [s["raw"] for s in r["samples"]]
        lp_cands.append((len(tmsgs) - 1, list(r["cand_logprob"].keys())))
        probes = {s["raw"] for s in r["samples"] if s["raw"] and (s.get("env_accepted") or s["action"] is None)}
        c["accept_probes"] += len(probes); c["env_replay_steps"] += t * len(probes); c["env_new_steps"] += len(probes)
        nc = len(r.get("cands", [])); c["cands_executed"] += nc; c["env_replay_steps"] += t * nc; c["env_new_steps"] += nc
    AP = ntok(an_prompts); AG = [min(g + 1, 96) for g in ntok(an_txt)]; TP = ntok(tmsgs); SG = [g + 1 for g in ntok(samp_raw)]
    allc = [x for _, cs in lp_cands for x in cs]; CT = [g + 1 for g in ntok(allc)]; it = iter(CT)
    lp_tokens = 0; n_lp = 0
    for j, cs in lp_cands:
        for _ in cs: lp_tokens += TP[j] + next(it); n_lp += 1
    samp_prompt = sum(TP[j] * n for j, n in enumerate(n_sample_calls))
    d = dict(c); d.update(analysis_calls=len(AP), analysis_prompt_tokens=sum(AP), analysis_gen_tokens=sum(AG), sample_calls=sum(n_sample_calls),
                          sample_prompt_tokens=samp_prompt, sample_prompt_tokens_if_per_sequence=sum(TP[j] * 8 * n for j, n in enumerate(n_sample_calls)),
                          sample_gen_tokens=sum(SG), logprob_passes=n_lp, logprob_tokens=lp_tokens, mean_teacher_prompt=sum(TP) / len(TP),
                          calls=len(AP) + sum(n_sample_calls) + n_lp, input_tokens=sum(AP) + samp_prompt + lp_tokens, gen_tokens=sum(AG) + sum(SG))
    save(name, d)


def rollout(name, D, col_part):
    if done(name): return
    cp = json.load(open(f"{OUT}/{col_part}.json")); bts = cp["prompt_by_task_step"]; bs = cp["prompt_by_step"]
    eps = eps_of(f"{D}/traj.jsonl"); R = []
    for p in sorted(glob.glob(f"{D}/rollout/rows_s*.jsonl")): R += L(p)
    c = collections.Counter(); est_prompt = 0.0; gens = []
    for r in R:
        e = eps[r["id"]]; t = r["t"]; n = r.get("n_steps", t + 1); c["rollouts"] += 1; c["won"] += bool(r.get("won"))
        c["env_replay_steps"] += t; c["env_new_steps"] += n - t; c["policy_calls"] += n - t - 1
        for k in range(t + 1, n): est_prompt += bts.get(f"{e['task']}|{k}", bs.get(str(k), cp["mean_prompt"]))
        gens += r.get("actions", [])[1:]
    G = [g + 1 for g in ntok(gens)]
    d = dict(c); d.update(prompt_tokens_est=round(est_prompt), gen_tokens=sum(G)); save(name, d)


def training(name, files, kl=False, epochs=1):
    if done(name): return
    rows = []
    for f in files: rows += L(f)
    prompts = []; tg = []; kinds = []
    for r in rows:
        e = eps_of(r["src"])[r["id"]]; t = r["t"]
        hist = [{"obs": s["obs"], "action": s["action"]} for s in e["steps"][:t]] + [{"obs": e["steps"][t]["obs"], "action": None}]
        prompts.append(chat(tok, render(e["goal"], hist, 10))); tg.append(r["action"]); kinds.append(r["kind"])
    P = ntok(prompts); T = [x + 1 for x in ntok(tg)]
    fw = [min(p + t, 3072) for p, t in zip(P, T)]
    ref = sum(f for f, k in zip(fw, kinds) if k == "replay") if kl else 0
    save(name, {"rows": len(rows), "by_kind": dict(collections.Counter(kinds)), "epochs": epochs, "forward_tokens": sum(fw) * epochs, "loss_tokens": sum(T) * epochs,
                "backward_seq_tokens": sum(fw) * epochs, "kl_ref_forward_rows": sum(k == "replay" for k in kinds) if kl else 0, "kl_ref_forward_tokens": ref,
                "mean_prompt": sum(P) / len(P), "mean_target": sum(T) / len(T)})


if __name__ == "__main__":
    collection("col_R1_x1", f"{SF}/x1/traj.jsonl")
    collection("col_R2_w2", f"{SF}/w2/traj.jsonl")
    collection("col_R3_wk3", f"{SF}/wk3/traj.jsonl")
    collection("col_x2rft", f"{SF}/x2rft/traj.jsonl")
    modelA("modelA_R1", f"{SF}/x1"); modelA("modelA_R2", f"{SF}/w2"); modelA("modelA_R3", f"{SF}/wk3")
    teacher("teacher_R1", f"{SF}/z1"); teacher("teacher_R2", f"{SF}/w2"); teacher("teacher_R3", f"{SF}/wk3")
    rollout("rollout_roll1", f"{SF}/roll1", "col_R1_x1")
    training("train_stepfix_R1", [f"{SF}/w1/targets_cap.jsonl", f"{SF}/w1/replay_ret.jsonl"])
    training("train_stepfix_R2", [f"{SF}/w2k/targets_cap.jsonl", f"{SF}/w2k/replay_ret.jsonl"], kl=True)
    training("train_stepfix_R3", [f"{SF}/wk3/targets_cap.jsonl", f"{SF}/wk3/replay_ret.jsonl"], kl=True)
    training("train_roll1", [f"{SF}/roll1/targets_roll.jsonl", f"{SF}/w1/replay_ret.jsonl"])
    training("train_rftm_R1", [f"{SF}/x1rftm/train_rows.jsonl"])
    training("train_rft_R1", [f"{SF}/x1rft/train_rows.jsonl"])
    training("train_rft_R2", [f"{SF}/x2rft/train_rows.jsonl"])
