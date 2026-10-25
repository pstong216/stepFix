"""HINT-SD (Yeo et al., EMNLP Findings 2026, arXiv 2605.17873) -- ONLINE, budget-matched re-implementation in our ALFWorld framework
(Oct 8; user-approved "option B").  Mirrors the official trainer (github.com/wgcyeo/HINT-SD, read-only copy in
/scratch.global/pan00389/error-attribution/third_party_ro/HINT-SD, cited as file:line below) inside our own loop:

  every optimizer step   B tasks x 4 rollouts with the CURRENT student (our admissible-command policy, budget 50, T=1.0, the exact
                         prompt / sampling of alfworld_collect.py; envs stepped in parallel worker processes, policy batched on the GPU)
  failed rollouts only   the EMA TEACHER adapter analyses the whole failed trajectory with the official multi-step prompt (ALFWorld
                         wording) -> <= 3 (step, feedback) pairs, T=0.6, top_p=0.95, <= 512 new tokens; official parser; unparseable -> skip
                         (trainer.py:514-526, 582-605, 986-1080; appworld_feedback.py:20-61, 87-165)
  teacher                same model, EMA adapter, prompt = the student's prompt at that step + the official feedback turn
                         "Consider the following feedback to solve this step:\\n\\n{hint}" (trainer.py:619-629)
  loss                   token-level reverse KL(student || stop-grad teacher) on the sampled action's tokens at the selected steps,
                         student top-100 + tail bucket, token-mean per task (= HF micro-batch), / #tasks per step (self_distillation.py:26-66,
                         trainer.py:736-800, 1738-1791)
  teacher EMA            teacher <- 0.999 teacher + 0.001 student after every optimizer step (trainer.py:53-64, 242-254)
Budget / start / eval are ours: start = SFT student LoRA (kept training), ALFWorld train games 0..1080 shuffled (seed 0), one epoch,
4 rollouts per task (~4324 episodes = our 3-round budget); adapters saved at 1323 / 2823 / all episodes (ckpt1..3/adapter).
Resumable: state/ (student + teacher adapters, AdamW + scheduler state, counters) is rewritten every --save_every steps.

    python scripts/hintsd_online.py --out reports/alfworld/self_attr_rounds/a3b_hintsd_online
"""
import argparse, json, math, os, random, shutil, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("ERROR_ATTR_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
# NOTE: heavy imports (torch / transformers / alfworld) live inside functions so that spawned env workers stay light.

FB_TEMPLATE = "Consider the following feedback to solve this step:\n\n{hint}"          # trainer.py:628 (verbatim)
# official multi-step analyzer prompt (appworld_feedback.py:87-110), "AppWorld tool-use" -> "ALFWorld household-task" (as scripts/hintsd.py)
ANALYZER_SYSTEM = ("You analyze failed ALFWorld household-task trajectories. "
                   "Identify up to {k} problematic steps where the agent made mistakes "
                   "(between 1 and {k} steps total, ordered earliest first). "
                   "For EACH problematic step, write a distinct correction in less than three sentences "
                   "that targets only that step's mistake. Output valid JSON only, no other text.")
ANALYZER_USER = ("Review this failed ALFWorld attempt and identify the problematic steps. "
                 "Return one (step, feedback) pair per problematic step.\n\n"
                 "{traj}\n\n"
                 "Output format (JSON only, no other text):\n"
                 '{{"failures": [{{"step": <1-indexed step number>, '
                 '"feedback": "<correction for this step>"}}, ...]}}')


# =============================================================== environment workers (one ALFWorld game per process at a time)
def _env_worker(conn, root, budget):
    sys.path.insert(0, os.path.join(root, "src")); os.environ.setdefault("ERROR_ATTR_ROOT", root)
    from error_attr.env.alfworld_env import make_env
    env = None
    while True:
        try:
            msg = conn.recv()
        except (EOFError, KeyboardInterrupt):
            break
        try:
            if msg[0] == "reset":
                if env is not None:
                    try: env.close()
                    except Exception: pass
                env = make_env(msg[1], budget + 1)            # same as alfworld_collect.py: max_episode_steps = budget + 1
                obs, info = env.reset()
                conn.send(("ok", obs[0], list(info["admissible_commands"][0])))
            elif msg[0] == "step":
                obs, _, dn, info = env.step([msg[1]])
                conn.send(("ok", obs[0], bool(dn[0]), bool(info["won"][0]), list(info["admissible_commands"][0])))
            elif msg[0] == "close":
                break
        except Exception as e:  # report and keep serving
            try: conn.send(("err", repr(e)))
            except Exception: break
    try:
        if env is not None: env.close()
    except Exception:
        pass


class EnvPool:
    """n worker processes; each is restarted after `recycle` episodes (TextWorld leaks ~45 MB of host RAM per env) or on a hang/crash."""

    def __init__(self, n, budget, recycle=25, timeout=180):
        import multiprocessing as mp
        self.ctx = mp.get_context("spawn"); self.n, self.budget, self.recycle, self.timeout = n, budget, recycle, timeout
        self.w = [None] * n; self.uses = [0] * n; self.restarts = 0
        for i in range(n): self._start(i)

    def _start(self, i):
        a, b = self.ctx.Pipe()
        p = self.ctx.Process(target=_env_worker, args=(b, str(ROOT), self.budget)); p.start(); b.close()
        self.w[i] = (p, a); self.uses[i] = 0

    def _kill(self, i):
        p, c = self.w[i]
        try: c.send(("close",))
        except Exception: pass
        p.join(timeout=3)
        if p.is_alive(): p.kill(); p.join(timeout=5)
        try: c.close()
        except Exception: pass

    def restart(self, i):
        self._kill(i); self._start(i); self.restarts += 1

    def recycle_if_needed(self):
        for i in range(self.n):
            if self.uses[i] >= self.recycle or not self.w[i][0].is_alive(): self.restart(i)

    def request(self, reqs):
        """reqs {worker: msg}: send all, then gather (workers run in parallel). A timed-out / crashed worker is restarted."""
        for i, m in reqs.items(): self.w[i][1].send(m)
        out = {}; deadline = time.time() + self.timeout
        for i in reqs:
            c = self.w[i][1]
            try:
                r = c.recv() if c.poll(max(0.05, deadline - time.time())) else ("err", "timeout")
            except (EOFError, OSError):
                r = ("err", "worker died")
            if r[0] != "ok": self.restart(i)
            out[i] = r
        return out

    def close(self):
        for i in range(self.n): self._kill(i)


# =============================================================== policy: batched admissible-command scoring (tree attention)
_CMD_CACHE = {}


def _cmd_ids(tok, c):
    if c not in _CMD_CACHE: _CMD_CACHE[c] = tok(c, add_special_tokens=False).input_ids
    return _CMD_CACHE[c]


def score_batch(model, tok, prompts, cmd_lists, max_tokens=16384, lm_chunk=2048):
    import torch
    with torch.no_grad():
        return _score_batch(model, tok, prompts, cmd_lists, max_tokens, lm_chunk)


def _score_batch(model, tok, prompts, cmd_lists, max_tokens, lm_chunk):
    """log p(cmd | prompt) = sum of the command's token log-probs, for every admissible command of every env -- the same quantity as
    LLMPolicy.score_admissible_commands_cached, but for many envs in one forward: each row is prompt + the tails of all commands, and a
    4D mask lets every command tail attend to the prompt and to its own earlier tokens only (positions continue from the prompt)."""
    import torch
    dev = next(model.parameters()).device; pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    P = tok(prompts, add_special_tokens=False).input_ids
    C = [[_cmd_ids(tok, c) for c in cl] for cl in cmd_lists]
    lens = [len(P[i]) + sum(max(0, len(c) - 1) for c in C[i]) for i in range(len(P))]
    chunks, cur, curmax = [], [], 0
    for i in sorted(range(len(P)), key=lambda i: lens[i]):
        m = max(curmax, lens[i])
        if cur and m * (len(cur) + 1) > max_tokens: chunks.append(cur); cur, m = [], lens[i]
        cur.append(i); curmax = m
    if cur: chunks.append(cur)
    base = model.get_base_model(); res = [None] * len(P); n_tok = 0
    for ch in chunks:
        N, T = len(ch), max(lens[i] for i in ch); n_tok += N * T
        ids = torch.full((N, T), pad, dtype=torch.long); pos = torch.zeros((N, T), dtype=torch.long); seg = torch.full((N, T), -2, dtype=torch.long)
        read_r, read_p, tq, tt, to, owners = [], [], [], [], [], []
        for r, i in enumerate(ch):
            L = len(P[i]); ids[r, :L] = torch.tensor(P[i]); pos[r, :L] = torch.arange(L); seg[r, :L] = -1
            q0 = len(read_r); read_r.append(r); read_p.append(L - 1); off = L
            for k, c in enumerate(C[i]):
                o = len(owners); owners.append((i, k)); tq.append(q0); tt.append(c[0]); to.append(o)
                for j, tkn in enumerate(c[:-1]):
                    ids[r, off + j] = tkn; pos[r, off + j] = L + j; seg[r, off + j] = k
                    read_r.append(r); read_p.append(off + j); tq.append(len(read_r) - 1); tt.append(c[j + 1]); to.append(o)
                off += max(0, len(c) - 1)
        seg = seg.to(dev); ar = torch.arange(T, device=dev)
        mask = (ar[None, :] <= ar[:, None])[None] & ((seg[:, None, :] == -1) | (seg[:, :, None] == seg[:, None, :]))
        hs = base.model(input_ids=ids.to(dev), attention_mask=mask[:, None], position_ids=pos.to(dev), use_cache=False).last_hidden_state
        sel = hs[torch.tensor(read_r, device=dev), torch.tensor(read_p, device=dev)]
        tq_t, tt_t, to_t = (torch.tensor(x, device=dev) for x in (tq, tt, to))
        lp_tok = torch.zeros(len(tq), device=dev)
        for s in range(0, sel.shape[0], lm_chunk):
            ls = torch.log_softmax(base.lm_head(sel[s: s + lm_chunk]).float(), -1)
            m = (tq_t >= s) & (tq_t < s + lm_chunk)
            lp_tok[m] = ls[tq_t[m] - s, tt_t[m]]
        sums = torch.zeros(len(owners), device=dev).index_add_(0, to_t, lp_tok).tolist()
        for o, (i, k) in enumerate(owners):
            if res[i] is None: res[i] = [0.0] * len(C[i])
            res[i][k] = sums[o]
    return res, n_tok


def rollout(pool, model, tok, games, n_rollouts, budget, temperature, gen, step, max_tokens):
    """B tasks x n_rollouts episodes in parallel, exactly the alfworld_collect.py episode loop (admissible-command policy, sampling from
    softmax(logp / T) over the admissible commands, <= budget actions, history window 10)."""
    import torch
    from error_attr.env.alfworld_env import task_goal
    from error_attr.policy.alfworld_agent import ALFWorldHistory, InteractionStep, LLMPolicy
    pol = LLMPolicy(model, tok)
    eps = [{"id": f"hsdon_s{step:04d}_{Path(gf).parent.name}_r{k}", "game_file": gf, "task": ti, "rollout": k, "steps": [], "won": False,
            "done": False} for ti, gf in enumerate(games) for k in range(n_rollouts)]
    assert len(eps) <= pool.n
    pool.recycle_if_needed()
    rep = pool.request({i: ("reset", e["game_file"]) for i, e in enumerate(eps)}); hist = {}
    for i, e in enumerate(eps):
        pool.uses[i] += 1
        if rep[i][0] != "ok": e.update(done=True, error=f"reset: {rep[i][1]}"); continue
        e["initial_obs"] = rep[i][1]; e["goal"] = task_goal(rep[i][1])
        hist[i] = ALFWorldHistory(goal=e["goal"]); hist[i].steps.append(InteractionStep(observation=rep[i][1], admissible_commands=rep[i][2]))
    n_fwd_tok = 0; env_s = 0.0; gpu_s = 0.0
    for t in range(budget):
        act = [i for i, e in enumerate(eps) if not e["done"]]
        if not act: break
        g0 = time.time()
        lps, nt = score_batch(model, tok, [pol._build_chat_prompt(hist[i]) for i in act], [hist[i].steps[-1].admissible_commands for i in act], max_tokens)
        n_fwd_tok += nt; actions = {}
        for i, lp in zip(act, lps):
            cmds = hist[i].steps[-1].admissible_commands
            if temperature <= 0: actions[i] = cmds[max(range(len(cmds)), key=lambda j: lp[j])]
            else: actions[i] = cmds[torch.multinomial(torch.softmax(torch.tensor(lp) / temperature, 0), 1, generator=gen).item()]
        g1 = time.time(); gpu_s += g1 - g0
        rep = pool.request({i: ("step", actions[i]) for i in act}); env_s += time.time() - g1
        for i in act:
            e, cur = eps[i], hist[i].steps[-1]
            if rep[i][0] != "ok": e.update(done=True, error=f"step {t}: {rep[i][1]}"); continue
            cur.action = actions[i]; _, nxt, dn, won, adm = rep[i]
            e["steps"].append({"obs": cur.observation, "admissible": list(cur.admissible_commands), "action": actions[i], "next_obs": nxt})
            if dn: e.update(done=True, won=bool(won)); continue
            hist[i].steps.append(InteractionStep(observation=nxt, admissible_commands=adm))
    for e in eps:
        e["n_steps"] = len(e["steps"]); e["budget"] = budget; e["temperature"] = temperature; e.pop("done", None)
    return eps, {"fwd_tokens": n_fwd_tok, "sec_env": env_s, "sec_policy": gpu_s}


# =============================================================== analyzer (official multi-step hindsight prompt + parser)
def render_for_detection(e, sys_prompt, first_user, max_chars=12000):
    """ALFWorld version of render_episode_for_detection (appworld_feedback.py:10-61): header, initial prompt as the agent saw it, then
    'Step k / Assistant action / Environment output / Step reward / Terminated' blocks, tail-clipped to max_chars."""
    initial = f"SYSTEM:\n{sys_prompt.strip()}\n\nUSER:\n{first_user.strip()}"
    blocks = []
    for k, s in enumerate(e["steps"]):
        last = k == len(e["steps"]) - 1
        blocks.append("\n".join([f"Step {k + 1}", f"Assistant action:\n{s['action']}", f"Environment output:\n{s['next_obs'].strip()}",
                                 f"Step reward: {1.0 if (last and e['won']) else 0.0}", f"Terminated: {bool(last and e['won'])}"]))
    traj = "\n\n".join(blocks)
    if max_chars > 0 and len(traj) > max_chars: traj = "[earlier trajectory omitted]\n" + traj[-max_chars:]   # clip_text_tail
    parts = [f"Task id: {Path(e['game_file']).parent.name}", "Dataset: alfworld", f"Final reward: {1.0 if e['won'] else 0.0}",
             f"Success: {e['won']}", f"Terminated: {e['won']}", f"Initial task prompt:\n{initial}", f"Attempt trajectory:\n{traj}"]
    return "\n\n".join(parts)


def remove_thinking_trace(text):   # self_distillation.py:9-12
    import re
    result = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return re.sub(r"^.*?</think>\s*", "", result, flags=re.DOTALL)


def parse_multi_step_failure_json(raw_text, num_steps, max_steps=3):   # appworld_feedback.py:113-165 (verbatim logic)
    cleaned = remove_thinking_trace(str(raw_text)).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start: return None
    try: parsed = json.loads(cleaned[start: end + 1])
    except json.JSONDecodeError: return None
    if not isinstance(parsed, dict): return None
    raw_pairs = parsed.get("failures"); aggregate_feedback = None
    if not isinstance(raw_pairs, list):
        legacy_feedback = parsed.get("feedback")
        if isinstance(legacy_feedback, str) and legacy_feedback.strip(): aggregate_feedback = legacy_feedback.strip()
        legacy_steps = parsed.get("failure_steps")
        if not isinstance(legacy_steps, list):
            legacy_single = parsed.get("failure_step")
            if isinstance(legacy_single, (int, float)): legacy_steps = [legacy_single]
        if isinstance(legacy_steps, list) and aggregate_feedback:
            raw_pairs = [{"step": x, "feedback": aggregate_feedback} for x in legacy_steps if isinstance(x, (int, float))]
        else:
            return None
    pairs, seen = [], set()
    for entry in raw_pairs:
        if not isinstance(entry, dict): continue
        raw_step, raw_feedback = entry.get("step"), entry.get("feedback", "")
        if not isinstance(raw_step, (int, float)) or not isinstance(raw_feedback, str): continue
        idx = int(raw_step) - 1; feedback = raw_feedback.strip()
        if idx < 0 or idx >= num_steps or not feedback or idx in seen: continue
        seen.add(idx); pairs.append((idx, feedback))
    if not pairs: return None
    pairs.sort(key=lambda item: item[0])
    return pairs[:max_steps] if max_steps > 0 else pairs


def generate_batch(model, tok, prompts, max_new_tokens, temperature, top_p, bs):
    """Sampling as the official vLLM SamplingParams(temperature, top_p, max_tokens) (trainer.py:519-525): no top-k, no repetition penalty
    (the HF generation_config defaults of Qwen2.5 -- top_k 20, repetition_penalty 1.05 -- are overridden)."""
    import torch
    dev = next(model.parameters()).device; texts, n_gen = [], 0
    eos = {tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    old = tok.padding_side; tok.padding_side = "left"
    try:
        for s in range(0, len(prompts), bs):
            x = tok(prompts[s: s + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            o = model.generate(**x, do_sample=True, temperature=temperature, top_p=top_p, top_k=0, repetition_penalty=1.0,
                               max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, use_cache=True)
            for row in o[:, x.input_ids.shape[1]:].tolist():
                n = next((j + 1 for j, t in enumerate(row) if t in eos), len(row)); n_gen += n
                texts.append(tok.decode(row[:n], skip_special_tokens=True).strip())
    finally:
        tok.padding_side = old
    return texts, n_gen


# =============================================================== distillation loss (self_distillation.py, verbatim)
def _append_tail_log_prob(log_probs):
    import torch
    log_sum = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_sum = torch.clamp(log_sum, max=-1e-7)
    tail_log_prob = torch.log(-torch.expm1(log_sum))
    return torch.cat([log_probs, tail_log_prob], dim=-1)


def _renormalize_log_probs(log_probs):
    import torch
    return log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)


def select_student_topk_log_probs(student_log_probs, teacher_log_probs, topk, add_tail):
    import torch
    if topk is None: return student_log_probs, teacher_log_probs
    topk = max(1, min(int(topk), student_log_probs.size(-1)))
    student_topk_log_probs, topk_indices = torch.topk(student_log_probs, k=topk, dim=-1)
    teacher_topk_log_probs = torch.gather(teacher_log_probs, dim=-1, index=topk_indices)
    if add_tail: return _append_tail_log_prob(student_topk_log_probs), _append_tail_log_prob(teacher_topk_log_probs)
    return _renormalize_log_probs(student_topk_log_probs), _renormalize_log_probs(teacher_topk_log_probs)


def compute_distillation_loss(student_log_probs, teacher_log_probs, completion_mask, distillation_topk, distillation_add_tail):
    import torch.nn.functional as F
    s, t = select_student_topk_log_probs(student_log_probs, teacher_log_probs, distillation_topk, distillation_add_tail)
    tokenwise_loss = F.kl_div(t, s, reduction="none", log_target=True)   # = p_s * (log p_s - log p_t): reverse KL(student || teacher)
    per_token_loss = tokenwise_loss.sum(dim=-1)
    loss_mask = completion_mask.to(dtype=per_token_loss.dtype)
    valid_tokens = loss_mask.sum().clamp(min=1.0)
    return (per_token_loss * loss_mask).sum() / valid_tokens


def completion_log_probs(model, prompt_ids, comp_ids):
    """log-softmax over the full vocabulary at the completion positions (trainer.py:639-674: prompt + completion[:-1], logits_to_keep)."""
    import torch
    dev = next(model.parameters()).device
    ids = torch.tensor([prompt_ids + comp_ids[:-1]], device=dev)
    keep = torch.arange(len(prompt_ids) - 1, len(prompt_ids) - 1 + len(comp_ids), device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):   # HF Trainer bf16=True (train.py:159)
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=keep).logits
    return torch.log_softmax(logits.float(), dim=-1)   # [1, n_comp, V]


# =============================================================== adapters / state
def student_teacher_pairs(model):
    pm = dict(model.named_parameters()); pairs = []
    for n, p in pm.items():
        if ".student." in n: pairs.append((p, pm[n.replace(".student.", ".teacher.", 1)]))
    assert pairs, "no student adapter parameters"
    return pairs


def save_adapter(model, tok, name, target):
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{name}_save_", dir=str(target.parent)))
    try:
        model.save_pretrained(str(tmp), selected_adapters=[name], safe_serialization=True)
        src = tmp / name if (tmp / name).is_dir() else tmp
        new = target.with_name(target.name + ".new"); shutil.rmtree(new, ignore_errors=True); new.mkdir()
        for f in src.iterdir():
            if f.is_file(): shutil.copy2(f, new / f.name)
        cfg = json.loads((new / "adapter_config.json").read_text()); cfg["inference_mode"] = True
        (new / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
        tok.save_pretrained(str(new))
        if target.exists(): shutil.rmtree(target)
        new.rename(target)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def load_adapter_into(model, name, src):
    from peft import load_peft_weights, set_peft_model_state_dict
    res = set_peft_model_state_dict(model, load_peft_weights(str(src), device="cpu"), adapter_name=name)   # trainer.py:286-295
    bad = [k for k in getattr(res, "unexpected_keys", []) if "lora" in k]
    assert not bad, f"unexpected keys loading {src}: {bad[:3]}"


def checksum(pairs):
    import torch
    with torch.no_grad():
        return (round(sum(s.double().abs().sum().item() for s, _ in pairs), 6), round(sum(t.double().abs().sum().item() for _, t in pairs), 6),
                float(max((s - t).abs().max().item() for s, t in pairs)))


def save_state(out, model, tok, opt, sched, meta):
    import torch
    st = out / "state.tmp"; shutil.rmtree(st, ignore_errors=True); st.mkdir(parents=True)
    save_adapter(model, tok, "student", st / "student"); save_adapter(model, tok, "teacher", st / "teacher")
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict()}, st / "optim.pt")
    (st / "meta.json").write_text(json.dumps(meta, indent=1))
    if (out / "state").exists():
        shutil.rmtree(out / "state.old", ignore_errors=True); (out / "state").rename(out / "state.old")
    st.rename(out / "state"); shutil.rmtree(out / "state.old", ignore_errors=True)


def find_state(out):
    for d in (out / "state", out / "state.old"):
        if (d / "meta.json").exists() and (d / "optim.pt").exists(): return d
    return None


def truncate_jsonl(path, max_step):
    if not path.exists(): return
    keep = []
    for l in open(path):
        try:
            if json.loads(l).get("step", -1) <= max_step: keep.append(l)
        except Exception:
            pass
    path.write_text("".join(keep))


def jdump(f, obj):
    f.write(json.dumps(obj) + "\n"); f.flush()


# =============================================================== main loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="reports/alfworld/student3b_low100/final", help="start = our SFT student LoRA (kept training)")
    ap.add_argument("--n_tasks", type=int, default=1081, help="ALFWorld train games 0..n_tasks-1 of game_files('train') (never 1500+)")
    ap.add_argument("--B", type=int, default=2, help="tasks per optimizer step (official: batch_size 1 x grad-accum 2, arguments.py:12-13)")
    ap.add_argument("--rollout_n", type=int, default=4); ap.add_argument("--budget", type=int, default=50); ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-6); ap.add_argument("--warmup_frac", type=float, default=0.05); ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=0.0); ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--topk", type=int, default=100); ap.add_argument("--no_tail", action="store_true")
    ap.add_argument("--fb_max_steps", type=int, default=3); ap.add_argument("--fb_temperature", type=float, default=0.6); ap.add_argument("--fb_top_p", type=float, default=0.95)
    ap.add_argument("--fb_max_tokens", type=int, default=512); ap.add_argument("--fb_max_chars", type=int, default=12000)
    ap.add_argument("--fb_place", choices=["turn", "inline"], default="turn", help="turn = official extra user turn (trainer.py:628); inline = note before the admissible list")
    ap.add_argument("--ckpt_at", default="1323,2823", help="cumulative episodes at which ckpt1, ckpt2 are saved; ckpt3 = end of the epoch")
    ap.add_argument("--save_every", type=int, default=10); ap.add_argument("--max_steps", type=int, default=0, help="stop after this many optimizer steps (smoke test)")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--score_max_tokens", type=int, default=16384); ap.add_argument("--gen_bs", type=int, default=16)
    ap.add_argument("--env_recycle", type=int, default=25); ap.add_argument("--env_timeout", type=int, default=180)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    from peft import PeftModel
    from error_attr.env.alfworld_env import game_files
    from error_attr.policy.alfworld_agent import LLMPolicy, SYSTEM_PROMPT
    from alfworld_attr_opd import history_at

    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    adapter = a.adapter if Path(a.adapter).is_absolute() else str(ROOT / a.adapter)
    tasks = game_files("train")[: a.n_tasks]; assert a.n_tasks <= 1500, "train games 1500-1999 are the held-out test set"
    random.Random(a.seed).shuffle(tasks)
    total_steps = math.ceil(len(tasks) / a.B); ckpt_at = [int(x) for x in a.ckpt_at.split(",") if x]
    print(f"HINT-SD online: {len(tasks)} tasks x {a.rollout_n} rollouts = {len(tasks) * a.rollout_n} episodes, B={a.B} tasks/step -> {total_steps} optimizer steps; "
          f"ckpts at {ckpt_at} episodes + end; out {out}", flush=True)

    # ---- model: base bf16 + two LoRA adapters (student trainable, teacher frozen EMA copy), as train.py:190-205 / trainer.py:99-103
    tok = AutoTokenizer.from_pretrained(a.model); tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    base.gradient_checkpointing_enable(); base.enable_input_require_grads(); base.config.use_cache = False   # train.py:199-203
    model = PeftModel.from_pretrained(base, adapter, adapter_name="student", is_trainable=True)
    model.load_adapter(adapter, adapter_name="teacher", is_trainable=False)   # teacher init = start adapter (trainer.py:100)
    model.set_adapter("student")
    pairs = student_teacher_pairs(model); params = [s for s, _ in pairs]
    for s, t in pairs: s.requires_grad_(True); t.requires_grad_(False)
    print(f"student params {sum(p.numel() for p in params) / 1e6:.1f}M ({params[0].dtype}), teacher {pairs[0][1].dtype}", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=a.weight_decay)   # HF Trainer AdamW defaults
    sched = get_linear_schedule_with_warmup(opt, math.ceil(a.warmup_frac * total_steps), total_steps)
    pol = LLMPolicy(model, tok)

    meta = {"step": 0, "cum_episodes": 0, "ckpts": [], "sec_total": 0.0, "gen_tokens": 0, "fwd_tokens": 0, "targets": 0, "target_tokens": 0,
            "episodes_won": 0, "failed": 0, "analyzed": 0, "env_timeouts": 0, "nonfinite_rows": 0, "empty_steps": 0, "args": vars(a)}
    st = find_state(out)
    if st is not None:
        meta = json.loads((st / "meta.json").read_text())
        load_adapter_into(model, "student", st / "student"); load_adapter_into(model, "teacher", st / "teacher")
        sd = torch.load(st / "optim.pt", map_location="cpu", weights_only=False); opt.load_state_dict(sd["opt"]); sched.load_state_dict(sd["sched"])
        model.set_adapter("student")
        for s, t in pairs: s.requires_grad_(True); t.requires_grad_(False)
        for f in ("train_log.jsonl", "rollouts.jsonl", "analysis.jsonl", "targets.jsonl"): truncate_jsonl(out / f, meta["step"])
        cs = checksum(pairs)
        print(f"RESUMED from {st} at step {meta['step']} ({meta['cum_episodes']} episodes): |student| {cs[0]} |teacher| {cs[1]} max|student-teacher| {cs[2]:.3e}", flush=True)
    if meta["step"] >= total_steps and (out / "ckpt3" / "adapter" / "adapter_config.json").exists():
        print("DONE hintsd_online (already complete)", flush=True); return

    flog, frol, fana, ftgt = (open(out / f, "a") for f in ("train_log.jsonl", "rollouts.jsonl", "analysis.jsonl", "targets.jsonl"))
    pool = EnvPool(a.B * a.rollout_n, a.budget, a.env_recycle, a.env_timeout)
    first_user = lambda e: history_at(e, 0).render_prompt()
    steps_this_run = 0
    try:
        while meta["step"] < total_steps:
            step = meta["step"] + 1; games = tasks[(step - 1) * a.B: step * a.B]
            random.seed(a.seed * 1000003 + step); torch.manual_seed(a.seed * 1000003 + step)
            gen = torch.Generator().manual_seed(a.seed * 1000003 + step)
            t0 = time.time()
            # ---------------- 1. rollouts with the current student (eval mode: no dropout, KV-free tree scoring)
            model.eval(); model.set_adapter("student")
            with torch.no_grad():
                eps, rst = rollout(pool, model, tok, games, a.rollout_n, a.budget, a.temperature, gen, step, a.score_max_tokens)
            t1 = time.time()
            for e in eps: jdump(frol, {"step": step, **e})
            # ---------------- 2. hindsight analysis of the failed rollouts by the EMA teacher (trainer.py:514, 588-605)
            failed = [i for i, e in enumerate(eps) if not e["won"] and e["steps"] and "error" not in e]
            prompts = [tok.apply_chat_template([{"role": "system", "content": ANALYZER_SYSTEM.format(k=a.fb_max_steps)},
                                                {"role": "user", "content": ANALYZER_USER.format(traj=render_for_detection(eps[i], SYSTEM_PROMPT, first_user(eps[i]), a.fb_max_chars))}],
                                               tokenize=False, add_generation_prompt=True) for i in failed]
            model.set_adapter("teacher")
            with torch.no_grad():
                raws, n_gen = generate_batch(model, tok, prompts, a.fb_max_tokens, a.fb_temperature, a.fb_top_p, a.gen_bs) if prompts else ([], 0)
            model.set_adapter("student")
            for s, t in pairs: s.requires_grad_(True); t.requires_grad_(False)
            detections = {}
            for i, raw in zip(failed, raws):
                det = parse_multi_step_failure_json(raw, len(eps[i]["steps"]), a.fb_max_steps); detections[i] = det
                jdump(fana, {"step": step, "id": eps[i]["id"], "n_steps": len(eps[i]["steps"]), "parsed": det is not None,
                             "targets": [{"t": t, "feedback": fb} for t, fb in (det or [])], "raw": raw})
            t2 = time.time()
            # ---------------- 3. targets: (student prompt at t, sampled action tokens, teacher prompt = student prompt + feedback turn)
            rows_by_task = {ti: [] for ti in range(len(games))}
            for i, det in detections.items():
                for t, fb in det or []:
                    h = history_at(eps[i], t); sp = pol._build_chat_prompt(h); action = eps[i]["steps"][t]["action"]
                    if a.fb_place == "turn":
                        tp = tok.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": h.render_prompt()},
                                                      {"role": "user", "content": FB_TEMPLATE.format(hint=fb)}], tokenize=False, add_generation_prompt=True)
                    else:
                        tp = tok.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT},
                                                      {"role": "user", "content": h.render_prompt().replace("Admissible actions:", FB_TEMPLATE.format(hint=fb) + "\nAdmissible actions:", 1)}],
                                                     tokenize=False, add_generation_prompt=True)
                    comp = tok(action, add_special_tokens=False).input_ids
                    if comp: rows_by_task[eps[i]["task"]].append({"id": eps[i]["id"], "t": t, "feedback": fb, "action": action, "sp": sp, "tp": tp, "comp": comp})
            # ---------------- 4. loss: per task (= one HF micro-batch of 1 task x 4 rollouts) token-mean reverse KL, / #tasks (grad accumulation)
            model.train(); n_rows = n_tok = 0; loss_sum = 0.0; kl_full_w = 0.0; tgt_logs = []
            opt.zero_grad(set_to_none=True)
            n_acc = len(games)
            for ti, rows in rows_by_task.items():
                valid = sum(len(r["comp"]) for r in rows)
                for r in rows:   # distillation_microbatch_size = 1 (arguments.py:69): one target row per forward / backward
                    sp_ids = tok(r["sp"], add_special_tokens=False).input_ids; tp_ids = tok(r["tp"], add_special_tokens=False).input_ids
                    model.set_adapter("teacher")
                    with torch.no_grad(): t_lp = completion_log_probs(model, tp_ids, r["comp"])
                    model.set_adapter("student")
                    for s, t in pairs: s.requires_grad_(True); t.requires_grad_(False)
                    s_lp = completion_log_probs(model, sp_ids, r["comp"])
                    mask = torch.ones(s_lp.shape[:2], device=s_lp.device)
                    row_loss = compute_distillation_loss(s_lp, t_lp, mask, a.topk, not a.no_tail)
                    w = len(r["comp"]) / valid                      # trainer.py:796-799 (microbatch tokens / valid tokens)
                    if not torch.isfinite(row_loss):
                        meta["nonfinite_rows"] += 1; print(f"SKIP non-finite row {r['id']} t={r['t']}", flush=True); continue
                    (row_loss * w / n_acc).backward()               # trainer.py:1771-1772 (loss / gradient_accumulation_steps)
                    with torch.no_grad():
                        kl_full = (s_lp.exp() * (s_lp - t_lp)).sum(-1).mean().item()
                        t_top1 = t_lp[0, 0].argmax().item(); first_ok = int(t_top1 == r["comp"][0])
                    loss_sum += row_loss.item() * len(r["comp"]); kl_full_w += kl_full * len(r["comp"]); n_rows += 1; n_tok += len(r["comp"])
                    tgt_logs.append({"step": step, "id": r["id"], "t": r["t"], "feedback": r["feedback"], "action": r["action"], "n_tok": len(r["comp"]),
                                     "loss": round(row_loss.item(), 5), "kl_full": round(kl_full, 5), "teacher_top1_is_action_tok0": first_ok,
                                     "teacher_lp_action": round(t_lp[0].gather(1, torch.tensor(r["comp"], device=t_lp.device)[:, None]).sum().item(), 4),
                                     "student_lp_action": round(s_lp[0].gather(1, torch.tensor(r["comp"], device=s_lp.device)[:, None]).sum().item(), 4)})
                    del s_lp, t_lp, row_loss
            gnorm = float("nan"); lr_used = sched.get_last_lr()[0]
            if n_rows:
                gnorm = float(torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm)); opt.step()
            else:
                meta["empty_steps"] += 1   # no target in the whole step: HF still counts the step (scheduler + EMA) with zero gradients
            sched.step(); opt.zero_grad(set_to_none=True)
            with torch.no_grad():                                   # TeacherEMACallback.on_step_end -> lerp_(student, 1 - decay)
                for s, t in pairs: t.lerp_(s.detach().to(t.dtype), 1.0 - a.ema_decay)
            t3 = time.time()
            for x in tgt_logs: jdump(ftgt, x)
            # ---------------- 5. bookkeeping, checkpoints, state
            n_won = sum(e["won"] for e in eps); n_err = sum("error" in e for e in eps); n_parsed = sum(d is not None for d in detections.values())
            meta.update(step=step, cum_episodes=meta["cum_episodes"] + len(eps), sec_total=meta["sec_total"] + (t3 - t0))
            for k, v in (("gen_tokens", n_gen), ("fwd_tokens", rst["fwd_tokens"]), ("targets", n_rows), ("target_tokens", n_tok), ("episodes_won", n_won),
                         ("failed", len(failed)), ("analyzed", n_parsed), ("env_timeouts", n_err)):
                meta[k] = meta.get(k, 0) + v
            rec = {"step": step, "of": total_steps, "cum_episodes": meta["cum_episodes"], "episodes": len(eps), "success_rate": round(n_won / len(eps), 4),
                   "n_failed": len(failed), "n_analyzed": n_parsed, "n_targets": n_rows, "n_target_tokens": n_tok,
                   "loss": round(loss_sum / n_tok, 6) if n_tok else None, "kl_full": round(kl_full_w / n_tok, 6) if n_tok else None,
                   "lr": lr_used, "grad_norm": round(gnorm, 5) if n_rows else None, "env_errors": n_err, "gen_tokens": n_gen,
                   "sec": round(t3 - t0, 1), "sec_rollout": round(t1 - t0, 1), "sec_rollout_env": round(rst["sec_env"], 1), "sec_analyze": round(t2 - t1, 1),
                   "sec_loss": round(t3 - t2, 1), "env_restarts": pool.restarts,
                   "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            jdump(flog, rec); print(json.dumps(rec), flush=True)
            saved_ckpt = False
            for c, thr in enumerate(ckpt_at, 1):
                if meta["cum_episodes"] >= thr and c not in meta["ckpts"]:
                    save_adapter(model, tok, "student", out / f"ckpt{c}" / "adapter"); meta["ckpts"].append(c); saved_ckpt = True
                    print(f"CKPT{c} saved at step {step}, {meta['cum_episodes']} episodes", flush=True)
            if step == total_steps:
                c = len(ckpt_at) + 1; save_adapter(model, tok, "student", out / f"ckpt{c}" / "adapter"); meta["ckpts"].append(c); saved_ckpt = True
                print(f"CKPT{c} (end) saved at step {step}, {meta['cum_episodes']} episodes", flush=True)
            steps_this_run += 1
            stop = a.max_steps and steps_this_run >= a.max_steps
            if step % a.save_every == 0 or step == total_steps or stop or saved_ckpt:
                save_state(out, model, tok, opt, sched, meta)
                cs = checksum(pairs); print(f"STATE saved at step {step}: |student| {cs[0]} |teacher| {cs[1]} max|student-teacher| {cs[2]:.3e}", flush=True)
            if stop:
                print(f"STOP after {steps_this_run} steps (--max_steps)", flush=True); break
    finally:
        pool.close()
    if meta["step"] >= total_steps:
        summ = {"gpu_hours": round(meta["sec_total"] / 3600, 2), "episodes": meta["cum_episodes"], "optimizer_steps": meta["step"],
                "rollout_success_rate": round(meta["episodes_won"] / max(1, meta["cum_episodes"]), 4), "failed": meta["failed"], "analyzed": meta["analyzed"],
                "analyzer_generated_tokens": meta["gen_tokens"], "policy_scoring_forward_tokens": meta["fwd_tokens"],
                "backprop_target_steps": meta["targets"], "backprop_target_tokens": meta["target_tokens"], "nonfinite_rows": meta["nonfinite_rows"],
                "empty_steps": meta["empty_steps"], "env_errors": meta["env_timeouts"]}
        (out / "summary.json").write_text(json.dumps(summ, indent=1))
        print("SUMMARY " + json.dumps(summ), flush=True); print("DONE hintsd_online", flush=True)


if __name__ == "__main__":
    main()
