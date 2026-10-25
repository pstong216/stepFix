"""Offline variants of the round-1 recipe that also use SUCCESS data. All arms start from the same student adapter.
  prep  -- build example files from existing round-1 artefacts (needs the start policy on GPU):
           kl      verified KL targets of an arm (reconstructed exactly from examples.jsonl + targets.jsonl)
           anchor  start-policy action distributions on states of the student's own WON trajectories (KL anchor)
           dpo     clean contrastive pairs (alternative wins, original action loses; counterfactual.jsonl) + ref scores
           cont    greedy continuations after each verified alternative, recorded; won ones become SFT states
           rft     states of the student's own won trajectories (rejection-sampling fine-tuning)
  train -- one pass over a weighted mix of example files; losses: kl / anchor = KL(target || student) over the
           admissible commands; sft = next-token CE on the action (same format as alfworld_sft.py); dpo = DPO on the
           summed command log-probs (the admissible-set normaliser cancels in the chosen-minus-rejected difference).
"""
import argparse, json, math, random, signal, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from error_attr.policy.alfworld_agent import LLMPolicy, InteractionStep
from alfworld_attr_opd import history_at, student_logprobs_grad, load_jsonl


def load_policy(student, adapter, trainable=False):
    tok = AutoTokenizer.from_pretrained(student); tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(student, dtype=torch.bfloat16, device_map={"": 0})
    if trainable:
        base.gradient_checkpointing_enable(); base.enable_input_require_grads()
        if adapter:
            model = PeftModel.from_pretrained(base, adapter, is_trainable=True)
        else:   # fresh LoRA on an untrained base model (self-evolution from zero-shot); same config as the student LoRA
            from peft import LoraConfig, get_peft_model
            model = get_peft_model(base, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
                                                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        model.train()
    else:
        model = (PeftModel.from_pretrained(base, adapter) if adapter else base).eval()
    return tok, model, LLMPolicy(model, tok)


def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; z = sum(e); return [x / z for x in e]


def record_continuation(ep, t, alt, pol, budget, max_steps):
    """Replay actions[:t], take alt, continue greedily; return (won, synthetic episode with every step logged)."""
    from error_attr.env.alfworld_env import make_env
    def _alarm(s, f): raise TimeoutError()
    signal.signal(signal.SIGALRM, _alarm); signal.alarm(300)
    env = make_env(ep["game_file"], budget + 1); steps = [dict(s) for s in ep["steps"][:t]]
    try:
        obs, info = env.reset()
        for s in ep["steps"][:t]:
            obs, rw, dn, info = env.step([s["action"]])
            if dn[0]: return False, None
        cur = {"obs": ep["steps"][t]["obs"], "admissible": list(ep["steps"][t]["admissible"]), "action": alt}
        obs, rw, dn, info = env.step([alt]); cur["next_obs"] = obs[0]; steps.append(cur)
        if dn[0]: return bool(info["won"][0]), {**ep, "steps": steps}
        h = history_at(ep, t); h.steps[-1].action = alt
        h.steps.append(InteractionStep(observation=obs[0], admissible_commands=list(info["admissible_commands"][0])))
        k = t + 1; limit = min(budget, t + 1 + max_steps)
        while k < limit:
            act = pol.act(h, temperature=0.0); h.steps[-1].action = act
            st = {"obs": h.steps[-1].observation, "admissible": list(h.steps[-1].admissible_commands), "action": act}
            obs, rw, dn, info = env.step([act]); st["next_obs"] = obs[0]; steps.append(st); k += 1
            if dn[0]: return bool(info["won"][0]), {**ep, "steps": steps}
            h.steps.append(InteractionStep(observation=obs[0], admissible_commands=list(info["admissible_commands"][0])))
        return False, None
    except TimeoutError:
        return False, None
    finally:
        signal.alarm(0)
        try: env.close()
        except Exception: pass


@torch.no_grad()
def cmd_prep(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rng = random.Random(0)
    arm = ROOT / "reports/alfworld/opd/student3b_low100" / a.arm
    traj = str(ROOT / a.traj); eps = {e["id"]: e for e in load_jsonl(traj)}
    tg = {(r["id"], r["t"]): r for r in load_jsonl(arm / "targets.jsonl")}
    ex = [r for r in load_jsonl(arm / "round1/examples.jsonl") if r.get("kind") == "verified"]
    # kl: exact reconstruction of the verified targets used by cmd_distill (weights over winning alternatives, a_minus zeroed)
    with open(out / "kl.jsonl", "w") as f:
        for r in ex:
            t = tg[(r["id"], r["t"])]; cmds = t["admissible"]; good = [c for c, w in (r.get("alternatives") or {}).items() if w and c != t["a_minus"]]
            if not good: continue
            w = [math.exp(t["teacher_logprobs"][c]) for c in good]; z = sum(w)
            target = [0.0] * len(cmds)
            for c, wi in zip(good, w): target[cmds.index(c)] = wi / z
            f.write(json.dumps({"kind": "kl", "src": a.traj, "id": r["id"], "t": r["t"], "cmds": cmds, "target": target}) + "\n")
    print("kl", sum(1 for _ in open(out / "kl.jsonl")), flush=True)
    won = [e for e in eps.values() if e["won"]]
    states = [(e["id"], t) for e in won for t in range(len(e["steps"]))]; rng.shuffle(states)
    with open(out / "rft.jsonl", "w") as f:
        for i, t in states[: a.n_rft]:
            f.write(json.dumps({"kind": "sft", "src": a.traj, "id": i, "t": t, "action": eps[i]["steps"][t]["action"]}) + "\n")
    print(f"rft {min(a.n_rft, len(states))} states from {len(won)} won games ({len(states)} states)", flush=True)
    tok, model, pol = load_policy(a.student, a.adapter)
    anchor_states = states[a.n_rft: a.n_rft + a.n_anchor]
    with open(out / "anchor.jsonl", "w") as f:
        for k, (i, t) in enumerate(anchor_states):
            lp = pol.score_admissible_commands_cached(history_at(eps[i], t)); cmds = list(lp)
            f.write(json.dumps({"kind": "anchor", "src": a.traj, "id": i, "t": t, "cmds": cmds, "target": softmax([lp[c] for c in cmds])}) + "\n")
    print("anchor", len(anchor_states), flush=True)
    cf = [r for r in load_jsonl(arm / "round1/counterfactual.jsonl") if r["alt_won"] and not r["a_minus_won"]]
    with open(out / "dpo.jsonl", "w") as f:
        for r in cf:
            lp = pol.score_admissible_commands_cached(history_at(eps[r["id"]], r["t"]))
            f.write(json.dumps({"kind": "dpo", "src": a.traj, "id": r["id"], "t": r["t"], "chosen": r["alt"], "rejected": r["a_minus"],
                                "ref_c": lp[r["alt"]], "ref_r": lp[r["a_minus"]]}) + "\n")
    print("dpo", len(cf), flush=True)
    n_won = 0
    with open(out / "cont_episodes.jsonl", "w") as fe, open(out / "cont.jsonl", "w") as fs:
        for k, r in enumerate(ex):
            t = tg[(r["id"], r["t"])]; good = [c for c, w in (r.get("alternatives") or {}).items() if w and c != t["a_minus"]]
            if not good: continue
            alt = max(good, key=lambda c: t["teacher_logprobs"][c]); e = eps[r["id"]]
            w, syn = record_continuation(e, r["t"], alt, pol, e["budget"], a.max_continue_steps)
            if not w: continue
            n_won += 1; syn = {**syn, "id": f"{r['id']}__cont_t{r['t']}"}; fe.write(json.dumps(syn) + "\n")
            for s in range(r["t"] + 1, len(syn["steps"])):
                fs.write(json.dumps({"kind": "sft", "src": str((out / "cont_episodes.jsonl").relative_to(ROOT)), "id": syn["id"], "t": s, "action": syn["steps"][s]["action"]}) + "\n")
            if k % 20 == 0: print(f"cont [{k + 1}/{len(ex)}] won {n_won}", flush=True)
    print("cont episodes won", n_won, "sft states", sum(1 for _ in open(out / "cont.jsonl")), flush=True)
    print("DONE prep", flush=True)


def sft_loss(pol, tok, h, action):
    prompt = pol._build_chat_prompt(h)
    p_ids = tok(prompt, add_special_tokens=False).input_ids; t_ids = tok(action, add_special_tokens=False).input_ids + [tok.eos_token_id]
    ids = torch.tensor([p_ids + t_ids], device=pol.device); labels = torch.tensor([[-100] * len(p_ids) + t_ids], device=pol.device)
    return pol.model(input_ids=ids, labels=labels).loss


def cmd_train(a):
    rng = random.Random(a.seed); items = []
    for spec in a.data:
        path, _, w = spec.partition(":"); w = float(w or 1.0)
        rows = load_jsonl(ROOT / path); items += [(r, w) for r in rows]
        print(f"{path}: {len(rows)} examples, weight {w}", flush=True)
    items = items * int(a.epochs); rng.shuffle(items)
    episodes = {}
    def ep_of(r):
        if r["src"] not in episodes: episodes[r["src"]] = {e["id"]: e for e in load_jsonl(ROOT / r["src"])}
        return episodes[r["src"]][r["id"]]
    tok, model, pol = load_policy(a.student, a.adapter, trainable=True)
    params = [p for p in model.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    out = Path(ROOT / a.out); out.mkdir(parents=True, exist_ok=True); log = open(out / "train_log.jsonl", "w")
    acc = {}; n = 0; t0 = time.time()
    for k, (r, w) in enumerate(items):
        h = history_at(ep_of(r), r["t"])
        if r["kind"] in ("kl", "anchor"):
            lp = student_logprobs_grad(pol, tok, h, r["cmds"]); lq = torch.log_softmax(lp, 0)
            tgt = torch.tensor(r["target"], device=lq.device)
            loss = (tgt * (torch.log(tgt.clamp_min(1e-8)) - lq)).sum()
        elif r["kind"] == "sft":
            loss = sft_loss(pol, tok, h, r["action"])
        elif r["kind"] == "dpo":
            lp = student_logprobs_grad(pol, tok, h, [r["chosen"], r["rejected"]])
            margin = (lp[0] - r["ref_c"]) - (lp[1] - r["ref_r"])
            loss = -torch.nn.functional.logsigmoid(a.beta * margin)
        else:
            raise ValueError(r["kind"])
        if not torch.isfinite(loss):   # NaN guard (Oct 5 23:15: one non-finite example poisoned two runs) -- skip the example, log it
            print(f"SKIP non-finite loss at item {k} ({r.get('id', '?')}, t={r.get('t', '?')})", flush=True); continue
        (w * loss / a.batch).backward(); acc.setdefault(r["kind"], []).append(loss.item()); n += 1
        if n == a.batch or k == len(items) - 1:
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(gn): opt.zero_grad(); n = 0; print(f"SKIP non-finite grad at item {k}", flush=True); continue
            opt.step(); opt.zero_grad(); n = 0
            rec = {"k": k + 1, "of": len(items), **{f"loss_{kk}": sum(v) / len(v) for kk, v in acc.items()}, "elapsed_s": round(time.time() - t0)}
            log.write(json.dumps(rec) + "\n"); log.flush(); acc = {}
            if (k + 1) % (a.batch * 10) == 0: print(json.dumps(rec), flush=True)
    model.save_pretrained(str(out / "adapter")); tok.save_pretrained(str(out / "adapter"))
    print("DONE train", len(items), "examples", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep"); p.add_argument("--arm", default="attr_nofb"); p.add_argument("--traj", default="reports/alfworld/student3b_low100_train_b50.jsonl")
    p.add_argument("--student", default="Qwen/Qwen2.5-3B-Instruct"); p.add_argument("--adapter", default=str(ROOT / "reports/alfworld/student3b_low100/final"))
    p.add_argument("--n_rft", type=int, default=1000); p.add_argument("--n_anchor", type=int, default=500); p.add_argument("--max_continue_steps", type=int, default=20)
    p.add_argument("--out", default=str(ROOT / "data/opd_plus/attr_nofb_r1"))
    t = sub.add_parser("train"); t.add_argument("--data", nargs="+", required=True, help="path[:weight] ...")
    t.add_argument("--student", default="Qwen/Qwen2.5-3B-Instruct"); t.add_argument("--adapter", default=str(ROOT / "reports/alfworld/student3b_low100/final"))
    t.add_argument("--out", required=True); t.add_argument("--lr", type=float, default=1e-5); t.add_argument("--batch", type=int, default=8)
    t.add_argument("--epochs", type=int, default=1); t.add_argument("--beta", type=float, default=0.1); t.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); cmd_prep(a) if a.cmd == "prep" else cmd_train(a)


if __name__ == "__main__":
    main()
