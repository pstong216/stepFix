"""ETO-style baseline: trajectory-level DPO on (success, failure) trajectory pairs of the SAME game (the failure from the
original collection, the success from any collection), continued from the SFT student.
  prep  -- reference log-probs (sum over steps of log pi_ref(a_t | h_t)) for both trajectories of every pair, under --adapter
  train -- DPO:  L = -log sigmoid( beta * [ (pi_c - ref_c) - (pi_r - ref_r) ] )  with pi_x = sum_t log pi(a_t | h_t) over the first
           --max_steps steps.  Exact gradient without keeping 50 graphs: m is computed without grad, then every step's
           log-prob is back-propagated with weight beta*sigmoid(-beta*m) (sign + for chosen, - for rejected).
"""
import os, argparse, glob, json, math, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch
from alfworld_attr_opd import history_at, load_jsonl
from alfworld_opd_plus import load_policy


def action_logprob(pol, tok, h, action):
    prompt = pol._build_chat_prompt(h)
    p_ids = tok(prompt, add_special_tokens=False).input_ids; t_ids = tok(action, add_special_tokens=False).input_ids + [tok.eos_token_id]
    ids = torch.tensor([p_ids + t_ids], device=pol.device)
    logits = pol.model(input_ids=ids).logits[0, len(p_ids) - 1: -1]
    lp = torch.log_softmax(logits.float(), -1).gather(1, torch.tensor(t_ids, device=pol.device)[:, None]).sum()
    return lp


def traj_steps(e, max_steps):
    return [(history_at(e, t), e["steps"][t]["action"]) for t in range(min(max_steps, len(e["steps"])))]


def load_all(src):
    eps = {e["id"]: e for e in load_jsonl(ROOT / src)}
    extra = os.environ.get("ETO_WIN_SRC")   # fair ETO (Oct 5 22:20): winners ONLY from this file (expert demos), never later-round chains
    if extra:
        for e in load_jsonl(ROOT / extra): eps.setdefault(e["id"], e)
        return eps
    for p in sorted(glob.glob(str(ROOT / "reports/alfworld/self_evolve3b_rounds/*/r*/traj.jsonl"))):
        for e in load_jsonl(p): eps.setdefault(e["id"], e)
    return eps


def cmd_prep(a):
    pairs = [r for r in load_jsonl(ROOT / a.pairs) if r["src"] == a.src]; eps = load_all(a.src)
    tok, model, pol = load_policy(a.model, str(ROOT / a.adapter)); model.eval(); out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set(); t0 = time.time()
    with open(out, "a") as f, torch.no_grad():
        for k, r in enumerate(pairs):
            if r["id"] in done or r["win_id"] not in eps: continue
            rc = sum(action_logprob(pol, tok, h, act).item() for h, act in traj_steps(eps[r["win_id"]], a.max_steps))
            rr = sum(action_logprob(pol, tok, h, act).item() for h, act in traj_steps(eps[r["id"]], a.max_steps))
            f.write(json.dumps({"id": r["id"], "win_id": r["win_id"], "ref_c": rc, "ref_r": rr}) + "\n"); f.flush()
            if k % 20 == 0: print(f"[{k + 1}/{len(pairs)}] {time.time() - t0:.0f}s", flush=True)
    print("DONE prep", flush=True)


def cmd_train(a):
    refs = load_jsonl(ROOT / a.refs); eps = load_all(a.src); rng = random.Random(a.seed); torch.manual_seed(a.seed); rng.shuffle(refs)
    tok, model, pol = load_policy(a.model, str(ROOT / a.adapter), trainable=True); model.train()
    params = [p for p in model.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0); t0 = time.time(); acc = []
    refs = [r for ep in range(a.epochs) for r in (rng.sample(refs, len(refs)) if ep else refs)]   # --epochs (ETO: 3)
    for k, r in enumerate(refs):
        if r["win_id"] not in eps: continue
        C = traj_steps(eps[r["win_id"]], a.max_steps); R = traj_steps(eps[r["id"]], a.max_steps)
        with torch.no_grad():
            pc = sum(action_logprob(pol, tok, h, act).item() for h, act in C); pr = sum(action_logprob(pol, tok, h, act).item() for h, act in R)
        m = (pc - r["ref_c"]) - (pr - r["ref_r"]); w = a.beta * (1.0 / (1.0 + math.exp(a.beta * m))); acc.append(-math.log(1.0 / (1.0 + math.exp(-a.beta * m)) + 1e-12))
        for h, act in C: (-w * action_logprob(pol, tok, h, act) / a.batch).backward()
        for h, act in R: (w * action_logprob(pol, tok, h, act) / a.batch).backward()
        if (k + 1) % a.batch == 0: torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
        if (k + 1) % 20 == 0: print(f"[{k + 1}/{len(refs)}] dpo loss {sum(acc[-20:]) / 20:.3f} margin {m:.2f} {time.time() - t0:.0f}s", flush=True)
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True); model.save_pretrained(str(out / "adapter")); tok.save_pretrained(str(out / "adapter"))
    print(f"DONE train {len(refs)} pairs", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("prep", "train"):
        s = sub.add_parser(n); s.add_argument("--src", default="reports/alfworld/student3b_low100_train_b50.jsonl"); s.add_argument("--adapter", default="reports/alfworld/student3b_low100/final")
        s.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); s.add_argument("--max_steps", type=int, default=25); s.add_argument("--out", required=True)
        if n == "prep": s.add_argument("--pairs", required=True)
        else: s.add_argument("--refs", required=True); s.add_argument("--epochs", type=int, default=1); s.add_argument("--beta", type=float, default=0.1); s.add_argument("--lr", type=float, default=1e-5); s.add_argument("--batch", type=int, default=4); s.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); {"prep": cmd_prep, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
