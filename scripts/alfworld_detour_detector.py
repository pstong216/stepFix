"""Self-trained step attributor from replay-verified detour labels (no planner, no expert, no external model).
Labels come from scripts/compress_trajectories.py: in a WON trajectory a step is 'necessary' when it survives detour pruning
(the pruned action list still wins on exact env replay) and a 'detour' otherwise.
  train   -- LoRA from the student adapter (same recipe as the M5 progress estimator): policy prompt at step t + the action
             taken + "Necessary: " -> "yes" / "no".
  score   -- P(detour) for every step of the given failed trajectories.
  pointer -- first step with P(detour) > --thr (fallback: argmax), written in the steps.jsonl pointer schema.
"""
import argparse, json, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch
from alfworld_attr_opd import history_at, load_jsonl
from alfworld_progress_model import load

PREFIX = "Necessary: "


def prompt_at(pol, e, t):
    return pol._build_chat_prompt(history_at(e, t)) + f"Action: {e['steps'][t]['action']}\n" + PREFIX


def cmd_train(a):
    rng = random.Random(0); raw = {e["id"]: e for e in load_jsonl(ROOT / a.traj) if e["won"]}
    items = []
    for r in load_jsonl(ROOT / a.compressed):
        if r["variant"] != "pruned" or not r["won"] or r.get("timeout") or r["src_id"] not in raw: continue
        e = raw[r["src_id"]]
        items += [(e, t, bool(k)) for t, k in enumerate(r["kept"]) if t < len(e["steps"])]
    rng.shuffle(items); items = items[: a.n]
    print(f"{len(items)} states, {sum(not k for _, _, k in items)} detours ({sum(not k for _, _, k in items) / len(items):.1%})", flush=True)
    tok, model, pol = load(a.student, a.adapter, True)
    params = [p for p in model.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True); t0 = time.time(); run = 0.0
    for k, (e, t, kept) in enumerate(items):
        p_ids = tok(prompt_at(pol, e, t), add_special_tokens=False).input_ids; t_ids = tok(("yes" if kept else "no") + "\n", add_special_tokens=False).input_ids
        ids = torch.tensor([p_ids + t_ids], device=model.device); lab = torch.tensor([[-100] * len(p_ids) + t_ids], device=model.device)
        loss = model(input_ids=ids, labels=lab).loss; (loss / a.batch).backward(); run += loss.item()
        if (k + 1) % a.batch == 0 or k == len(items) - 1:
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
        if (k + 1) % 200 == 0: print(f"[{k + 1}/{len(items)}] loss {run / 200:.3f} {time.time() - t0:.0f}s", flush=True); run = 0.0
    model.save_pretrained(str(out / "adapter")); tok.save_pretrained(str(out / "adapter")); print("DONE train", flush=True)


@torch.no_grad()
def p_detour(model, tok, pol, e, t, yes_id, no_id):
    ids = tok(prompt_at(pol, e, t), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    lg = model(input_ids=ids).logits[0, -1].float(); two = torch.log_softmax(lg[[yes_id, no_id]], -1)
    return two[1].exp().item()


def cmd_score(a):
    eps = {e["id"]: e for e in load_jsonl(ROOT / a.traj)}
    ids = [r["id"] for r in load_jsonl(ROOT / a.ids_from)] if a.ids_from else [i for i, e in eps.items() if not e["won"]]
    done = {json.loads(l)["id"] for l in open(ROOT / a.out)} if (ROOT / a.out).exists() else set()
    ids = [i for i in ids if i not in done]; print(f"{len(ids)} trajectories to score", flush=True)
    tok, model, pol = load(a.student, a.adapter, False)
    yes_id = tok("yes\n", add_special_tokens=False).input_ids[0]; no_id = tok("no\n", add_special_tokens=False).input_ids[0]
    with open(ROOT / a.out, "a") as f:
        for j, i in enumerate(ids):
            e = eps[i]; ps = [p_detour(model, tok, pol, e, t, yes_id, no_id) for t in range(len(e["steps"]))]
            f.write(json.dumps({"id": i, "p_detour": ps}) + "\n"); f.flush()
            if j % 20 == 0: print(f"[{j + 1}/{len(ids)}]", flush=True)
    print("DONE score", flush=True)


def cmd_score_cands(a):
    """P(necessary) of arbitrary candidate actions at failed states: rows {id, t, cand} -> + p_necessary (proxy for rollout)."""
    eps = {e["id"]: e for e in load_jsonl(ROOT / a.traj)}; rows = load_jsonl(ROOT / a.cands)
    done = {(r["id"], r["t"], r["cand"]) for r in load_jsonl(ROOT / a.out)} if (ROOT / a.out).exists() else set()
    rows = [r for r in rows if (r["id"], r["t"], r["cand"]) not in done]; print(f"{len(rows)} candidates to score", flush=True)
    tok, model, pol = load(a.student, a.adapter, False)
    yes_id = tok("yes\n", add_special_tokens=False).input_ids[0]; no_id = tok("no\n", add_special_tokens=False).input_ids[0]
    with open(ROOT / a.out, "a") as f, torch.no_grad():
        for j, r in enumerate(rows):
            e = eps[r["id"]]
            ids = tok(pol._build_chat_prompt(history_at(e, r["t"])) + f"Action: {r['cand']}\n" + PREFIX, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            two = torch.log_softmax(model(input_ids=ids).logits[0, -1].float()[[yes_id, no_id]], -1)
            f.write(json.dumps({**r, "p_necessary": two[0].exp().item()}) + "\n"); f.flush()
            if j % 200 == 0: print(f"[{j + 1}/{len(rows)}]", flush=True)
    print("DONE score_cands", flush=True)


def cmd_pointer(a):
    with open(ROOT / a.out, "w") as f:
        for r in load_jsonl(ROOT / a.scores):
            ps = r["p_detour"]; t = next((t for t, p in enumerate(ps) if p > a.thr), None)
            if t is None: t = max(range(len(ps)), key=lambda t: ps[t])
            f.write(json.dumps({"id": r["id"], "steps": [t], "mode": None, "t_hat": t, "p": ps[t]}) + "\n")
    print("DONE pointer", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "score"):
        s = sub.add_parser(name); s.add_argument("--traj", default="reports/alfworld/student3b_low100_train_b50.jsonl")
        s.add_argument("--student", default="Qwen/Qwen2.5-3B-Instruct"); s.add_argument("--out", required=True)
        if name == "train":
            s.add_argument("--compressed", default="reports/alfworld/compress_traj/wins.jsonl")
            s.add_argument("--adapter", default=str(ROOT / "reports/alfworld/student3b_low100/final")); s.add_argument("--n", type=int, default=6000)
            s.add_argument("--lr", type=float, default=1e-4); s.add_argument("--batch", type=int, default=16)
        else:
            s.add_argument("--adapter", required=True); s.add_argument("--ids_from", default=None)
    p = sub.add_parser("pointer"); p.add_argument("--scores", required=True); p.add_argument("--out", required=True); p.add_argument("--thr", type=float, default=0.5)
    c = sub.add_parser("score_cands"); c.add_argument("--traj", default="reports/alfworld/student3b_low100_train_b50.jsonl"); c.add_argument("--student", default="Qwen/Qwen2.5-3B-Instruct")
    c.add_argument("--adapter", required=True); c.add_argument("--cands", required=True); c.add_argument("--out", required=True)
    a = ap.parse_args(); {"train": cmd_train, "score": cmd_score, "pointer": cmd_pointer, "score_cands": cmd_score_cands}[a.cmd](a)


if __name__ == "__main__":
    main()
