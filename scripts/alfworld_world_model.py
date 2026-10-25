"""Action-conditioned world model of the student (phase 0). One LoRA on the student backbone, four heads selected by
a text prefix appended after the policy prompt and the candidate action:
  dyn  "Next observation: <text>"                      -- environment dynamics
  find "Goal object visible after this action: yes/no" -- search belief (where is the object?)
  sub  "Current subgoal done within 10 steps: yes/no"  -- subgoal reachability under the student
  q    "Task success after greedy continuation: yes/no"-- global success under the student, mostly off-trajectory actions
  train -- multitask SFT from the student adapter
  eval  -- test AUC for find/sub/q (plus baselines: student action prob; location prior for find), dyn token NLL
"""
import argparse, collections, json, math, random, re, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

PREFIX = {"dyn": "[World model] Next observation: ", "find": "[World model] Goal object visible after this action: ",
          "sub": "[World model] Current subgoal done within 10 steps: ", "q": "[World model] Task success after greedy continuation: "}


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


class Eps:
    def __init__(self):
        self.cache = {}
    def get(self, src, i):
        if src not in self.cache: self.cache[src] = {e["id"]: e for e in load_jsonl(ROOT / src)}
        return self.cache[src][i]


def texts(pol, eps, head, r):
    from alfworld_attr_opd import history_at
    e = eps.get(r["src"], r["id"]); h = history_at(e, r["t"])
    prompt = pol._build_chat_prompt(h) + r["action"] + "\n" + PREFIX[head]
    tgt = r["target"] if head == "dyn" else ("yes" if r["label"] else "no")
    return prompt, tgt + "\n"


def load(student, adapter, trainable):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from error_attr.policy.alfworld_agent import LLMPolicy
    tok = AutoTokenizer.from_pretrained(student); tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(student, dtype=torch.bfloat16, device_map={"": 0})
    if trainable:
        base.gradient_checkpointing_enable(); base.enable_input_require_grads()
        m = PeftModel.from_pretrained(base, adapter, is_trainable=True); m.train()
    else:
        m = PeftModel.from_pretrained(base, adapter).eval()
    return tok, m, LLMPolicy(m, tok)


def cmd_train(a):
    import torch
    rng = random.Random(a.seed); D = ROOT / a.data; items = []
    for head, n in [("dyn", a.n_dyn), ("find", a.n_find), ("sub", a.n_sub), ("q", a.n_q)]:
        rs = load_jsonl(D / f"{head}_train.jsonl"); rng.shuffle(rs); items += [(head, r) for r in rs[:n]]
    rng.shuffle(items); print(f"{len(items)} rows: {collections.Counter(h for h, _ in items)}", flush=True)
    tok, model, pol = load(a.student, a.adapter, True); eps = Eps()
    params = [p for p in model.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True); t0 = time.time(); run = collections.defaultdict(list)
    for k, (head, r) in enumerate(items):
        p, t = texts(pol, eps, head, r)
        p_ids = tok(p, add_special_tokens=False).input_ids; t_ids = tok(t, add_special_tokens=False).input_ids
        ids = torch.tensor([p_ids + t_ids], device=model.device); lab = torch.tensor([[-100] * len(p_ids) + t_ids], device=model.device)
        loss = model(input_ids=ids, labels=lab).loss; (loss / a.batch).backward(); run[head].append(loss.item())
        if (k + 1) % a.batch == 0 or k == len(items) - 1:
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
        if (k + 1) % 500 == 0:
            print(f"[{k + 1}/{len(items)}] " + " ".join(f"{h} {sum(v) / len(v):.3f}" for h, v in sorted(run.items())) + f" {time.time() - t0:.0f}s", flush=True); run.clear()
        if (k + 1) % 10000 == 0: model.save_pretrained(str(out / "adapter"))
    model.save_pretrained(str(out / "adapter")); tok.save_pretrained(str(out / "adapter")); print("DONE train", flush=True)


def auc(y, s):
    pos = [x for x, l in zip(s, y) if l]; neg = [x for x, l in zip(s, y) if not l]
    if not pos or not neg: return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg]); rank = {}; i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]: j += 1
        for k in range(i, j): rank.setdefault(allv[k][0], (i + j + 1) / 2)
        i = j
    return (sum(rank[v] for v in pos) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def cmd_eval(a):
    import torch
    D = ROOT / a.data; rng = random.Random(0); out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    rows = {h: load_jsonl(D / f"{h}_test.jsonl") for h in ("find", "sub", "q", "dyn")}
    for h, n in (("find", a.n_eval), ("sub", a.n_eval), ("dyn", a.n_eval_dyn)): rng.shuffle(rows[h]); rows[h] = rows[h][:n]
    res = {}; eps = Eps()
    for tag, adapter in (("wm", a.adapter), ("student", a.student_adapter)):
        tok, model, pol = load(a.student, adapter, False)
        yes, no = tok("yes", add_special_tokens=False).input_ids[0], tok("no", add_special_tokens=False).input_ids[0]
        with torch.no_grad():
            for h in ("find", "sub", "q"):
                sc = []
                for r in rows[h]:
                    if tag == "wm":
                        p, _ = texts(pol, eps, h, r); ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                        lg = model(input_ids=ids).logits[0, -1].float(); sc.append((lg[yes] - lg[no]).item())
                    else:   # baseline: the student's own probability of taking this action
                        from alfworld_attr_opd import history_at
                        e = eps.get(r["src"], r["id"]); lp = pol.score_admissible_commands(history_at(e, r["t"]))
                        z = max(lp.values()); sc.append(lp.get(r["action"], -1e9) - z - math.log(sum(math.exp(v - z) for v in lp.values())))
                res[f"{h}_{tag}"] = sc; print(f"{tag} {h}: AUC {auc([r['label'] for r in rows[h]], sc):.3f} (n={len(sc)})", flush=True)
            nll = []
            for r in rows["dyn"]:
                p, t = texts(pol, eps, "dyn", r) if tag == "wm" else (pol._build_chat_prompt(__import__("alfworld_attr_opd").history_at(eps.get(r["src"], r["id"]), r["t"])) + r["action"] + "\n" + PREFIX["dyn"], r["target"] + "\n")
                p_ids = tok(p, add_special_tokens=False).input_ids; t_ids = tok(t, add_special_tokens=False).input_ids
                ids = torch.tensor([p_ids + t_ids], device=model.device); lab = torch.tensor([[-100] * len(p_ids) + t_ids], device=model.device)
                nll.append(model(input_ids=ids, labels=lab).loss.item())
            res[f"dyn_nll_{tag}"] = sum(nll) / len(nll); print(f"{tag} dyn token NLL {res[f'dyn_nll_{tag}']:.3f}", flush=True)
        del model; torch.cuda.empty_cache()
    # non-LM baseline for find: P(goal type visible | location type) from the training rows
    tr = load_jsonl(D / "find_train.jsonl"); cnt = collections.defaultdict(lambda: [1, 2])
    def key(r):
        e = eps.get(r["src"], r["id"]); from alfworld_relaxed_verify import parse_goal
        loc = re.sub(r"^(go to|open)\s+", "", r["action"]); return (parse_goal(e["goal"])["obj"], re.sub(r"\s*\d+$", "", loc))
    for r in tr: c = cnt[key(r)]; c[0] += r["label"]; c[1] += 1
    prior = [cnt[key(r)][0] / cnt[key(r)][1] for r in rows["find"]]
    print(f"prior find: AUC {auc([r['label'] for r in rows['find']], prior):.3f}", flush=True)
    summ = {h: {"wm": auc([r["label"] for r in rows[h]], res[f"{h}_wm"]), "student_prob": auc([r["label"] for r in rows[h]], res[f"{h}_student"])} for h in ("find", "sub", "q")}
    summ["find"]["location_prior"] = auc([r["label"] for r in rows["find"]], prior)
    for sub_name, f in (("q_orig", lambda r: r["orig"]), ("q_alt", lambda r: not r["orig"]), ("q_round1", lambda r: r["round"] == "round1"), ("q_round2", lambda r: r["round"] == "round2")):
        idx = [k for k, r in enumerate(rows["q"]) if f(r)]
        summ[sub_name] = {"n": len(idx), "wm": auc([rows["q"][k]["label"] for k in idx], [res["q_wm"][k] for k in idx]),
                          "student_prob": auc([rows["q"][k]["label"] for k in idx], [res["q_student"][k] for k in idx])}
    summ["dyn_nll"] = {"wm": res["dyn_nll_wm"], "student": res["dyn_nll_student"]}
    json.dump(summ, open(out / "eval_summary.json", "w"), indent=1)
    with open(out / "eval_scores.jsonl", "w") as f:
        for h in ("find", "sub", "q"):
            for r, s1, s2 in zip(rows[h], res[f"{h}_wm"], res[f"{h}_student"]): f.write(json.dumps({"head": h, **r, "wm": s1, "student_prob": s2}) + "\n")
    print(json.dumps(summ, indent=1)); print("DONE eval", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "eval"):
        s = sub.add_parser(name); s.add_argument("--data", default="data/world_model"); s.add_argument("--student", default="Qwen/Qwen2.5-3B-Instruct")
        s.add_argument("--out", required=True)
        if name == "train":
            s.add_argument("--adapter", default=str(ROOT / "reports/alfworld/student3b_low100/final"))
            s.add_argument("--n_dyn", type=int, default=10000); s.add_argument("--n_find", type=int, default=8000)
            s.add_argument("--n_sub", type=int, default=8000); s.add_argument("--n_q", type=int, default=100000)
            s.add_argument("--lr", type=float, default=1e-4); s.add_argument("--batch", type=int, default=16); s.add_argument("--seed", type=int, default=0)
        else:
            s.add_argument("--adapter", required=True); s.add_argument("--student_adapter", default=str(ROOT / "reports/alfworld/student3b_low100/final"))
            s.add_argument("--n_eval", type=int, default=2000); s.add_argument("--n_eval_dyn", type=int, default=500)
    a = ap.parse_args(); {"train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
