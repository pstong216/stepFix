"""Uncertainty signals on ALFWorld failures, from both sides:
  student    -- at every step of a failed trajectory, the generating policy's distribution over admissible commands
                (entropy, p(chosen action), top-1 minus top-2 margin, rank of the chosen action)
  attributor -- Model A's full distribution over the blamed step: p("Step Number: k\\n") for k = 1..n under teacher
                forcing after the fixed prefix of its 3-line answer, renormalised over k (entropy, p_max, argmax)
Usage:
  python scripts/alfworld_uq.py student    --traj T.jsonl --adapter DIR --out OUT.jsonl
  python scripts/alfworld_uq.py attributor --release R.release.jsonl --attributor DIR --out OUT.jsonl
"""
import argparse, copy, json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PREFIX = "Agent Name: agent\nStep Number: "


def done_ids(out):
    return {json.loads(l)["id"] for l in open(out)} if Path(out).exists() else set()


def dist_stats(lp):
    m = max(lp); p = [math.exp(x - m) for x in lp]; z = sum(p); p = [x / z for x in p]
    ent = -sum(x * math.log(x) for x in p if x > 0)
    return p, ent


def cmd_student(a):
    from peft import PeftModel
    from error_attr.policy.alfworld_agent import LLMPolicy
    from alfworld_attr_opd import history_at, load_jsonl
    eps = [e for e in load_jsonl(a.traj) if not e["won"]]
    done = done_ids(a.out); eps = [e for e in eps if e["id"] not in done]
    print(f"{len(eps)} failed trajectories to score", flush=True)
    tok = AutoTokenizer.from_pretrained(a.student); tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(a.student, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, a.adapter).merge_and_unload().eval()
    pol = LLMPolicy(model, tok)
    with open(a.out, "a") as f:
        for k, e in enumerate(eps):
            rows = []
            for t, st in enumerate(e["steps"]):
                lp = pol.score_admissible_commands_cached(history_at(e, t))
                cmds = list(lp); p, ent = dist_stats([lp[c] for c in cmds])
                pc = dict(zip(cmds, p)); srt = sorted(p, reverse=True)
                ch = st["action"]
                row = {"t": t, "n_adm": len(cmds), "entropy": ent, "p_chosen": pc.get(ch, 0.0),
                       "margin": srt[0] - (srt[1] if len(srt) > 1 else 0.0),
                       "rank_chosen": 1 + sum(1 for x in p if x > pc.get(ch, 0.0)), "p_top": srt[0]}
                if a.save_raw:  # summed log-prob and token count per command, so any normalisation can be recomputed offline
                    row.update({"cmds": cmds, "sum_lp": [lp[c] for c in cmds],
                                "ntok": [len(tok(c, add_special_tokens=False).input_ids) for c in cmds], "chosen": ch})
                rows.append(row)
            f.write(json.dumps({"id": e["id"], "steps": rows}) + "\n"); f.flush()
            if k % 20 == 0: print(f"[{k + 1}/{len(eps)}]", flush=True)
    print("DONE", flush=True)


@torch.no_grad()
def cmd_attributor(a):
    from peft import PeftConfig, PeftModel
    from whowhen_eval.prompts import load_taxonomy
    from whowhen_eval.render import get_renderer
    from error_attr.attribution.ablation import build_prompt
    rows = [json.loads(l) for l in open(a.release)]
    done = done_ids(a.out); rows = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} failures to score with {a.attributor}", flush=True)
    cfg = PeftConfig.from_pretrained(a.attributor)
    tok = AutoTokenizer.from_pretrained(a.attributor); tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, a.attributor).merge_and_unload().eval()
    tax = load_taxonomy(ROOT / "data" / "whowhen_pro_official"); render = get_renderer("alfagent")
    pad = tok.pad_token_id
    with open(a.out, "a") as f:
        for k, r in enumerate(rows):
            rel = dict(r); rel.update(r.get("extras") or {}); rr = render(rel)
            text, _, _ = build_prompt("no_observations", rel, rr, tax)
            msg = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True) + PREFIX
            ids = tok(msg, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device); plen = ids.shape[1]
            out = model(input_ids=ids, use_cache=True); cache = out.past_key_values
            first = torch.log_softmax(out.logits[0, -1].float(), -1)
            n = sum(1 for x in r["trajectory"] if x.get("kind") == "action")
            cands = [tok(f"{s}\n", add_special_tokens=False).input_ids for s in range(1, n + 1)]
            lps = []
            for s0 in range(0, len(cands), a.batch):
                chunk = cands[s0: s0 + a.batch]; m = len(chunk); tails = [c[:-1] for c in chunk]; L = max(len(t) for t in tails)
                c = copy.deepcopy(cache); c.batch_repeat_interleave(m)
                inp = torch.full((m, L), pad, dtype=torch.long); att = torch.zeros((m, plen + L), dtype=torch.long); att[:, :plen] = 1
                for i, t in enumerate(tails):
                    inp[i, :len(t)] = torch.tensor(t); att[i, plen: plen + len(t)] = 1
                o = model(input_ids=inp.to(model.device), attention_mask=att.to(model.device), past_key_values=c, use_cache=True)
                tl = torch.log_softmax(o.logits.float(), -1)
                for i, cid in enumerate(chunk):
                    lp = first[cid[0]].item()
                    if len(cid) > 1:
                        tg = torch.tensor(cid[1:], device=model.device)
                        lp += tl[i, :len(tg)].gather(1, tg.unsqueeze(1)).sum().item()
                    lps.append(lp)
            p, ent = dist_stats(lps)
            am = max(range(n), key=lambda i: p[i])
            f.write(json.dumps({"id": r["id"], "n": n, "p": p, "entropy": ent, "p_max": p[am], "argmax_t": am,
                                "mass_outside_steps": 1.0 - math.exp(torch.logsumexp(torch.tensor(lps), 0).item())}) + "\n"); f.flush()
            if k % 50 == 0: print(f"[{k + 1}/{len(rows)}]", flush=True)
    print("DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("student"); s.add_argument("--traj", required=True); s.add_argument("--adapter", required=True)
    s.add_argument("--student", default="Qwen/Qwen2.5-3B-Instruct"); s.add_argument("--out", required=True)
    s.add_argument("--save_raw", action="store_true")
    t = sub.add_parser("attributor"); t.add_argument("--release", required=True); t.add_argument("--attributor", required=True)
    t.add_argument("--batch", type=int, default=25); t.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cmd_student(a) if a.cmd == "student" else cmd_attributor(a)


if __name__ == "__main__":
    main()
