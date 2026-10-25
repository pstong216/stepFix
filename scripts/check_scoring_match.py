"""Does the training-time scorer (student_logprobs_grad) match the inference scorer used for anchor targets / acting?"""
import json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch
from alfworld_attr_opd import history_at, load_jsonl, student_logprobs_grad
from alfworld_opd_plus import load_policy
O = "reports/alfworld/self_evolve7b"; rows = load_jsonl(ROOT / O / "anchor_kl300.jsonl")[:8]; eps = {e["id"]: e for e in load_jsonl(ROOT / O / "r1_train.jsonl")}
tok, model, pol = load_policy("Qwen/Qwen2.5-7B-Instruct", "", trainable=True)   # fresh LoRA (B=0): identical function to the base model
model.eval()
for r in rows:
    h = history_at(eps[r["id"]], r["t"])
    with torch.no_grad():
        lq = torch.log_softmax(student_logprobs_grad(pol, tok, h, r["cmds"]).float(), 0)
        lp_full = pol.score_admissible_commands(h); lq2 = torch.log_softmax(torch.tensor([lp_full[c] for c in r["cmds"]]), 0)
    tgt = torch.tensor(r["target"]); kl = (tgt * (torch.log(tgt.clamp_min(1e-8)) - lq.cpu())).sum().item()
    kl2 = (tgt * (torch.log(tgt.clamp_min(1e-8)) - lq2)).sum().item()
    print(f"t={r['t']:2d} n_cmds={len(r['cmds']):2d}  KL(target || train-scorer) {kl:.4f}   KL(target || act-scorer) {kl2:.4f}   argmax equal: {int(tgt.argmax()) == int(lq.argmax())}", flush=True)
