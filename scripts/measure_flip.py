"""Update-size probe: on 300 held-out own states (150 from wins, 150 from failures), how many argmax actions flip vs the
untrained model, mean KL(base || trained), and (on the training pairs) how often chosen now beats rejected."""
import json, math, sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from error_attr.policy.alfworld_agent import LLMPolicy
from alfworld_attr_opd import history_at, load_jsonl
O = ROOT / "reports/alfworld/self_evolve7b"; eps = {e["id"]: e for e in load_jsonl(O / "r1_train.jsonl")}
_st = [(r["id"], r["t"]) for r in load_jsonl(O / "anchor_kl300.jsonl")]; states = _st[:75] + _st[150:225]   # 75 from wins, 75 from failures
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct"); base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", dtype=torch.bfloat16, device_map={"": 0}).eval()
def dists(pol):
    out = {}
    with torch.no_grad():
        for i, t in states:
            lp = pol.score_admissible_commands_cached(history_at(eps[i], t)); z = max(lp.values()); Z = z + math.log(sum(math.exp(v - z) for v in lp.values()))
            out[(i, t)] = {c: v - Z for c, v in lp.items()}
    return out
cache = O / "base_dists150_cached.json"
if cache.exists(): B = {tuple(k.split("||")): v for k, v in json.load(open(cache)).items()}; B = {(k[0], int(k[1])): v for k, v in B.items()}
else:
    B = dists(LLMPolicy(base, tok)); json.dump({f"{k[0]}||{k[1]}": v for k, v in B.items()}, open(cache, "w"))
for spec in sys.argv[1:]:
    name, adapter = spec.split("=")
    m = PeftModel.from_pretrained(base, str(ROOT / adapter)).eval(); T = dists(LLMPolicy(m, tok))
    flips = sum(max(B[k], key=B[k].get) != max(T[k], key=T[k].get) for k in states)
    kl = sum(sum(math.exp(B[k][c]) * (B[k][c] - T[k].get(c, -30)) for c in B[k]) for k in states) / len(states)
    print(f"{name:28s} argmax flips {flips}/{len(states)} ({flips / len(states):.1%})  mean KL(base||trained) {kl:.3f}", flush=True)
    m.unload(); del m; torch.cuda.empty_cache()
