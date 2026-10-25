"""Generic update-size probe: argmax flips and KL(base || trained) on a fixed set of the model's own states.
usage: measure_flip_generic.py --base_model M --base_adapter A|'' --traj T --states S.jsonl --cache C.json name=adapter ..."""
import argparse, json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from error_attr.policy.alfworld_agent import LLMPolicy
from alfworld_attr_opd import history_at, load_jsonl
ap = argparse.ArgumentParser(); ap.add_argument("--base_model"); ap.add_argument("--base_adapter", default=""); ap.add_argument("--traj"); ap.add_argument("--states"); ap.add_argument("--cache"); ap.add_argument("specs", nargs="+")
a = ap.parse_args(); eps = {e["id"]: e for e in load_jsonl(ROOT / a.traj)}; states = [(r["id"], r["t"]) for r in load_jsonl(ROOT / a.states)]
tok = AutoTokenizer.from_pretrained(a.base_model); raw = AutoModelForCausalLM.from_pretrained(a.base_model, dtype=torch.bfloat16, device_map={"": 0}).eval()
def dists(model):
    pol = LLMPolicy(model, tok); out = {}
    with torch.no_grad():
        for i, t in states:
            lp = pol.score_admissible_commands_cached(history_at(eps[i], t)); z = max(lp.values()); Z = z + math.log(sum(math.exp(v - z) for v in lp.values()))
            out[f"{i}||{t}"] = {c: v - Z for c, v in lp.items()}
    return out
cache = ROOT / a.cache
if cache.exists(): B = json.load(open(cache))
else:
    m = PeftModel.from_pretrained(raw, str(ROOT / a.base_adapter)).eval() if a.base_adapter else raw
    B = dists(m); json.dump(B, open(cache, "w"))
    if a.base_adapter: raw = m.unload()
for spec in a.specs:
    name, adapter = spec.split("=")
    m = PeftModel.from_pretrained(raw, str(ROOT / adapter)).eval(); T = dists(m); keys = list(B)
    flips = sum(max(B[k], key=B[k].get) != max(T[k], key=T[k].get) for k in keys)
    kl = sum(sum(math.exp(B[k][c]) * (B[k][c] - T[k].get(c, -30)) for c in B[k]) for k in keys) / len(keys)
    print(f"{name:30s} argmax flips {flips}/{len(keys)} ({flips / len(keys):.1%})  mean KL(base||trained) {kl:.3f}", flush=True)
    raw = m.unload(); del m; torch.cuda.empty_cache()
