"""LESS-style (Xia et al. 2024) cheap gradient-similarity score for candidate (id, t) states, adapted to this
project: instead of scoring whole training EXAMPLES against a fixed target capability, we score individual
step-localised candidates against a reference direction built from a small calibration set of already-VERIFIED
examples, and validate the score against real (already-known) verification outcomes -- a check LESS's own setting
(instruction-tuning data selection) has no equivalent of.

Mechanism: for each candidate, do a single-example (no accumulation) forward+backward of the KL loss at that one
state, using the SAME target construction as `alfworld_attr_opd.py cmd_distill`. Project the resulting per-tensor
LoRA gradients into one shared low-dimensional space via a fixed-seed random sign projection applied per tensor
(generated on the fly, never materialised as one big matrix), summed across tensors -- a feature-hashing-style
sketch of the full gradient, not a single huge projection matrix. Cosine similarity to a reference direction
(the same sketch, averaged over a calibration subset) is the score.

Usage:
  python scripts/alfworld_gradient_score.py --traj <low100_train_b50.jsonl> --calib_ids <json list or auto>
    --candidates <jsonl: one row per (arm, id, t, target_dict, outcome)> --out <jsonl>
"""
import argparse, hashlib, json, math
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import sys
sys.path.insert(0, "scripts")
from alfworld_attr_opd import history_at, student_logprobs_grad, load_jsonl
from error_attr.policy.alfworld_agent import LLMPolicy

PROJ_DIM = 2048


def hashed_bucket_sign(name: str, n: int, device):
    """Feature hashing (Weinberger et al. 2009): for each of the n elements in a tensor, a deterministic
    (seeded by tensor name) random bucket index in [0, PROJ_DIM) and a random +-1 sign. O(n) to generate, O(1)
    memory beyond the two length-n arrays -- never materialises an (n x PROJ_DIM) matrix, which for a ~100k-element
    LoRA tensor projected to PROJ_DIM=2048 would be ~800MB and, summed over ~200 LoRA tensors, infeasible."""
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randint(0, PROJ_DIM, (n,), generator=g)
    sign = (torch.randint(0, 2, (n,), generator=g, dtype=torch.float32) * 2 - 1)
    return idx.to(device), sign.to(device)


def sketch_gradients(model) -> torch.Tensor:
    """Sum, over every trainable (LoRA) tensor with a gradient, that tensor's flattened grad scattered into a
    shared PROJ_DIM-size sketch via feature hashing. O(total LoRA params) time, not O(params x PROJ_DIM)."""
    device = next(model.parameters()).device
    out = torch.zeros(PROJ_DIM, device=device)
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            g = p.grad.detach().float().reshape(-1)
            idx, sign = hashed_bucket_sign(name, g.shape[0], device)
            out.scatter_add_(0, idx, g * sign)
    return out


def build_target(tg, kind, target_top, cmds, exclude_a_minus=True):
    """Reconstruct the exact training target used in the real run, from the already-logged example/skipped
    record, without needing to re-verify (the outcome is already known)."""
    if kind == "verified":
        target = torch.zeros(len(cmds)); target[cmds.index(target_top)] = 1.0  # examples.jsonl only logs target_top, not the full weighted vector when 2 candidates tie; single-winner is the common case (see Sec 3 note)
    else:
        tl = torch.tensor([tg["teacher_logprobs"][c] for c in cmds]); target = torch.softmax(tl, 0)
    if exclude_a_minus and tg["a_minus"] in cmds:
        target[cmds.index(tg["a_minus"])] = 0.0; target = target / target.sum().clamp_min(1e-8)
    return target


def one_example_grad(model, tok, pol, ep, t, tg, kind, target_top, opt):
    cmds = tg["admissible"]; target = build_target(tg, kind, target_top, cmds)
    h = history_at(ep, t)
    lpS = student_logprobs_grad(pol, tok, h, cmds); logqS = torch.log_softmax(lpS, 0)
    loss = -(target.to(logqS.device) * logqS).sum()
    opt.zero_grad(set_to_none=True); loss.backward()
    return sketch_gradients(model).detach().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True); ap.add_argument("--student", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="reports/alfworld/student3b_low100/final")
    ap.add_argument("--candidates", required=True, help="jsonl rows: {arm,id,t,kind,target_top,outcome}")
    ap.add_argument("--calib_n", type=int, default=25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    eps = {e["id"]: e for e in load_jsonl(a.traj)}
    cands = load_jsonl(a.candidates)
    targets_by_arm = {}

    tok = AutoTokenizer.from_pretrained(a.student); tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(a.student, dtype=torch.bfloat16, device_map={"": 0})
    base.gradient_checkpointing_enable(); base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, a.adapter, is_trainable=True)
    model.train()
    pol = LLMPolicy(model, tok)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0)  # never actually steps; just a vehicle for .grad

    def get_targets(arm):
        if arm not in targets_by_arm:
            targets_by_arm[arm] = {(r["id"], r["t"]): r for r in load_jsonl(f"reports/alfworld/opd/student3b_low100/{arm}/targets.jsonl")}
        return targets_by_arm[arm]

    calib = [c for c in cands if c["outcome"] == 1][: a.calib_n]
    calib_keys = {(c["arm"], c["id"], c["t"]) for c in calib}
    print(f"{len(cands)} candidates, {len(calib)} held out for calibration", flush=True)

    ref = torch.zeros(PROJ_DIM)
    for i, c in enumerate(calib):
        tg = get_targets(c["arm"])[(c["id"], c["t"])]
        g = one_example_grad(model, tok, pol, eps[c["id"]], c["t"], tg, c["kind"], c["target_top"], opt)
        ref += g
        if (i + 1) % 5 == 0: print(f"  calib {i+1}/{len(calib)}", flush=True)
    ref = ref / len(calib); ref = ref / ref.norm().clamp_min(1e-8)

    out_f = open(a.out, "w")
    for i, c in enumerate(cands):
        key = (c["arm"], c["id"], c["t"])
        if key in calib_keys:
            out_f.write(json.dumps({**c, "grad_sim": None, "is_calib": True}) + "\n"); continue
        tg = get_targets(c["arm"])[(c["id"], c["t"])]
        g = one_example_grad(model, tok, pol, eps[c["id"]], c["t"], tg, c["kind"], c["target_top"], opt)
        gn = g / g.norm().clamp_min(1e-8)
        sim = float((gn * ref).sum())
        out_f.write(json.dumps({**c, "grad_sim": sim, "is_calib": False}) + "\n"); out_f.flush()
        if (i + 1) % 20 == 0: print(f"  scored {i+1}/{len(cands)}", flush=True)
    out_f.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
