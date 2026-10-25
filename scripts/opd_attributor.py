"""On-policy distillation (GKD-style) of the ATTRIBUTOR (Model A): the third
training paradigm after SFT and GRPO (researcher, 2026-09-10).

Student: the best 3B attributor (SFT on agent-text-only prompts) + a fresh LoRA.
Teacher: the 7B attributor trained on the same prompt format (frozen). Both are
Qwen2.5, so their vocabularies coincide and token-level KL is well defined.

Loop (per training prompt, ON-POLICY): the student SAMPLES its 3-line answer
(temperature 1); the teacher and the student are then run on that same sampled
text; loss = sum_t w_t * KL( q_T(.|prefix_t) || pi_theta(.|prefix_t) ) over the
full vocabulary at every sampled token. w_t = 1 (uniform) or up-weights the
`Error Mode:` line (mode-weighted -- mode is the weak axis, the Model-A analogue
of "attribution decides where to supervise"). No gold labels are used.

Usage:
  python scripts/opd_attributor.py --arm uniform  --output_dir reports/attributor_3b_opd_uniform
  python scripts/opd_attributor.py --arm modew    --output_dir reports/attributor_3b_opd_modew
Evaluate with eval_attributor.py (the saved adapter's base is the merged 3B init).
"""
import argparse, json, math, random, sys, time
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, PeftModel, get_peft_model
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from error_attr.attribution.parse import parse_prediction  # noqa: E402

STUDENT_INIT = ROOT / "reports" / "attributor_no_observations_keepids" / "final"   # 3B SFT (agent-text-only), adapter
STUDENT_MERGED = ROOT / "reports" / "attributor_3b_noobs_merged"                    # merged once, becomes the LoRA base
TEACHER = ROOT / "reports" / "attributor_7b_no_observations" / "final"              # 7B SFT (same format), adapter
DATA = ROOT / "data" / "attributor_sft_no_observations" / "train.jsonl"


def load_merged(adapter_dir, dtype=torch.bfloat16, device="cuda"):
    from peft import PeftConfig
    cfg = PeftConfig.from_pretrained(str(adapter_dir))
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path, dtype=dtype, device_map={"": device})
    return PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()


def line_weights(tok, resp_ids, mode_w):
    """Per-token weight: mode_w on tokens of the 'Error Mode:' line, 1 elsewhere."""
    text = tok.decode(resp_ids)
    k = text.find("Error Mode")
    if k < 0 or mode_w == 1.0:
        return torch.ones(len(resp_ids))
    n_before = len(tok(text[:k], add_special_tokens=False).input_ids)
    w = torch.ones(len(resp_ids)); w[n_before:] = mode_w
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["uniform", "modew"])
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--mode_weight", type=float, default=3.0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_prompt_tokens", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=200, help="optimizer steps")
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    out_dir = Path(a.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(STUDENT_INIT))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # --- student init: merge the 3B SFT adapter once and reuse the merged dir as the new LoRA's base
    if not (STUDENT_MERGED / "config.json").is_file():
        m = load_merged(STUDENT_INIT); m.save_pretrained(str(STUDENT_MERGED)); tok.save_pretrained(str(STUDENT_MERGED)); del m; torch.cuda.empty_cache()
        print(f"merged student init -> {STUDENT_MERGED}", flush=True)
    student = AutoModelForCausalLM.from_pretrained(str(STUDENT_MERGED), dtype=torch.bfloat16, device_map={"": 0})
    student.gradient_checkpointing_enable(); student.enable_input_require_grads()
    student = get_peft_model(student, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    student.print_trainable_parameters()
    teacher = load_merged(TEACHER).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    rows = [json.loads(l) for l in open(DATA)]
    random.shuffle(rows)
    if a.limit:
        rows = rows[: a.limit]
    prompts = []
    for r in rows:
        ids = tok(tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True),
                  add_special_tokens=False).input_ids
        if len(ids) <= a.max_prompt_tokens:
            prompts.append((r["id"], ids, r["ground_truth"], r["framework"]))
    print(f"arm={a.arm} prompts={len(prompts)} (of {len(rows)}) epochs={a.epochs}", flush=True)
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = math.ceil(len(prompts) * a.epochs / a.grad_accum)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * total)), total)
    dev = student.device; step = 0; t0 = time.time(); run_kl = 0.0; run_n = 0
    from whowhen_eval.score import score as official_score
    sample_acc = {"agent": 0, "step": 0, "mode": 0, "n": 0}
    n_iters = int(len(prompts) * a.epochs)
    for k in range(n_iters):
        pid, p_ids, gt, fw = prompts[k % len(prompts)]
        p_in = torch.tensor([p_ids], device=dev)
        # ---- on-policy sample from the student (no grad, adapter on)
        student.eval()
        with torch.no_grad():
            gen = student.generate(input_ids=p_in, do_sample=True, temperature=a.temperature, top_p=1.0, max_new_tokens=a.max_new_tokens,
                                   pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        student.train()
        resp = gen[0, p_in.shape[1]:]
        if resp.numel() == 0:
            continue
        pred = parse_prediction(tok.decode(resp, skip_special_tokens=True))
        try:
            sc = official_score(pred, gt, fw)
            for ax in ("agent", "step", "mode"):
                sample_acc[ax] += int(sc[ax])
            sample_acc["n"] += 1
        except Exception:
            pass
        full = torch.cat([p_in[0], resp]).unsqueeze(0)
        n_p = p_in.shape[1]
        # ---- teacher distribution at the sampled positions
        with torch.no_grad():
            t_logits = teacher(input_ids=full).logits[0, n_p - 1:-1].float()
        s_logits = student(input_ids=full).logits[0, n_p - 1:-1].float()
        # Qwen2.5-3B and -7B share the tokenizer (151,665 tokens) but pad the output layer differently
        # (151,936 vs 152,064 rows); the extra rows are never-used padding -- compare on the common vocab.
        V = min(s_logits.shape[-1], t_logits.shape[-1]); s_logits, t_logits = s_logits[:, :V], t_logits[:, :V]
        kl = F.kl_div(F.log_softmax(s_logits, -1), F.log_softmax(t_logits, -1), log_target=True, reduction="none").sum(-1)  # per token
        w = line_weights(tok, resp.tolist(), a.mode_weight if a.arm == "modew" else 1.0).to(dev)
        loss = (w * kl).sum() / w.sum()
        (loss / a.grad_accum).backward(); run_kl += kl.mean().item(); run_n += 1
        if (k + 1) % a.grad_accum == 0 or k + 1 == n_iters:
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step % 10 == 0:
                n = max(1, sample_acc["n"])
                print(f"step {step}/{total} ({k+1}/{n_iters} prompts, {time.time()-t0:.0f}s) mean_tokKL {run_kl/run_n:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} | on-policy sample acc agent {sample_acc['agent']/n:.3f} step {sample_acc['step']/n:.3f} mode {sample_acc['mode']/n:.3f}", flush=True)
                run_kl = 0.0; run_n = 0; sample_acc = {"agent": 0, "step": 0, "mode": 0, "n": 0}
            if step % a.save_every == 0:
                student.save_pretrained(str(out_dir / f"step{step}")); tok.save_pretrained(str(out_dir / f"step{step}"))
    student.save_pretrained(str(out_dir / "final")); tok.save_pretrained(str(out_dir / "final"))
    json.dump(vars(a) | {"n_prompts": len(prompts), "optimizer_steps": step, "teacher": str(TEACHER), "student_init": str(STUDENT_INIT)},
              open(out_dir / "train_meta.json", "w"), indent=1)
    print(f"saved {out_dir / 'final'}")


if __name__ == "__main__":
    main()
