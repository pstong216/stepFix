"""Attr-OPD training (Research Plan Sec. 10, steps 4-5). Student = base model + LoRA.
For each dataset row: student sees `messages_student` (NO hindsight) and is pulled
toward the privileged teacher's response, weighted by w_t:

  --loss kl : token-level KL( q_T || pi_theta ) over the vocabulary at every token of the
              teacher response. q_T = the SAME base model (adapter disabled) scoring the
              teacher response given the PRIVILEGED prompt; pi_theta = student (adapter on)
              given the plain prompt. This is the plan's formula on the teacher's own sample.
  --loss ce : cross-entropy on the teacher response (sequence-level distillation), same weights.

    python scripts/opd_train.py --student_model Qwen/Qwen2.5-7B-Instruct \
        --data reports/opd_debate/dataset.jsonl --arm attr --output_dir reports/opd_debate/student_attr
"""
import argparse, json, math, random, sys
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
ROOT = Path(__file__).resolve().parent.parent


def encode(tok, messages, response, max_len):
    """Returns (input_ids, n_prompt) with the assistant response + eos appended; None if too long."""
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    r_ids = tok(response, add_special_tokens=False).input_ids + [tok.eos_token_id]
    if len(p_ids) + len(r_ids) > max_len:
        return None
    return p_ids + r_ids, len(p_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--arm", required=True, choices=["attr", "random", "uniform"])
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--loss", default="kl", choices=["kl", "ce"])
    ap.add_argument("--verified_only", action="store_true", help="keep only rows whose teacher response is correct (Verified Attr-OPD)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=6144)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    out_dir = Path(a.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(a.data)]
    rows = [r for r in rows if r["arm"] == a.arm and (r["teacher_correct"] or not a.verified_only)]
    tok = AutoTokenizer.from_pretrained(a.student_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.student_model, dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    examples, n_drop = [], 0
    for r in rows:
        s = encode(tok, r["messages_student"], r["teacher_response"], a.max_len)
        t = encode(tok, r["messages_teacher"], r["teacher_response"], a.max_len) if a.loss == "kl" else s
        if s is None or t is None:
            n_drop += 1; continue
        examples.append((s, t, float(r["weight"])))
    print(f"arm={a.arm} loss={a.loss} rows={len(rows)} kept={len(examples)} dropped(too long)={n_drop}", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = math.ceil(len(examples) * a.epochs / a.grad_accum)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * total)), total)
    dev = model.device; step = 0; model.train()
    for ep in range(a.epochs):
        random.shuffle(examples); run_loss = 0.0
        for k, ((s_ids, n_ps), (t_ids, n_pt), w) in enumerate(examples):
            s_in = torch.tensor([s_ids], device=dev)
            s_logits = model(input_ids=s_in).logits[0, n_ps - 1:-1].float()  # predicts response tokens
            resp = s_in[0, n_ps:]
            if a.loss == "ce":
                loss = F.cross_entropy(s_logits, resp)
            else:
                with torch.no_grad(), model.disable_adapter():
                    t_in = torch.tensor([t_ids], device=dev)
                    t_logits = model(input_ids=t_in).logits[0, n_pt - 1:-1].float()
                assert t_logits.shape[0] == s_logits.shape[0], (t_logits.shape, s_logits.shape)
                loss = F.kl_div(F.log_softmax(s_logits, -1), F.log_softmax(t_logits, -1), log_target=True, reduction="none").sum(-1).mean()
            (w * loss / a.grad_accum).backward(); run_loss += loss.item()
            if (k + 1) % a.grad_accum == 0 or k + 1 == len(examples):
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 5 == 0:
                    print(f"epoch {ep} step {step}/{total} loss {run_loss / min(a.grad_accum, k + 1):.4f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
                run_loss = 0.0
    model.save_pretrained(str(out_dir / "final")); tok.save_pretrained(str(out_dir / "final"))
    json.dump(vars(a) | {"n_examples": len(examples), "n_dropped": n_drop, "optimizer_steps": step}, open(out_dir / "train_meta.json", "w"), indent=1)
    print(f"saved {out_dir / 'final'}")


if __name__ == "__main__":
    main()
