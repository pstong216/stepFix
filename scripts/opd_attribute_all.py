"""Attribute EVERY usable failure of a self-play collection with Model A (the
chain only attributes its 150-sample). Output: one line per failed trace,
{id, benchmark, raw, pred} -- the W in Attr-OPD (Research Plan Sec. 10, step 2).

    python scripts/opd_attribute_all.py --tag debate   [--attributor reports/attributor_qwen2.5-3b-lora/final]
    python scripts/opd_attribute_all.py --tag debate3b
"""
import argparse, gc, json, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from whowhen_eval.prompts import all_at_once, load_taxonomy, parts_to_text  # noqa: E402
from whowhen_eval.render import get_renderer  # noqa: E402
from whowhen_eval.run import release_from_row  # noqa: E402
from error_attr.attribution.parse import parse_prediction  # noqa: E402
DATA_ROOT = ROOT / "data" / "whowhen_pro_official"


def load_model(path):
    p = Path(path)
    if (p / "adapter_config.json").is_file():
        from peft import PeftConfig, PeftModel
        cfg = PeftConfig.from_pretrained(path); tok = AutoTokenizer.from_pretrained(path)
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path, dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(base, path).merge_and_unload()
    else:
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="auto")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return model.eval(), tok


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=64):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    inputs = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--attributor", default=str(ROOT / "reports" / "attributor_qwen2.5-3b-lora" / "final"))
    ap.add_argument("--benchmarks", default="math,gpqa,mmlu_pro,scibench")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    traj = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data" / "selfplay" / f"{a.tag}_trajectories.jsonl")}
    rows = [json.loads(l) for l in open(ROOT / "data" / "selfplay" / f"{a.tag}_release_rows.jsonl")]
    keep = set(a.benchmarks.split(","))
    failed = [r for r in rows if not traj[r["id"]]["final_correct"] and traj[r["id"]]["final_answer"] is not None and r["benchmark"] in keep]
    out = Path(a.out or ROOT / "reports" / f"opd_{a.tag}" / "attributions_all.jsonl"); out.parent.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy(DATA_ROOT)
    prompts, ids = [], []
    for r in failed:
        rel = release_from_row(r, "text", DATA_ROOT); rr = get_renderer(rel["framework"])(rel)
        prompts.append(parts_to_text(all_at_once(rr, problem=rel["task"]["query"], taxonomy=taxonomy))); ids.append(r["id"])
    print(f"{a.tag}: {len(failed)} usable failures -> {out}", flush=True)
    model, tok = load_model(a.attributor)
    order = sorted(range(len(ids)), key=lambda k: len(prompts[k]))
    n_ok = 0
    with open(out, "w") as f:
        for s in range(0, len(order), a.batch_size):
            idx = order[s:s + a.batch_size]
            try:
                outs = generate(model, tok, [prompts[k] for k in idx])
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache(); outs = [generate(model, tok, [prompts[k]])[0] for k in idx]
            for k, raw in zip(idx, outs):
                pred = parse_prediction(raw); n_ok += all(v is not None for v in pred.values())
                f.write(json.dumps({"id": ids[k], "benchmark": traj[ids[k]]["benchmark"], "raw": raw, "pred": pred}) + "\n")
            f.flush()
            if (s // a.batch_size) % 10 == 0:
                print(f"[{min(s + a.batch_size, len(order))}/{len(order)}]", flush=True)
    print(f"done; parse-complete {n_ok}/{len(ids)}")


if __name__ == "__main__":
    main()
