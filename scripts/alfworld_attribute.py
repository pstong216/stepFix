"""Model A on ALFWorld failures: render alfagent release rows, generate the diagnosis, parse step -> t_hat (0-based action index).
    python scripts/alfworld_attribute.py --attributor reports/attributor_7b_full/final --release reports/alfworld/student3b_train.release.jsonl --out reports/alfworld/attr7bfull_student3b_train.jsonl
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel
from whowhen_eval.prompts import all_at_once, load_taxonomy, parts_to_text
from whowhen_eval.render import get_renderer
from error_attr.attribution.parse import parse_prediction
from error_attr.attribution.ablation import build_prompt
DATA_ROOT = ROOT / "data" / "whowhen_pro_official"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attributor", required=True); ap.add_argument("--release", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="full", help="input variant: full | no_observations (via error_attr.attribution.ablation)")
    ap.add_argument("--batch", type=int, default=4); ap.add_argument("--max_new_tokens", type=int, default=96); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.release)]
    if a.limit: rows = rows[: a.limit]
    done = set()
    if Path(a.out).exists(): done = {json.loads(l)["id"] for l in open(a.out)}
    rows = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} rows to attribute with {a.attributor} ({a.variant})", flush=True)
    cfg = PeftConfig.from_pretrained(a.attributor)
    tok = AutoTokenizer.from_pretrained(a.attributor); tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, a.attributor).merge_and_unload().eval()
    taxonomy = load_taxonomy(DATA_ROOT)
    render = get_renderer("alfagent")
    with open(a.out, "a") as f:
        for s in range(0, len(rows), a.batch):
            chunk = rows[s: s + a.batch]; prompts = []
            for r in chunk:
                rel = dict(r); rel.update(r.get("extras") or {})
                rr = render(rel)
                if a.variant == "full":
                    text = parts_to_text(all_at_once(rr, problem=r["task"]["query"], taxonomy=taxonomy))
                else:
                    text, _, _ = build_prompt(a.variant, rel, rr, taxonomy)
                prompts.append(text)
            msgs = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
            inputs = tok(msgs, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, do_sample=False, max_new_tokens=a.max_new_tokens, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for r, raw in zip(chunk, texts):
                pred = parse_prediction(raw) or {}
                sc = pred.get("step_coord"); t_hat = int(sc) - 1 if sc and str(sc).isdigit() else None
                f.write(json.dumps({"id": r["id"], "raw": raw, "pred": pred, "t_hat": t_hat, "n_actions": sum(1 for e in r["trajectory"] if e.get("kind") == "action")}) + "\n")
            f.flush()
            if (s // a.batch) % 10 == 0: print(f"[{s + len(chunk)}/{len(rows)}]", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
