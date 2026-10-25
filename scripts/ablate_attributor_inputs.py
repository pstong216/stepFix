"""TODO item 2b -- "Do we really need the full trace for attribution?"

Inference-time input ablations for the SFT attributor (Model A). Each
variant re-renders a trace with the official renderer, edits the rendered
transcript blocks, rebuilds the identical judge prompt, and scores Model A's
prediction with the official scorer. Same model, same scorer, same rows --
only the INPUT changes.

Variants (see TODO.md):
  full            reference
  no_observations drop [observation]...[/observation] bodies (env/tool outputs)
  no_agent_ids    hide agent identity in step headers ("who" removed)
  no_task         withhold the task query
  prefix_half     keep only the first half of the steps (online attribution)
  suffix_half     keep only the last half
  shuffled        random permutation of step blocks (temporal order destroyed;
                  coordinates stay in the headers so the model can still name them)
  first_line      each step's body truncated to its first non-empty line
                  (heavy generic compression)
  extractive      keep only body lines that carry numbers / answers / '=' /
                  boxed / FINAL ANSWER (extractive compression)
  final_only      no transcript at all -- task + the system's final answer

Reports per variant x split: agent/step/mode/all-3 accuracy, per-framework
breakdown, compression ratio (chars kept / chars in full), and for the
truncating variants the fraction of rows whose gold step is still visible.
Per-example files are written in eval_attributor.py's schema so
scripts/compare_attributors.py works on them.

Caveat (in TODO.md): Model A was trained on full traces, so these numbers
mix "information necessity" with distribution shift; they pick candidates
for the retraining experiment (item 3), they don't settle it.

Usage (GPU):
    python scripts/ablate_attributor_inputs.py --splits val,transfer_alfworld --batch_size 8
"""
import argparse
import dataclasses
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from whowhen_eval.prompts import all_at_once, load_taxonomy, parts_to_text  # noqa: E402
from whowhen_eval.render import TASK_ANCHOR, get_renderer  # noqa: E402
from whowhen_eval.run import release_from_row  # noqa: E402
from whowhen_eval.score import score as official_score  # noqa: E402

from error_attr.attribution.parse import parse_prediction  # noqa: E402

DATA_ROOT = ROOT / "data" / "whowhen_pro_official"
SFT_DIR = ROOT / "data" / "attributor_sft"
DEFAULT_MODEL = ROOT / "reports" / "attributor_qwen2.5-3b-sft-merged"
OUT_DIR = ROOT / "reports" / "ablations"

_OBS_RE = re.compile(r"\[observation\].*?\[/observation\]\n?", re.DOTALL)
_AGENT_RE = re.compile(r"(\| Agent: )([^\n]+)")
_KEEP_LINE_RE = re.compile(r"\d|\\boxed|FINAL ANSWER|=|answer", re.IGNORECASE)
_HEADER_RE = re.compile(r"^Step [^\n]*\n")


def _step_blocks(rr):
    return [b for b in rr.blocks if b.coord is not None and b.coord != TASK_ANCHOR]


def _rebuild(rr, blocks):
    """New RenderResult with the given step blocks (task/non-step blocks kept in place)."""
    others = [b for b in rr.blocks if b.coord is None or b.coord == TASK_ANCHOR]
    return dataclasses.replace(rr, blocks=others + blocks, step_index=[(b.coord, ()) for b in blocks],
                               trajectory_length=len(blocks))


def _map_body(block, fn):
    """Apply fn(body_text) -> new body, keeping the 'Step X | Agent: Y' header line."""
    m = _HEADER_RE.match(block.text)
    header, body = (m.group(0), block.text[m.end():]) if m else ("", block.text)
    return dataclasses.replace(block, text=header + fn(body))


def variant_blocks(name, rr, seed):
    steps = _step_blocks(rr)
    n = len(steps)
    if name == "full":
        return steps
    if name == "no_observations":
        return [dataclasses.replace(b, text=_OBS_RE.sub("", b.text)) for b in steps]
    if name == "no_agent_ids":
        return [dataclasses.replace(b, text=_AGENT_RE.sub(r"\1[hidden]", b.text)) for b in steps]
    if name == "no_task":
        return steps  # handled via problem text
    if name == "prefix_half":
        return steps[: max(1, n // 2)]
    if name == "suffix_half":
        return steps[n - max(1, n // 2):]
    if name == "shuffled":
        out = list(steps); random.Random(seed).shuffle(out); return out
    if name == "first_line":
        def fl(body):
            lines = [l for l in body.splitlines() if l.strip() and not l.strip().startswith("[") ]
            return (lines[0] + "\n") if lines else ""
        return [_map_body(b, fl) for b in steps]
    if name == "extractive":
        def ex(body):
            keep = [l for l in body.splitlines() if l.strip().startswith("[") or _KEEP_LINE_RE.search(l)]
            return "\n".join(keep) + ("\n" if keep else "")
        return [_map_body(b, ex) for b in steps]
    if name == "final_only":
        return []
    raise ValueError(name)


def build_prompt(name, release, rr, taxonomy, seed):
    blocks = variant_blocks(name, rr, seed)
    rr2 = _rebuild(rr, blocks)
    problem = release["task"].get("query", "")
    if name == "no_task":
        problem = "[task description withheld]"
    if name == "final_only":
        problem = f"{problem}\n\n(The transcript is withheld. The system's final answer was: {rr.final_answer!r})"
    text = parts_to_text(all_at_once(rr2, problem=problem, taxonomy=taxonomy))
    full_chars = sum(len(b.text) for b in _step_blocks(rr))
    kept_chars = sum(len(b.text) for b in blocks)
    return text, (kept_chars / full_chars if full_chars else 1.0), blocks


@torch.no_grad()
def generate_batch(model, tok, prompts, max_new_tokens=64):
    tok.padding_side = "left"
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    inputs = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def generate_oom_safe(model, tok, prompts, max_new_tokens=64):
    try:
        return generate_batch(model, tok, prompts, max_new_tokens)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(prompts) == 1:
            return [""]
        mid = len(prompts) // 2
        return generate_oom_safe(model, tok, prompts[:mid], max_new_tokens) + generate_oom_safe(model, tok, prompts[mid:], max_new_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--splits", default="val,transfer_alfworld")
    ap.add_argument("--variants", default="full,no_observations,no_agent_ids,no_task,prefix_half,suffix_half,shuffled,first_line,extractive,final_only")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy(DATA_ROOT)

    # rows to evaluate, by split, with their raw release rows for re-rendering
    want = {}
    for split in args.splits.split(","):
        for l in open(SFT_DIR / f"{split}.jsonl"):
            r = json.loads(l); want[r["id"]] = (split, r["gold_step"], r["ground_truth"], r["framework"])
    raw = {}
    for l in open(DATA_ROOT / "data" / "text.jsonl"):
        r = json.loads(l)
        if r["id"] in want:
            raw[r["id"]] = r
    by_split = defaultdict(list)
    for i, (split, *_rest) in want.items():
        by_split[split].append(i)
    if args.limit:
        by_split = {s: ids[: args.limit] for s, ids in by_split.items()}

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto").eval()

    summary_lines = []
    for split, ids in by_split.items():
        rendered = {}
        for i in ids:
            rel = release_from_row(raw[i], "text", DATA_ROOT)
            rendered[i] = (rel, get_renderer(rel["framework"])(rel))
        for name in args.variants.split(","):
            prompts, ratios, gold_visible = [], [], []
            for i in ids:
                rel, rr = rendered[i]
                text, ratio, blocks = build_prompt(name, rel, rr, taxonomy, args.seed)
                prompts.append(text); ratios.append(ratio)
                gold_visible.append(any(b.coord == want[i][1] for b in blocks))
            # length-sorted batching
            order = sorted(range(len(ids)), key=lambda k: len(prompts[k]))
            raw_out = [None] * len(ids)
            for s in range(0, len(order), args.batch_size):
                idx = order[s:s + args.batch_size]
                outs = generate_oom_safe(model, tok, [prompts[k] for k in idx])
                for k, o in zip(idx, outs):
                    raw_out[k] = o
            n = a = st = m = all3 = 0
            per_fw = defaultdict(lambda: [0, 0, 0, 0])
            with open(out_dir / f"{split}__{name}.jsonl", "w") as f:
                for k, i in enumerate(ids):
                    _, _, gt, fw = want[i]
                    pred = parse_prediction(raw_out[k]); res = official_score(pred, gt, fw)
                    n += 1; a += res["agent"]; st += res["step"]; m += res["mode"]; all3 += all(res.values())
                    p = per_fw[fw]; p[0] += 1; p[1] += res["agent"]; p[2] += res["step"]; p[3] += res["mode"]
                    f.write(json.dumps({"id": i, "framework": fw, "raw_pred": raw_out[k], "parsed": pred, "gt": gt,
                                        "result": res, "variant": name, "compression": ratios[k], "gold_step_visible": gold_visible[k]}) + "\n")
            line = (f"{split:18s} {name:16s} n={n} agent={a/n:.3f} step={st/n:.3f} mode={m/n:.3f} all3={all3/n:.3f} "
                    f"| kept_chars={sum(ratios)/n:.2f} | gold_step_visible={sum(gold_visible)/n:.2f}")
            print(line, flush=True); summary_lines.append(line)
            for fw, (c, aa, ss, mm) in sorted(per_fw.items(), key=lambda kv: -kv[1][0]):
                summary_lines.append(f"    {fw:13s} n={c:3d} agent={aa/c:.2f} step={ss/c:.2f} mode={mm/c:.2f}")
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
