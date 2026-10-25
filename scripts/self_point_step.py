"""Teacher-pointed step (user request Oct 5 22:20): the student model itself, reading a COMPACT view of its own failed trajectory
(step number: action -> one-line observation), names the first step that went wrong.  No training; one short generation per failure.
    python scripts/self_point_step.py --traj T --ids D/ids_fail.jsonl --adapter S --out D/pointer.jsonl   (also writes pointer.raw.jsonl)
"""
import argparse, json, os, re, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("ERROR_ATTR_ROOT", str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
SYSTEM = ("You are reviewing a failed attempt at a household task. You will see the task and the numbered list of actions the agent "
          "took, each followed by what the environment showed next. The attempt failed. Identify the FIRST step at which the agent's "
          "action was a mistake, i.e. the earliest step where a different action would have been needed to complete the task. "
          "Answer with the step number only.")
OBS = 90


def compact(e, max_steps=50):
    lines = [f"Task: {e['goal']}", ""]
    for t, s in enumerate(e["steps"][:max_steps]):
        o = " ".join(s["next_obs"].split())[:OBS]
        lines.append(f"{t}: {s['action']} -> {o}")
    lines += ["", f"The attempt failed after {len(e['steps'])} steps. First mistaken step number:"]
    return "\n".join(lines)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    ap = argparse.ArgumentParser(); ap.add_argument("--traj", required=True); ap.add_argument("--ids", required=True); ap.add_argument("--adapter", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); ap.add_argument("--out", required=True); ap.add_argument("--max_t", type=int, default=12); a = ap.parse_args()
    eps = {e["id"]: e for e in map(json.loads, open(ROOT / a.traj))}; ids = [json.loads(l)["id"] for l in open(ROOT / a.ids) if json.loads(l)["id"] in eps]
    tok = AutoTokenizer.from_pretrained(a.model); tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}); model = PeftModel.from_pretrained(model, str(ROOT / a.adapter)).eval()
    out = ROOT / a.out; raw = out.with_name("pointer.raw.jsonl"); rows = []; t0 = time.time(); n_parsed = 0
    for n, i in enumerate(ids):
        e = eps[i]
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": compact(e)}]
        ids_ = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        with torch.no_grad(): g = model.generate(ids_, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
        txt = tok.decode(g[0, ids_.shape[1]:], skip_special_tokens=True); m = re.search(r"\d+", txt)
        t = int(m.group()) if m else 0; parsed = bool(m) and 0 <= t < len(e["steps"]); n_parsed += parsed
        t = min(max(t, 0), min(len(e["steps"]) - 1, a.max_t))   # clamp into the scannable window
        rows.append({"id": i, "t_hat": t, "type": None, "raw": txt.strip()[:20], "parsed": parsed})
        if n % 50 == 0: print(f"[{n + 1}/{len(ids)}] {time.time() - t0:.0f}s", flush=True)
    out.write_text("".join(json.dumps({"id": r["id"], "steps": [r["t_hat"]], "mode": None}) + "\n" for r in rows))
    raw.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ts = [r["t_hat"] for r in rows]
    print(f"DONE teacher-pointed steps: {len(rows)} | parsed {n_parsed} | mean t {sum(ts) / len(ts):.2f} t0 {sum(t == 0 for t in ts) / len(ts):.2f} t>=5 {sum(t >= 5 for t in ts) / len(ts):.2f}", flush=True)


if __name__ == "__main__":
    main()
