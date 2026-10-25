"""LLM rephrasing of attributor-input step bodies (for the `llm_compressed` variant).
  dump     -- collect every unique step body (after no_observations) from Who&When Pro text rows and ALFWorld release files
  generate -- rephrase each body with an instruct LLM into a much shorter text that keeps every concrete fact; writes the
              cache read by error_attr.attribution.ablation (rows {"h": sha1(body), "c": compressed, "n0", "n1"})
Step headers (agent name, step coordinate) are never shown to the rephraser and stay verbatim in the prompt.
"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
DATA_ROOT = ROOT / "data" / "whowhen_pro_official"

INSTR = ("Rewrite the following step from an AI agent's transcript as briefly as possible, in at most {w} words. Keep every "
         "concrete fact exactly: entity names, numbers, code identifiers, tool names and arguments, retrieved results, claims, "
         "and any final answer. Keep bracketed section tags such as [output] or [plan] and every code or tool call (e.g. final_answer(...)) verbatim. Do not add, judge, or correct anything; do not mention that you are summarizing.\n\nStep:\n{body}")


def cmd_dump(a):
    from whowhen_eval.render import get_renderer
    from whowhen_eval.run import release_from_row
    from error_attr.attribution.ablation import step_blocks, _HEADER_RE, _OBS_RE, body_key
    seen = {}
    def add(rr):
        for b in step_blocks(rr):
            t = _OBS_RE.sub("", b.text); m = _HEADER_RE.match(t); body = t[m.end():] if m else t
            if body.strip(): seen.setdefault(body_key(body), body)
    n = 0
    for l in open(DATA_ROOT / "data" / "text.jsonl"):
        row = json.loads(l)
        try:
            rel = release_from_row(row, "text", DATA_ROOT); add(get_renderer(rel["framework"])(rel)); n += 1
        except Exception:
            continue
    render = get_renderer("alfagent")
    for f in a.release:
        for l in open(ROOT / f):
            r = json.loads(l); rel = dict(r); rel.update(r.get("extras") or {}); add(render(rel)); n += 1
    with open(ROOT / a.out, "w") as f:
        for h, b in seen.items(): f.write(json.dumps({"h": h, "body": b}) + "\n")
    print(f"{n} transcripts, {len(seen)} unique step bodies -> {a.out}", flush=True)


def cmd_generate(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = [json.loads(l) for l in open(ROOT / a.inp)]
    if a.only_hashes:
        keep = set(json.load(open(ROOT / a.only_hashes))); rows = [r for r in rows if r["h"] in keep]
    out = ROOT / a.out; done = {json.loads(l)["h"] for l in open(out)} if out.exists() else set()
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"; tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
    todo = []
    for r in rows:
        if r["h"] in done: continue
        ids = tok(r["body"], add_special_tokens=False).input_ids; n0 = len(ids)
        if n0 <= a.min_tokens:           # already short: keep verbatim
            todo.append((r, n0, None)); continue
        todo.append((r, n0, tok.decode(ids[: a.max_in_tokens])))
    todo.sort(key=lambda x: x[1]); print(f"{len(todo)} bodies to process", flush=True)
    f = open(out, "a"); t0 = time.time(); k = 0
    short = [x for x in todo if x[2] is None]
    for r, n0, _ in short: f.write(json.dumps({"h": r["h"], "c": r["body"], "n0": n0, "n1": n0}) + "\n")
    f.flush(); long = [x for x in todo if x[2] is not None]
    for s in range(0, len(long), a.batch):
        chunk = long[s: s + a.batch]
        prompts = [tok.apply_chat_template([{"role": "user", "content": INSTR.format(w=max(8, int(n0 * a.ratio * 0.75)), body=b)}],
                                           tokenize=False, add_generation_prompt=True) for _, n0, b in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        mx = max(48, int(max(n0 for _, n0, _ in chunk) * 0.7))
        with torch.no_grad():
            g = model.generate(**enc, do_sample=False, max_new_tokens=min(mx, 1024), pad_token_id=tok.pad_token_id)
        for (r, n0, _), seq in zip(chunk, g[:, enc.input_ids.shape[1]:]):
            c = tok.decode(seq, skip_special_tokens=True).strip()
            f.write(json.dumps({"h": r["h"], "c": c, "n0": n0, "n1": len(tok(c, add_special_tokens=False).input_ids)}) + "\n")
        f.flush(); k += len(chunk)
        if (s // a.batch) % 20 == 0: print(f"[{k}/{len(long)}] {time.time() - t0:.0f}s", flush=True)
    print("DONE generate", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump"); d.add_argument("--out", default="data/compress/bodies.jsonl")
    d.add_argument("--release", nargs="*", default=["reports/alfworld/student3b_low100_train_b50.release.jsonl"])
    g = sub.add_parser("generate"); g.add_argument("--inp", default="data/compress/bodies.jsonl"); g.add_argument("--out", default="data/compress/cache.jsonl")
    g.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct"); g.add_argument("--ratio", type=float, default=0.3)
    g.add_argument("--min_tokens", type=int, default=40); g.add_argument("--max_in_tokens", type=int, default=3000); g.add_argument("--batch", type=int, default=16)
    g.add_argument("--only_hashes", default=None, help="JSON list of body hashes to process (subset experiments)")
    a = ap.parse_args(); {"dump": cmd_dump, "generate": cmd_generate}[a.cmd](a)


if __name__ == "__main__":
    main()
