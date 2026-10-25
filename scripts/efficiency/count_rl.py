"""verl metrics -> per-step and cumulative token counts for GiGPO / SERL on ScienceWorld."""
import json, os, re, sys
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ.setdefault("HF_HOME", "/scratch.global/pan00389/hf_cache")
B = "/scratch.global/pan00389/error-attribution/serl_work/outputs"
OUT = os.path.dirname(os.path.abspath(__file__)) + "/parts"; os.makedirs(OUT, exist_ok=True)
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def metrics(path):
    M = {}
    for l in open(path, errors="ignore"):
        l = ANSI.sub("", l)
        if "step:" not in l or "perf/total_num_tokens" not in l: continue
        d = {}
        for kv in l.split(" - "):
            kv = kv.strip().split(" ")[-1] if kv.strip().startswith("(") else kv.strip()
            if ":" in kv:
                k, _, v = kv.rpartition(":")
                try: d[k.split(" ")[-1]] = float(v)
                except ValueError: pass
        if "step" in d: M[int(d["step"])] = d   # last occurrence wins (relaunch)
    return M


def run(name, teacher=False):
    M = metrics(f"{B}/sciworld_{name}/train.log"); steps = []
    tok = None
    if teacher:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    for s in sorted(M):
        if s > 72: break
        d = M[s]; rf = f"{B}/sciworld_{name}/rollout/{s}.jsonl"
        n = sum(1 for _ in open(rf)) if os.path.exists(rf) else None
        tot = d["perf/total_num_tokens"]; pm = d["prompt_length/mean"]; rm = d["response_length/mean"]
        n_est = tot / (pm + rm)
        row = {"step": s, "turns": n, "turns_from_tokens": round(n_est, 1), "total_tokens": tot, "prompt_tokens": pm * (n or n_est), "resp_tokens": rm * (n or n_est),
               "ep_len_mean": d.get("episode/length/mean"), "success": d.get("episode/success_rate")}
        if teacher:
            tf = f"{B}/sciworld_{name}/checkpoints/teacher/{s}.jsonl"
            if os.path.exists(tf):
                texts = [json.loads(l)["teacher_forward_input_text"] for l in open(tf)]
                lens = [len(x) for x in tok(texts, add_special_tokens=False).input_ids]
                row["teacher_fwd_tokens"] = sum(lens); row["teacher_rows"] = len(lens)
        steps.append(row)
    json.dump(steps, open(f"{OUT}/rl_{name}.json", "w"), indent=0)
    bad = [r["step"] for r in steps if r["turns"] and abs(r["turns"] - r["turns_from_tokens"]) > 2]
    print(name, len(steps), "steps; turns vs tokens mismatch at", bad[:10])
    for S in (8, 16, 24, 48, 72):
        sub = [r for r in steps if r["step"] <= S]
        if len(sub) < S: continue
        print(S, {k: round(sum(r[k] for r in sub)) for k in ("turns", "total_tokens", "prompt_tokens", "resp_tokens") + (("teacher_fwd_tokens",) if teacher else ())})


run("gigpo"); run("serl", teacher=True)
