"""Turn the reasoning-teacher variants into candidate sets for training arms: for every attr pointer state, the FULL distribution over
admissible commands under (reason_hind | reason_plain | modeA_hind), reusing the analyses saved by teacher_reason_auc.py.
Writes <SE>/<variant>/{steps,targets}.jsonl in the usual schema (teacher_logprobs = variant, plain_logprobs = student)."""
import json, math, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch
from alfworld_attr_opd import history_at, load_jsonl
from alfworld_opd_plus import load_policy
from self_evolve import HindsightHistory
from teacher_reason_auc import NoteHistory
SE = ROOT / os.environ.get("SE_DIR", "reports/alfworld/self_evolve3b"); ARM = sys.argv[1] if len(sys.argv) > 1 else "attr"
REASON = SE / "proxy" / f"teacher_reason_{ARM}.jsonl"; MODEA = ROOT / "reports/alfworld/attr7bnoobs_student3b_low100_train_b50.jsonl"
eps = {e["id"]: e for e in load_jsonl(ROOT / "reports/alfworld/student3b_low100_train_b50.jsonl")}
tg = {(r["id"], r["t"]): r for r in load_jsonl(SE / ARM / "targets.jsonl")}; an = {(r["id"], r["t"]): r for r in load_jsonl(REASON)}
modeA = {r["id"]: r for r in load_jsonl(MODEA)} if MODEA.exists() else {}
tok, model, pol = load_policy("Qwen/Qwen2.5-3B-Instruct", str(ROOT / "reports/alfworld/student3b_low100/final")); model.eval()
outs = {}
for v in ("reason_hind", "reason_plain", "modeA_hind"):
    (SE / v).mkdir(parents=True, exist_ok=True); outs[v] = open(SE / v / "targets.jsonl", "w")
    with open(SE / v / "steps.jsonl", "w") as f:
        for l in open(SE / ARM / "steps.jsonl"): f.write(l)
with torch.no_grad():
    for k, ((i, t), r) in enumerate(tg.items()):
        a = an.get((i, t))
        if not a: continue
        e = eps[i]; h = history_at(e, t); hh = HindsightHistory(h, e, t)
        variants = {"reason_hind": NoteHistory(h, "Analysis: " + a["analysis_hind"], hind=hh),
                    "reason_plain": NoteHistory(h, "Note: this attempt FAILED.\nAnalysis: " + a["analysis_plain"])}
        if i in modeA:
            m = modeA[i]; variants["modeA_hind"] = NoteHistory(h, f"An external error attributor says the first decisive error was at step {m['t_hat']} (error mode {m['pred'].get('error_mode')}).", hind=hh)
        for v, hist in variants.items():
            lp = pol.score_admissible_commands_cached(hist)
            outs[v].write(json.dumps({"id": i, "t": t, "a_minus": r["a_minus"], "admissible": r["admissible"], "teacher_logprobs": {c: lp.get(c, -30.0) for c in r["admissible"]},
                                      "plain_logprobs": r["plain_logprobs"], "mode": v}) + "\n")
        if k % 25 == 0: print(f"[{k + 1}/{len(tg)}]", flush=True)
for f in outs.values(): f.close()
print("DONE build_reason_targets", flush=True)
