"""Training files for the success/failure-contrast arms (plan doc, section 'Success trajectories'):
pairs at the divergence step of a failed trajectory and a same-game win (identical prefix => identical state).
  dpo_all.jsonl   kind=dpo  (chosen = win's action, rejected = failure's action, reference log-probs under the start policy)
  sft_all.jsonl   kind=kl   one-hot on the chosen action (same states)
  dpo_envf.jsonl / sft_envf.jsonl   subset where the chosen action's one-step evidence (from the win's logged observation) beats the
                                    rejected action's (from the failure's logged observation): 'went right' vs 'got lucky'
    python scripts/dpo_arms_prep.py --pairs reports/alfworld/self_attr_labels/div/dpo_pairs.jsonl --adapter reports/alfworld/student3b_low100/final --out_dir reports/alfworld/self_evolve3b/dpo_arms
"""
import argparse, glob, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch
from alfworld_attr_opd import history_at, load_jsonl
from alfworld_opd_plus import load_policy
from env_evidence_filter import evidence
from alfworld_relaxed_verify import parse_goal

ap = argparse.ArgumentParser(); ap.add_argument("--pairs", required=True); ap.add_argument("--src", default="reports/alfworld/student3b_low100_train_b50.jsonl")
ap.add_argument("--adapter", required=True); ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); ap.add_argument("--out_dir", required=True); a = ap.parse_args()
pairs = [r for r in load_jsonl(ROOT / a.pairs) if r["src"] == a.src]; eps = {e["id"]: e for e in load_jsonl(ROOT / a.src)}
wins = {}
for p in [a.src] + sorted(glob.glob(str(ROOT / "reports/alfworld/self_evolve3b_rounds/*/r*/traj.jsonl"))):
    for e in load_jsonl(p):
        if e["won"]: wins[e["id"]] = e
sc = lambda ev: 2.0 * ev["goal_revealed"] + 2.0 * ev["subgoal"] + min(ev["n_new"], 5) / 5.0 - 2.0 * ev["no_info"]
tok, model, pol = load_policy(a.model, str(ROOT / a.adapter)); model.eval(); out = ROOT / a.out_dir; out.mkdir(parents=True, exist_ok=True)
files = {n: open(out / f"{n}.jsonl", "w") for n in ("dpo_all", "sft_all", "dpo_envf", "sft_envf")}; n_envf = 0; n_all = 0
with torch.no_grad():
    for k, r in enumerate(pairs):
        e = eps[r["id"]]; t = r["t"]; w = wins.get(r["win_id"])
        if w is None or t >= len(e["steps"]) or t >= len(w["steps"]): continue
        lp = pol.score_admissible_commands_cached(history_at(e, t)); cmds = list(lp)
        if r["chosen"] not in lp or r["rejected"] not in lp: continue
        obj = parse_goal(e["goal"])["obj"]
        envf = sc(evidence(e, t, r["chosen"], w["steps"][t]["next_obs"], obj)) > sc(evidence(e, t, r["rejected"], e["steps"][t]["next_obs"], obj))
        dpo = {"kind": "dpo", "src": a.src, "id": r["id"], "t": t, "chosen": r["chosen"], "rejected": r["rejected"], "ref_c": lp[r["chosen"]], "ref_r": lp[r["rejected"]]}
        sft = {"kind": "kl", "src": a.src, "id": r["id"], "t": t, "cmds": cmds, "target": [1.0 if c == r["chosen"] else 0.0 for c in cmds], "cand": r["chosen"]}
        files["dpo_all"].write(json.dumps(dpo) + "\n"); files["sft_all"].write(json.dumps(sft) + "\n"); n_all += 1
        if envf: files["dpo_envf"].write(json.dumps(dpo) + "\n"); files["sft_envf"].write(json.dumps(sft) + "\n"); n_envf += 1
        if k % 50 == 0: print(f"[{k + 1}/{len(pairs)}]", flush=True)
for f in files.values(): f.close()
print(f"DONE dpo_prep: {n_all} pairs, {n_envf} with chosen env-evidence > rejected", flush=True)
