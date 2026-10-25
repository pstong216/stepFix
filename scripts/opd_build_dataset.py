"""Attr-OPD dataset builder (Research Plan Sec. 10, steps 2-3) for the debate
student. For every FAILED rollout of the student on a TRAIN task:

  * reconstruct the exact prompt each debater saw at each turn (h_t),
  * pick the turn(s) to supervise, per arm:
      attr    -- the turn Model A attributed (agent i*, round r*), weight 1
      random  -- a random other turn, same hindsight text, weight 1 (control: does WHERE matter?)
      uniform -- every turn, generic hindsight, weight 1/T (the plan's uniform-OPD baseline)
  * run the PRIVILEGED TEACHER = the same base model given h_t + hindsight g_t
    (team was wrong; this turn is the decisive error; its predicted error type;
    the original response) and record what it says the turn should have been.

Output rows (reports/opd_<tag>/dataset.jsonl):
  {id, arm, agent, round, weight, messages_student, messages_teacher, original,
   teacher_response, teacher_answer, teacher_correct}
The student is later trained to produce `teacher_response` from `messages_student`
(no hindsight) -- context distillation of privileged information at attributed steps.
"""
import argparse, json, random, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from error_attr.attribution.feedback_conditions import _mode_description  # noqa: E402
from error_attr.selfplay.debate_runner import ROUND0_SYSTEM, REVISE_SYSTEM, is_correct, parse_task_answer  # noqa: E402
DATA_ROOT = ROOT / "data" / "whowhen_pro_official"

HINDSIGHT_STEP = (
    "\n\n## Hindsight (privileged information that was NOT available to you at the time)\n"
    "Your team's final answer to this problem turned out to be INCORRECT, and this turn of yours was identified "
    "as the decisive error. Error type: {mode}\n"
    "Your original response at this point was:\n<<<\n{original}\n>>>\n"
    "Write the response you SHOULD have given at this point instead. Reason carefully and give your final answer "
    "in exactly the format the task specifies."
)
HINDSIGHT_GENERIC = (
    "\n\n## Hindsight (privileged information that was NOT available to you at the time)\n"
    "Your team's final answer to this problem turned out to be INCORRECT (error type somewhere in the team's work: {mode}).\n"
    "Your original response at this point was:\n<<<\n{original}\n>>>\n"
    "Write the response you SHOULD have given at this point instead. Reason carefully and give your final answer "
    "in exactly the format the task specifies."
)


def turn_prompts(rel):
    """(agent_id, round, [system,user] messages, original output) for every debater turn, as debate_runner built them."""
    query = rel["task"]["query"]; rounds = [e for e in rel["trajectory"] if e.get("kind") == "round"]
    out = []
    for e in rounds:
        r = e["round"]
        for i, t in enumerate(e["turns"]):
            if r == 0:
                msgs = [{"role": "system", "content": ROUND0_SYSTEM}, {"role": "user", "content": query}]
            else:
                prev = rounds[r - 1]["turns"]
                peers = "\n\n".join(f"[{p['agent_id']}]:\n{p['output']}" for p in prev if p["agent_id"] != t["agent_id"])
                user = (f"Problem:\n{query}\n\nYour previous answer:\n{prev[i]['output']}\n\n"
                        f"Other agents' answers (last round):\n{peers}\n\nGive your (possibly revised) final answer.")
                msgs = [{"role": "system", "content": REVISE_SYSTEM}, {"role": "user", "content": user}]
            out.append((t["agent_id"], r, msgs, t["output"]))
    return out


@torch.no_grad()
def chat_batch(model, tok, message_lists, max_new_tokens):
    tok.padding_side = "left"
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in message_lists]
    inputs = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="student rollouts: data/selfplay/<tag>_trajectories.jsonl")
    ap.add_argument("--teacher_model", required=True, help="the student's own base model (self-distillation with privileged info)")
    ap.add_argument("--attributions", default=None, help="default reports/opd_<tag>/attributions_all.jsonl")
    ap.add_argument("--split_file", default=str(ROOT / "data" / "selfplay" / "opd_task_split.json"))
    ap.add_argument("--arms", default="attr,random,uniform")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output file (default reports/opd_<tag>/dataset.jsonl); use e.g. dataset_teacher14b.jsonl for a stronger teacher")
    a = ap.parse_args()
    out_dir = ROOT / "reports" / f"opd_{a.tag}"; out_dir.mkdir(parents=True, exist_ok=True)
    att = {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(a.attributions or out_dir / "attributions_all.jsonl")}
    split = json.load(open(a.split_file))
    traj = [json.loads(l) for l in open(ROOT / "data" / "selfplay" / f"{a.tag}_trajectories.jsonl")]
    train_fail = [r for r in traj if r["id"] in att and split.get(r["task_source_id"]) == "train"]
    rng = random.Random(a.seed); arms = a.arms.split(",")
    jobs = []  # (row dict without teacher fields, teacher messages)
    n_skip = 0
    for rel in train_fail:
        pred = att[rel["id"]]
        if any(v is None for v in pred.values()):
            n_skip += 1; continue
        turns = turn_prompts(rel); mode_txt = _mode_description(pred["error_mode"], DATA_ROOT)
        r_star = int(pred["step_coord"].split(".")[0]) if pred["step_coord"].split(".")[0].isdigit() else 0
        k_attr = next((k for k, (ag, r, _, _) in enumerate(turns) if ag == pred["agent_name"] and r == r_star), None)
        if k_attr is None:
            k_attr = next((k for k, (ag, r, _, _) in enumerate(turns) if r == r_star), 0)
        def make(k, arm, weight, template):
            ag, r, msgs, orig = turns[k]
            tmsgs = [dict(msgs[0]), {"role": "user", "content": msgs[1]["content"] + template.format(mode=mode_txt, original=orig)}]
            return ({"id": rel["id"], "benchmark": rel["benchmark"], "gold": rel["task"]["answer"], "arm": arm, "agent": ag, "round": r,
                     "weight": weight, "messages_student": msgs, "original": orig}, tmsgs)
        if "attr" in arms:
            jobs.append(make(k_attr, "attr", 1.0, HINDSIGHT_STEP))
        if "random" in arms:
            others = [k for k in range(len(turns)) if k != k_attr] or [k_attr]
            jobs.append(make(rng.choice(others), "random", 1.0, HINDSIGHT_STEP))
        if "uniform" in arms:
            for k in range(len(turns)):
                jobs.append(make(k, "uniform", 1.0 / len(turns), HINDSIGHT_GENERIC))
    print(f"{a.tag}: {len(train_fail)} train failures ({n_skip} unparsed skipped) -> {len(jobs)} teacher generations", flush=True)
    tok = AutoTokenizer.from_pretrained(a.teacher_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.teacher_model, dtype=torch.bfloat16, device_map="auto").eval()
    order = sorted(range(len(jobs)), key=lambda k: len(jobs[k][1][1]["content"]))
    n_ok = 0
    out_path = Path(a.out) if a.out else out_dir / "dataset.jsonl"
    with open(out_path, "w") as f:
        for s in range(0, len(order), a.batch_size):
            idx = order[s:s + a.batch_size]
            try:
                outs = chat_batch(model, tok, [jobs[k][1] for k in idx], a.max_new_tokens)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache(); outs = [chat_batch(model, tok, [jobs[k][1]], a.max_new_tokens)[0] for k in idx]
            for k, resp in zip(idx, outs):
                row, tmsgs = jobs[k]
                ans = parse_task_answer(row["benchmark"], resp); ok = is_correct(row["benchmark"], ans, row["gold"])
                n_ok += ok
                row.update({"messages_teacher": tmsgs, "teacher_response": resp, "teacher_answer": ans, "teacher_correct": ok})
                f.write(json.dumps(row) + "\n")
            f.flush()
            if (s // a.batch_size) % 10 == 0:
                print(f"[{min(s + a.batch_size, len(order))}/{len(order)}] teacher_correct so far {n_ok}", flush=True)
    print(f"done -> {out_path}; teacher_correct {n_ok}/{len(jobs)}")


if __name__ == "__main__":
    main()
