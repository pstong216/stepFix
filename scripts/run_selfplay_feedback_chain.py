"""End-to-end chain on SELF-GENERATED failures (fixes the LIMITATION in
docs/feedback_granularity_study.md):

    Qwen self-play debate failure  ->  Model A attribution (predicted agent/step/mode)
      ->  4 feedback conditions built from the PREDICTED labels (+ a no-info control)
      ->  the SAME Qwen-7B that produced the failure is asked for a corrected answer
      ->  graded against gold; paired stats across conditions.

Here "Your team's answer was incorrect" is finally TRUE: the corrector is
the system that erred. This is also the first true end-to-end use of the
attributor (attribution -> feedback -> correction), not gold labels.

Conditions: the researcher's 4 (reason_only, vague_reason, precise_reason,
precise_only) + `incorrect_only` ("your answer was incorrect", nothing else)
-- a control that separates "any feedback at all" from "which feedback
type", which the external-transcript study could not ask.

Inputs:  data/selfplay/debate_trajectories.jsonl (+ *_release_rows.jsonl)
Outputs: reports/selfplay_chain/{attributions,corrections}.jsonl + summary

Usage (GPU; loads the 3B attributor, then frees it and loads the 7B corrector):
    python scripts/run_selfplay_feedback_chain.py \
        --attributor reports/attributor_qwen2.5-3b-lora/final --corrector Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from whowhen_eval.prompts import all_at_once, load_taxonomy, parts_to_text  # noqa: E402
from whowhen_eval.render import get_renderer  # noqa: E402
from whowhen_eval.run import release_from_row  # noqa: E402

from error_attr.attribution.feedback_conditions import build_feedback_conditions  # noqa: E402
from error_attr.attribution.parse import parse_prediction  # noqa: E402
from error_attr.selfplay.debate_runner import is_correct, parse_task_answer  # noqa: E402

DATA_ROOT = ROOT / "data" / "whowhen_pro_official"
# Self-play frameworks (researcher, 2026-09-09: "debate alone is not enough"):
# debate = parallel peers; mathchat = reasoner + Python-executing proxy;
# magentic-one = hierarchical orchestrator + specialists. Each has its own
# collection file and its own framework-native "extra round" revision protocol.
FW_FILES = {"debate": "debate", "mathchat": "mathchat", "magentic-one": "magentic"}
OUT_DIR = ROOT / "reports" / "selfplay_chain"
CONDITIONS = ["incorrect_only", "reason_only", "vague_reason", "precise_reason", "precise_only"]
# TODO 2a controls (advisor's "is that really true?"): built from deliberately
# WRONG or RANDOM labels. wrong_location = precise_only with a different agent
# + a coordinate from a different round; placebo_reason = reason_only with a
# wrong error type; random_attribution = precise_reason from random agent/step/mode.
CONTROL_CONDITIONS = ["wrong_location", "placebo_reason", "random_attribution"]
ALL_TAXONOMY_CODES = ["P.1", "P.2", "R.1", "R.2", "R.3", "R.4", "PL.1", "PL.2", "A.1", "A.2", "A.3", "A.4", "V.1", "V.2", "C.1", "C.2", "C.3"]

CONTINUATION = (
    "\n\n## Coordinator Feedback\n\n{feedback}\n\n"
    "Respond with only your corrected final answer, in the same format as the original task expects. "
    "Do not include any other commentary."
)
INCORRECT_ONLY = "Your team's answer to this task was incorrect. Please review your team's work and provide a corrected final answer."

# ---- Framing conditions (researcher's question, 2026-09-09): is the null a
# PRESENTATION effect? REVISE_SYSTEM tells agents to keep their answer unless
# a PEER is more convincing, and the coordinator feedback is an unattributed
# paragraph in the user turn. These vary the framing, holding content fixed.
FRAMING_CONDITIONS = ["expert_incorrect_only", "expert_precise_reason", "deferential_system_precise_reason"]
# no_feedback: extra debate rounds with NO coordinator message at all (debate_round only) -- the control for
# "does feedback help beyond simply debating longer?"
NOFB_CONDITIONS = ["no_feedback"]
# Location-precision gradient (researcher, 2026-09-18): the SAME precise_only template
# pointing 2 steps after / before Model A's predicted step (x). Paired with
# precise_only (x) and incorrect_only (no step). Traces where x+-2 fall outside the
# trajectory are skipped for that condition (recorded in the log).
OFFSET_CONDITIONS = ["plus2_only", "minus2_only"]
_LEAD = "Your team's answer to this task was incorrect."
EXPERT_LEAD = ("An independent expert reviewer has verified that your team's final answer is WRONG and has "
               "audited the full transcript. This diagnosis is reliable and should be acted on.")
DEFER_SYSTEM = (
    "You are one of several expert agents in a debate. Your team's previous answer has been verified as "
    "incorrect by an independent reviewer, whose diagnosis appears below under 'Coordinator Feedback'. "
    "Do NOT simply keep your previous answer: re-derive the solution from scratch, taking the reviewer's "
    "diagnosis into account, then give your final answer in exactly the format the task specifies."
)


def expert_framed(text: str) -> str:
    """Swap the neutral lead sentence for the authority-framed one; content after it is unchanged."""
    assert text.startswith(_LEAD), text[:80]
    return EXPERT_LEAD + text[len(_LEAD):]


def load_model(path, dtype=torch.bfloat16):
    p = Path(path)
    if (p / "adapter_config.json").is_file():
        from peft import PeftConfig, PeftModel
        cfg = PeftConfig.from_pretrained(path)
        tok = AutoTokenizer.from_pretrained(path)
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path, dtype=dtype, device_map="auto")
        model = PeftModel.from_pretrained(base, path).merge_and_unload()
    else:
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map="auto")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model.eval(), tok


@torch.no_grad()
def generate_batch(model, tok, prompts, max_new_tokens):
    tok.padding_side = "left"
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    inputs = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def generate_oom_safe(model, tok, prompts, max_new_tokens):
    try:
        return generate_batch(model, tok, prompts, max_new_tokens)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(prompts) == 1:
            return ["[SKIPPED: OOM]"]
        mid = len(prompts) // 2
        return generate_oom_safe(model, tok, prompts[:mid], max_new_tokens) + generate_oom_safe(model, tok, prompts[mid:], max_new_tokens)


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--attributor", default=str(ROOT / "reports" / "attributor_qwen2.5-3b-lora" / "final"))
    ap.add_argument("--attributor_max_new_tokens", type=int, default=64, help="use ~160 for base-derived (GRPO-from-base) attributors")
    ap.add_argument("--corrector", default="Qwen/Qwen2.5-7B-Instruct", help="must be the same model used as the debaters")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="cap number of failed trajectories (first N in file order -- benchmark-sorted!)")
    ap.add_argument("--sample", type=int, default=None, help="instead of --limit: uniform random subsample of N failed trajectories (seeded)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=str(OUT_DIR), help="output directory (default reports/selfplay_chain)")
    ap.add_argument("--n_extra_rounds", type=int, default=1, help="debate_round only: number of debate rounds after the feedback (1 = original protocol); the answer after every round is saved")
    ap.add_argument("--protocol", choices=["one_shot", "debate_round"], default="one_shot",
                     help="one_shot: a single corrected final answer from the corrector model. debate_round: inject the "
                          "coordinator feedback as an EXTRA debate round -- each agent sees the transcript + feedback and "
                          "revises, then majority vote (the faithful test of collaboration dynamics; ~3x the generations)")
    ap.add_argument("--conditions", default=",".join(CONDITIONS),
                     help="comma list from " + ",".join(CONDITIONS + CONTROL_CONDITIONS + FRAMING_CONDITIONS + OFFSET_CONDITIONS))
    ap.add_argument("--benchmarks", default="math,gpqa,mmlu_pro,scibench",
                     help="comma-separated; default = self-contained reasoning tasks. gaia/simpleqa_verified failures "
                          "of a tool-less debate are information-access failures (no retrieval), not attributable "
                          "reasoning errors, so they are analyzed separately if at all")
    ap.add_argument("--ids_file", default=None, help="JSON list of trace ids to run instead of --sample/--limit")
    ap.add_argument("--framework", default="debate", choices=list(FW_FILES),
                     help="which self-play collection to correct (data/selfplay/<fw>_trajectories.jsonl)")
    args = ap.parse_args()
    fw = args.framework
    TRAJ = ROOT / "data" / "selfplay" / f"{FW_FILES[fw]}_trajectories.jsonl"
    ROWS = ROOT / "data" / "selfplay" / f"{FW_FILES[fw]}_release_rows.jsonl"
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy(DATA_ROOT)

    # ---- failed self-play trajectories only ----
    traj = {json.loads(l)["id"]: json.loads(l) for l in open(TRAJ)}
    rows = [json.loads(l) for l in open(ROWS)]
    keep = set(args.benchmarks.split(",")) if args.benchmarks else None
    failed = [r for r in rows if not traj[r["id"]]["final_correct"] and traj[r["id"]]["final_answer"] is not None
              and (keep is None or r["benchmark"] in keep)]
    if args.ids_file:  # explicit trace subset (e.g. the nested-brace re-run, DECISIONS #25); overrides sampling
        want = set(json.load(open(args.ids_file)))
        failed = [r for r in failed if r["id"] in want]
        assert len(failed) == len(want), f"{len(want) - len(failed)} ids not found among failures"
    elif args.sample:
        import random
        random.Random(args.seed).shuffle(failed)
        failed = failed[: args.sample]
    elif args.limit:
        failed = failed[: args.limit]
    from collections import Counter as _C
    print(f"self-play trajectories: {len(rows)} | failed (usable): {len(failed)} | by benchmark: "
          f"{dict(_C(r['benchmark'] for r in failed))} | protocol={args.protocol} | out={OUT_DIR}", flush=True)

    rendered = {}
    for r in failed:
        rel = release_from_row(r, "text", DATA_ROOT)
        rr = get_renderer(rel["framework"])(rel)
        rendered[r["id"]] = (rel, rr)

    # ---- Stage 1: attribution with Model A ----
    model, tok = load_model(args.attributor)
    attributions = {}
    ids = [r["id"] for r in failed]
    prompts = [parts_to_text(all_at_once(rendered[i][1], problem=rendered[i][0]["task"]["query"], taxonomy=taxonomy)) for i in ids]
    with open(OUT_DIR / "attributions.jsonl", "w") as f:
        for s in range(0, len(ids), args.batch_size):
            outs = generate_oom_safe(model, tok, prompts[s:s + args.batch_size], args.attributor_max_new_tokens)
            for i, raw in zip(ids[s:s + args.batch_size], outs):
                pred = parse_prediction(raw)
                attributions[i] = pred
                f.write(json.dumps({"id": i, "raw": raw, "pred": pred}) + "\n")
            f.flush()
            print(f"[attribution {min(s + args.batch_size, len(ids))}/{len(ids)}]", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()
    parse_ok = sum(all(v is not None for v in p.values()) for p in attributions.values())
    print(f"attribution parse-complete: {parse_ok}/{len(ids)}", flush=True)

    # ---- Stage 2: feedback conditions from PREDICTED labels ----
    def consistent_step(rr, pred):
        """debate/dylan are scored round-only, and the SFT targets used the FIRST
        coordinate of the gold round, so Model A emits 'R.0' regardless of which
        agent it names. Re-derive the coordinate as (predicted round, the named
        agent's actual position in that round) so `precise_*` feedback never
        says 'step 0.0, in agent_2's turn' when agent_2 sits at 0.1."""
        step, agent = pred.get("step_coord"), pred.get("agent_name")
        if not step or not agent or "." not in step:
            return step
        rnd = step.split(".")[0]
        for coord, _ in rr.step_index:
            if coord.split(".")[0] == rnd:
                blk = next((b for b in rr.blocks if b.coord == coord), None)
                if blk and blk.text.startswith(f"Step {coord} | Agent: {agent}\n"):
                    return coord
        return step

    import random as _random
    selected = [c for c in args.conditions.split(",") if c]
    for c in selected:
        assert c in CONDITIONS + CONTROL_CONDITIONS + FRAMING_CONDITIONS + OFFSET_CONDITIONS + NOFB_CONDITIONS, c
    if "no_feedback" in selected:
        assert args.protocol == "debate_round" and fw == "debate", "no_feedback only makes sense as extra debate rounds"
    if "deferential_system_precise_reason" in selected:
        assert args.protocol == "debate_round" and fw == "debate", "the system-prompt framing only exists for debate's extra round"

    def wrong_labels(rr, pred, seed):
        """Deterministic per-trace wrong agent / other-round coordinate / wrong mode / random triple."""
        rng = _random.Random(seed)
        agents = sorted({b.text.split("| Agent: ")[1].split("\n")[0] for b in rr.blocks if b.coord and "| Agent: " in b.text})
        coords = [c for c, _ in rr.step_index]
        pa, ps, pm = pred["agent_name"], pred["step_coord"], pred["error_mode"]
        other_agents = [a for a in agents if a != pa] or agents
        wrong_agent = rng.choice(other_agents)
        # A wrong pointer must still be internally consistent (the named agent's
        # own turn), else the team can dismiss it as nonsense rather than as a
        # trusted-but-wrong location. Prefer a different round than predicted.
        def coord_of(agent, rnd):
            for b in rr.blocks:
                if b.coord and b.coord.split(".")[0] == rnd and b.text.startswith(f"Step {b.coord} | Agent: {agent}\n"):
                    return b.coord
            return None
        rounds = sorted({c.split(".")[0] for c in coords})
        pred_round = (ps or "").split(".")[0]
        cand = [coord_of(wrong_agent, r) for r in rounds if r != pred_round]
        cand = [c for c in cand if c] or [coord_of(wrong_agent, r) for r in rounds]
        cand = [c for c in cand if c] or [c for c in coords if c != ps] or coords
        wrong_step = rng.choice(cand)
        wrong_mode = rng.choice([m for m in ALL_TAXONOMY_CODES if m != pm])
        rand_agent, rand_step, rand_mode = rng.choice(agents), rng.choice(coords), rng.choice(ALL_TAXONOMY_CODES)
        return wrong_agent, wrong_step, wrong_mode, rand_agent, rand_step, rand_mode

    jobs = []  # (trace_id, condition, prompt)
    n_offset_skipped = {}
    for i in ids:
        rel, rr = rendered[i]
        pred = dict(attributions[i])
        if fw in ("debate", "dylan"):  # the R.0 artifact is specific to round-scored frameworks
            pred["step_coord"] = consistent_step(rr, pred)
        base_ctx = (f"# Task\n\n{rel['task']['query']}\n\n"
                    f"## Your Team's Attempt (this attempt was INCORRECT)\n\n{rr.chat_content}")
        if "no_feedback" in selected:
            jobs.append((i, "no_feedback", base_ctx + CONTINUATION.format(feedback="")))
        if "incorrect_only" in selected:
            jobs.append((i, "incorrect_only", base_ctx + CONTINUATION.format(feedback=INCORRECT_ONLY)))
        if "expert_incorrect_only" in selected:  # authority framing, NO diagnostic content
            jobs.append((i, "expert_incorrect_only", base_ctx + CONTINUATION.format(feedback=expert_framed(INCORRECT_ONLY))))
        if all(v is not None for v in pred.values()):
            fc = build_feedback_conditions(fw,{"agent": pred["agent_name"]}, pred["step_coord"],
                                           pred["error_mode"], rr.step_index, DATA_ROOT)
            for c in CONDITIONS[1:]:
                if c in selected:
                    jobs.append((i, c, base_ctx + CONTINUATION.format(feedback=getattr(fc, c))))
            if any(c in selected for c in CONTROL_CONDITIONS):
                wa, ws, wm, ra, rs, rm = wrong_labels(rr, pred, seed=hash(i) & 0xFFFF)
                if "wrong_location" in selected:
                    fcw = build_feedback_conditions(fw,{"agent": wa}, ws, pred["error_mode"], rr.step_index, DATA_ROOT)
                    jobs.append((i, "wrong_location", base_ctx + CONTINUATION.format(feedback=fcw.precise_only)))
                if "placebo_reason" in selected:
                    fcp = build_feedback_conditions(fw,{"agent": pred["agent_name"]}, pred["step_coord"], wm, rr.step_index, DATA_ROOT)
                    jobs.append((i, "placebo_reason", base_ctx + CONTINUATION.format(feedback=fcp.reason_only)))
                if "random_attribution" in selected:
                    fcr = build_feedback_conditions(fw,{"agent": ra}, rs, rm, rr.step_index, DATA_ROOT)
                    jobs.append((i, "random_attribution", base_ctx + CONTINUATION.format(feedback=fcr.precise_reason)))
            # framing conditions: identical precise_reason CONTENT, different presentation
            if "expert_precise_reason" in selected:
                jobs.append((i, "expert_precise_reason", base_ctx + CONTINUATION.format(feedback=expert_framed(fc.precise_reason))))
            if "deferential_system_precise_reason" in selected:  # system prompt swapped at generation time (see system_for)
                jobs.append((i, "deferential_system_precise_reason", base_ctx + CONTINUATION.format(feedback=fc.precise_reason)))
            # offset conditions: shift the predicted step +-2 along the renderer's step_index
            if any(c in selected for c in OFFSET_CONDITIONS):
                coords_all = [c for c, _ in rr.step_index]
                if pred["step_coord"] in coords_all:
                    xi = coords_all.index(pred["step_coord"])
                    for cname, off in (("plus2_only", 2), ("minus2_only", -2)):
                        if cname in selected and 0 <= xi + off < len(coords_all):
                            sc = coords_all[xi + off]
                            blk = next((b for b in rr.blocks if b.coord == sc), None)
                            ag = blk.text.split("\n")[0].split("| Agent: ")[1].strip() if blk and "| Agent: " in blk.text.split("\n")[0] else pred["agent_name"]
                            fco = build_feedback_conditions(fw, {"agent": ag}, sc, pred["error_mode"], rr.step_index, DATA_ROOT)
                            jobs.append((i, cname, base_ctx + CONTINUATION.format(feedback=fco.precise_only)))
                        elif cname in selected:
                            n_offset_skipped[cname] = n_offset_skipped.get(cname, 0) + 1

    if n_offset_skipped:
        print(f"offset conditions skipped (x+-2 outside trajectory): {n_offset_skipped}", flush=True)
    # ---- Stage 3: correction by the SAME model that erred ----
    model, tok = load_model(args.corrector)
    jobs.sort(key=lambda j: len(j[2]))
    n = defaultdict(int); correct = defaultdict(int); per_trace = defaultdict(dict)
    changed = defaultdict(int); changed_correct = defaultdict(int)
    n_bc = defaultdict(int); correct_bc = defaultdict(int)   # (benchmark, condition)
    from collections import Counter
    from error_attr.selfplay.debate_runner import REVISE_SYSTEM, normalize
    system_for = {"deferential_system_precise_reason": DEFER_SYSTEM}  # all other conditions: the original REVISE_SYSTEM
    # Framework-native "extra round" for the non-debate frameworks (greedy, like debate_round's generate).
    native_runner = None
    if fw == "mathchat":
        from error_attr.selfplay.mathchat_runner import MathChatConfig, QwenMathChat
        native_runner = QwenMathChat(model, tok, MathChatConfig(temperature=0.0))
    elif fw == "magentic-one":
        from error_attr.selfplay.magentic_runner import MagenticConfig, QwenMagentic
        native_runner = QwenMagentic(model, tok, MagenticConfig(temperature=0.0))

    def feedback_text(prompt: str) -> str:
        return prompt.split("## Coordinator Feedback")[1].split("\n\nRespond with only")[0].strip()

    def native_round_answers(chunk):
        """mathchat / magentic-one: hand the feedback to the framework's own next turn
        (proxy message / orchestrator round) and let it run its native protocol."""
        rels = [rendered[i][0] for (i, _c, _p) in chunk]
        finals = native_runner.revise_many(rels, [feedback_text(p) for (_i, _c, p) in chunk])
        wrapped = []
        for rel, fin in zip(rels, finals):
            b = rel["benchmark"]
            wrapped.append("" if fin is None else (f"FINAL ANSWER: {fin}" if b in ("gaia", "simpleqa_verified") else f"\\boxed{{{fin}}}"))
        return wrapped

    def debate_round_answer(i, feedback_prompt, system=REVISE_SYSTEM):
        """args.n_extra_rounds extra debate rounds (default 1 = the original protocol): every agent sees its own
        last output, its peers' last outputs, AND the coordinator feedback (kept visible in every round; omitted
        entirely for no_feedback); majority vote after each round.
        Returns (final_text_for_parsing of the LAST round, per_agent_outputs of the last round, wrapped answer per round)."""
        rel, rr = rendered[i]
        bench = rel["benchmark"]
        prev = [{"agent_id": t["agent_id"], "output": t["output"]} for t in [e for e in rel["trajectory"] if e.get("kind") == "round"][-1]["turns"]]
        feedback_only = feedback_prompt.split("## Coordinator Feedback")[1].split("\n\nRespond with only")[0].strip()
        fb_block = f"## Coordinator Feedback\n{feedback_only}\n\n" if feedback_only else ""
        per_round = []
        for _r in range(args.n_extra_rounds):
            users = []
            for t in prev:
                peers = "\n\n".join(f"[{p['agent_id']}]:\n{p['output']}" for p in prev if p["agent_id"] != t["agent_id"])
                users.append(f"Problem:\n{rel['task']['query']}\n\nYour previous answer:\n{t['output']}\n\n"
                             f"Other agents' answers (last round):\n{peers}\n\n{fb_block}"
                             "Give your (possibly revised) final answer in exactly the format the task specifies.")
            tok.padding_side = "left"
            texts = [tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": u}],
                                             tokenize=False, add_generation_prompt=True) for u in users]
            inputs = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, do_sample=False, max_new_tokens=768, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            outs = tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            parsed = [parse_task_answer(bench, o) for o in outs]
            votes = Counter(normalize(p) for p in parsed if p is not None)
            if votes:
                top, _ = votes.most_common(1)[0]
                final = next(p for p in parsed if p is not None and normalize(p) == top)
            else:
                final = None
            # wrap so parse_task_answer recovers `final` uniformly downstream
            per_round.append((f"FINAL ANSWER: {final}" if bench in ("gaia", "simpleqa_verified") else f"\\boxed{{{final}}}") if final is not None else "")
            prev = [{"agent_id": t["agent_id"], "output": o} for t, o in zip(prev, outs)]
        return per_round[-1], outs, per_round

    with open(OUT_DIR / "corrections.jsonl", "w") as f:
        for s in range(0, len(jobs), args.batch_size):
            chunk = jobs[s:s + args.batch_size]
            if args.protocol == "one_shot":
                outs = generate_oom_safe(model, tok, [j[2] for j in chunk], 512)
            elif native_runner is not None:  # mathchat / magentic-one native extra round, batched
                try:
                    outs = native_round_answers(chunk)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache(); outs = []
                    for job in chunk:
                        try:
                            outs += native_round_answers([job])
                        except torch.OutOfMemoryError:
                            torch.cuda.empty_cache(); outs.append("")
            else:
                outs = []; rounds_of = []
                for (i, c, prompt) in chunk:
                    try:
                        wrapped, _agents, per_round = debate_round_answer(i, prompt, system_for.get(c, REVISE_SYSTEM))
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache(); wrapped = ""; per_round = []
                    outs.append(wrapped); rounds_of.append(per_round)
            if args.protocol == "one_shot" or native_runner is not None:
                rounds_of = [[] for _ in outs]
            for ((i, c, _), raw), per_round in zip(zip(chunk, outs), rounds_of):
                rel = rendered[i][0]; bench = rel["benchmark"]; gold = rel["task"]["answer"]
                parsed = parse_task_answer(bench, raw); ok = is_correct(bench, parsed, gold)
                orig = traj[i]["final_answer"]
                did_change = not (parsed is not None and orig is not None and normalize(parsed) == normalize(orig))
                n[c] += 1; correct[c] += ok; per_trace[i][c] = ok
                changed[c] += did_change; changed_correct[c] += (did_change and ok)
                n_bc[(bench, c)] += 1; correct_bc[(bench, c)] += ok
                by_round = []
                for rw in per_round:
                    pr = parse_task_answer(bench, rw)
                    by_round.append({"parsed": pr, "correct": bool(is_correct(bench, pr, gold)),
                                     "changed": not (pr is not None and orig is not None and normalize(pr) == normalize(orig))})
                f.write(json.dumps({"id": i, "benchmark": bench, "condition": c, "gold": gold, "parsed": parsed,
                                    "original_answer": orig, "changed": did_change, "correct": ok, "raw": raw,
                                    "by_round": by_round}) + "\n")
            f.flush()  # rows land on disk per batch, so a run is inspectable mid-flight (learned the hard way)
            print(f"[correction {min(s + args.batch_size, len(jobs))}/{len(jobs)}] "
                  + str({c: f"{correct[c]}/{n[c]}" for c in selected if n[c]}), flush=True)

    # ---- Summary + paired McNemar ----
    from scipy.stats import binomtest
    lines = ["=== Self-correction recovery rate by condition (failures of the SAME Qwen-7B debate team) ==="]
    # `selected` (not CONDITIONS): a controls-only run (TODO 2a) otherwise prints an empty summary.
    for c in selected:
        if n[c]:
            lines.append(f"{c:16s}: {correct[c]}/{n[c]} = {correct[c]/n[c]:.4f}")
    # Per-benchmark x condition: the failure pool is scibench-heavy, so the
    # aggregate above is dominated by one benchmark. Chance-level re-guess
    # reference for multiple choice (switch away from a known-wrong option):
    # gpqa 4 options -> 1/3, mmlu_pro 10 options -> 1/9; numeric tasks ~0.
    chance = {"gpqa": 1 / 3, "mmlu_pro": 1 / 9, "math": 0.0, "scibench": 0.0, "gaia": 0.0, "simpleqa_verified": 0.0}
    benches = sorted({b for b, _ in n_bc})
    lines.append("\n=== Recovery rate by benchmark x condition (n per cell; MC re-guess chance in header) ===")
    lines.append(f"{'benchmark':18s} {'n':>4s} " + " ".join(f"{c[:14]:>14s}" for c in selected) + "   chance")
    for b in benches:
        nb = n_bc[(b, selected[0])]
        cells = [f"{correct_bc[(b, c)] / n_bc[(b, c)]:14.3f}" if n_bc[(b, c)] else f"{'-':>14s}" for c in selected]
        lines.append(f"{b:18s} {nb:4d} " + " ".join(cells) + f"   {chance.get(b, float('nan')):.3f}")
    lines.append("\n=== Anchoring: changed own answer | correct given changed  (MC re-guess chance: gpqa/mmlu_pro ~1/(k-1)) ===")
    for c in selected:
        if n[c]:
            lines.append(f"{c:16s}: changed={changed[c]}/{n[c]} = {changed[c]/n[c]:.3f} | "
                         f"correct|changed={changed_correct[c]}/{changed[c] or 1} = {changed_correct[c]/(changed[c] or 1):.3f}")
    lines.append("\n=== Paired McNemar (exact) vs incorrect_only and vs precise_only ===")
    for a, b in [("reason_only", "incorrect_only"), ("precise_only", "incorrect_only"), ("precise_reason", "incorrect_only"),
                 ("reason_only", "precise_only"), ("precise_reason", "precise_only"), ("vague_reason", "precise_only"),
                 ("reason_only", "precise_reason")]:
        a_only = sum(1 for t in per_trace.values() if a in t and b in t and t[a] and not t[b])
        b_only = sum(1 for t in per_trace.values() if a in t and b in t and t[b] and not t[a])
        p = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else float("nan")
        lines.append(f"{a:16s} vs {b:16s}: {a}-only={a_only:3d} {b}-only={b_only:3d} p={p:.4g}")
    summary = "\n".join(lines)
    print(summary)
    (OUT_DIR / "summary.txt").write_text(summary + "\n")


if __name__ == "__main__":
    main()
