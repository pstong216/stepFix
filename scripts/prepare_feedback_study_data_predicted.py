"""Build feedback-study prompts using Model A's (the trained attributor's)
OWN PREDICTIONS instead of gold labels -- the connection between the
attributor and the feedback-granularity study that was missing before.

Restricted to the 232 traces that are BOTH (a) one of the 4 gradable
multi-agent frameworks used in the feedback study and (b) in Model A's
held-out val split (i.e. NOT in its training set) -- using training-set
predictions would just reflect memorized gold labels, not genuine
attributor accuracy. See DECISIONS.md for why this subset was chosen.

For each of these 232 traces, this builds the same 4 conditions as
prepare_feedback_study_data.py, but from Model A's PREDICTED (agent, step,
mode) rather than the ground_truth. The matching GOLD-based comparison
for the same 232 traces is available for free by filtering the existing
data/feedback_study/prompts.jsonl / reports/feedback_study_results.jsonl
to these same trace ids -- no need to regenerate it.

Output: data/feedback_study/prompts_predicted.jsonl (232 x 4 = 928 rows).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from whowhen_eval.prompts import load_taxonomy  # noqa: E402
from whowhen_eval.render import get_renderer  # noqa: E402
from whowhen_eval.run import release_from_row  # noqa: E402

from error_attr.attribution.feedback_conditions import build_feedback_conditions  # noqa: E402

DATA_ROOT = ROOT / "data" / "whowhen_pro_official"
EVAL_VAL_PATH = ROOT / "reports" / "eval_val_finetuned.jsonl"
OUT_PATH = ROOT / "data" / "feedback_study" / "prompts_predicted.jsonl"
GRADABLE_FRAMEWORKS = {"debate", "dylan", "mathchat", "magentic-one"}

CONTINUATION_INSTRUCTION = (
    "\n\n## Coordinator Feedback\n\n{feedback}\n\n"
    "Respond with only your corrected final answer, in the same format as "
    "the original task expects. Do not include any other commentary."
)


def main():
    taxonomy = load_taxonomy(DATA_ROOT)

    # Model A's predictions on its own held-out val split, restricted to
    # the 4 gradable frameworks.
    predictions_by_id = {}
    with open(EVAL_VAL_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r["framework"] in GRADABLE_FRAMEWORKS:
                predictions_by_id[r["id"]] = r["parsed"]

    print(f"Model A predictions available for {len(predictions_by_id)} held-out traces "
          f"across {GRADABLE_FRAMEWORKS}")

    n_written = 0
    n_skipped_no_answer = 0
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(DATA_ROOT / "data" / "text.jsonl") as f, open(OUT_PATH, "w") as out:
        for line in f:
            row = json.loads(line)
            if row["id"] not in predictions_by_id:
                continue

            pred = predictions_by_id[row["id"]]
            release = release_from_row(row, "text", DATA_ROOT)
            rr = get_renderer(release["framework"])(release)
            gold_answer = release["task"].get("answer")
            if gold_answer is None:
                n_skipped_no_answer += 1
                continue

            # Feed Model A's PREDICTED fields where build_feedback_conditions
            # expects gold ones. `ground_truth` is only used internally to
            # resolve the agent label (gold_agent_label reads .get("agent")
            # for these non-fixed-label frameworks), so a synthetic dict
            # carrying the predicted agent under the same key works exactly
            # like the real thing.
            synthetic_gt = {"agent": pred.get("agent_name")}
            predicted_step = pred.get("step_coord")
            predicted_mode = pred.get("error_mode")

            conditions = build_feedback_conditions(
                release["framework"], synthetic_gt, predicted_step, predicted_mode,
                rr.step_index, DATA_ROOT,
            )

            base_context = (
                f"# Task\n\n{release['task'].get('query', '')}\n\n"
                f"## Your Team's Attempt (this attempt was INCORRECT)\n\n{rr.chat_content}"
            )

            for condition_name in ("reason_only", "vague_reason", "precise_reason", "precise_only"):
                feedback_text = getattr(conditions, condition_name)
                prompt = base_context + CONTINUATION_INSTRUCTION.format(feedback=feedback_text)
                out.write(json.dumps({
                    "id": row["id"],
                    "framework": row["framework"],
                    "benchmark": row["benchmark"],
                    "condition": condition_name,
                    "prompt": prompt,
                    "gold_answer": gold_answer,
                    "predicted_agent": pred.get("agent_name"),
                    "predicted_step": predicted_step,
                    "predicted_mode": predicted_mode,
                }) + "\n")
                n_written += 1

    print(f"wrote {n_written} (trace, condition) rows to {OUT_PATH}")
    print(f"skipped {n_skipped_no_answer} traces (missing gold answer)")


if __name__ == "__main__":
    main()
