"""Build the corrective-continuation prompts for the feedback-granularity
study (docs/feedback_granularity_study.md), one row per (trace, condition).

Protocol: render the full failed trajectory (same renderer used for
attributor training) + append one of the 4 feedback conditions as a
coordinator message + ask for a corrected final answer, one shot.

Scoped to the 4 easily-gradable multi-agent frameworks: debate, dylan,
mathchat, magentic-one (macnet/metagpt need code-execution grading,
deferred).

Output: data/feedback_study/prompts.jsonl, one row per (trace_id,
condition) = 2328 traces x 4 conditions.
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
from error_attr.attribution.sft_format import UnattributableRow, build_sft_example  # noqa: E402

DATA_ROOT = ROOT / "data" / "whowhen_pro_official"
OUT_DIR = ROOT / "data" / "feedback_study"
GRADABLE_FRAMEWORKS = {"debate", "dylan", "mathchat", "magentic-one"}

CONTINUATION_INSTRUCTION = (
    "\n\n## Coordinator Feedback\n\n{feedback}\n\n"
    "Respond with only your corrected final answer, in the same format as "
    "the original task expects. Do not include any other commentary."
)


def main():
    taxonomy = load_taxonomy(DATA_ROOT)

    n_written = 0
    n_skipped = 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "prompts.jsonl"

    with open(DATA_ROOT / "data" / "text.jsonl") as f, open(out_path, "w") as out:
        for line in f:
            row = json.loads(line)
            if row["framework"] not in GRADABLE_FRAMEWORKS:
                continue

            try:
                ex = build_sft_example(row, "text", DATA_ROOT, taxonomy)
            except UnattributableRow:
                n_skipped += 1
                continue

            release = release_from_row(row, "text", DATA_ROOT)
            rr = get_renderer(release["framework"])(release)
            gold_answer = release["task"].get("answer")
            if gold_answer is None:
                n_skipped += 1
                continue

            conditions = build_feedback_conditions(
                release["framework"], release["ground_truth"], ex["gold_step"], ex["gold_mode"],
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
                    "id": ex["id"],
                    "framework": ex["framework"],
                    "benchmark": ex["benchmark"],
                    "condition": condition_name,
                    "prompt": prompt,
                    "gold_answer": gold_answer,
                    "gold_agent": ex["gold_agent"],
                    "gold_step": ex["gold_step"],
                    "gold_mode": ex["gold_mode"],
                }) + "\n")
                n_written += 1

    print(f"wrote {n_written} (trace, condition) rows to {out_path}")
    print(f"skipped {n_skipped} traces (unattributable or missing gold answer)")


if __name__ == "__main__":
    main()
