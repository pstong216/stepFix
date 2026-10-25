"""Build the 4 failure-feedback conditions for the collaboration-dynamics
study (see docs/feedback_granularity_study.md).

Given a completed, failed multi-agent trajectory with gold (agent, step,
mode), produces the feedback message a "coordinator" would hand back to
the team under each condition:

  reason_only        -- what kind of error, no location at all.
  vague_reason        -- coarse location (which quarter of the interaction)
                         + what kind of error. No step number, no agent name.
  precise_reason      -- exact step + agent + what kind of error.
                         (= Research Plan Sec. 20's "T3: failure critique".)
  precise_only        -- exact step + agent, no error-type explanation.
                         (= Research Plan Sec. 20's "T1: critical-step
                         indicator".)

Location coarsening (vague_reason) is deliberately framework-agnostic: it
buckets the gold step's ordinal POSITION within the renderer's own
step_index into quarters (Q1..Q4) rather than trying to reproduce each
framework's native round/position semantics vaguely -- a quarter-of-the-
interaction hint is meaningful regardless of whether the underlying
coordinate is a flat step number or a round.position pair.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

from error_attr.attribution.sft_format import gold_agent_label


@dataclass
class FeedbackConditions:
    reason_only: str
    vague_reason: str
    precise_reason: str
    precise_only: str


@lru_cache(maxsize=None)
def _load_short_taxonomy(data_root: Path) -> dict[str, tuple[str, str]]:
    """{code: (name, first_sentence_of_description)}, read directly from
    taxonomy.yaml rather than the pre-formatted judge-prompt block. The
    judge-prompt block's descriptions are written for an annotator
    (multi-sentence, with "Distinguished from X" disambiguation asides) --
    fine for a diagnostic prompt, wrong register for a coordinator's
    natural-language feedback to a team. Only the name + first sentence
    are used here."""
    raw = yaml.safe_load((Path(data_root) / "taxonomy.yaml").read_text())
    out = {}
    for code, body in raw.items():
        if not isinstance(body, dict):
            continue
        name = (body.get("name") or "").strip()
        desc = re.sub(r"\s+", " ", (body.get("description") or "").strip())
        first_sentence = desc.split(". ")[0].rstrip(".") + "."
        out[str(code)] = (name, first_sentence)
    return out


def _mode_description(mode_code: str, data_root: Path) -> str:
    """'Task Misunderstanding -- The agent's mental model of the task is
    wrong at the structural level.' -- name + one plain sentence, no
    judge-facing disambiguation asides."""
    taxonomy_map = _load_short_taxonomy(Path(data_root))
    if mode_code not in taxonomy_map:
        return mode_code  # fallback: bare code if somehow not found
    name, first_sentence = taxonomy_map[mode_code]
    return f"{name} -- {first_sentence}"


def _quarter_bucket(gold_step: str, step_index: list[tuple[str, tuple]]) -> tuple[int, int, int, int]:
    """Returns (quarter, lo, hi, total) where quarter in {1,2,3,4} is which
    quarter of the trajectory the gold step falls in, and [lo, hi] is the
    1-indexed ordinal-position range of that quarter (positions within
    step_index, NOT the framework's native coordinate -- used only to
    phrase "steps N-M of T" in the vague hint)."""
    coords = [c for c, _ in step_index]
    total = len(coords)
    try:
        pos = coords.index(gold_step) + 1  # 1-indexed ordinal position
    except ValueError:
        pos = total  # shouldn't happen (gold_step comes from this same step_index)

    quarter = min(4, max(1, -(-pos * 4 // total)))  # ceil(pos*4/total), clamped to [1,4]
    q_size = -(-total // 4)  # ceil(total/4)
    hi = min(total, quarter * q_size)
    lo = min((quarter - 1) * q_size + 1, hi)  # clamp so lo<=hi even when total < 4
    return quarter, lo, hi, total


_QUARTER_WORDS = {1: "first", 2: "second", 3: "third", 4: "fourth"}


def build_feedback_conditions(
    framework: str,
    ground_truth: dict,
    gold_step: str,
    gold_mode: str,
    step_index: list[tuple[str, tuple]],
    data_root: Path,
) -> FeedbackConditions:
    gold_agent = gold_agent_label(framework, ground_truth)
    mode_desc = _mode_description(gold_mode, data_root)
    quarter, lo, hi, total = _quarter_bucket(gold_step, step_index)
    quarter_word = _QUARTER_WORDS[quarter]

    reason_only = (
        "Your team's answer to this task was incorrect. Somewhere in your "
        f"team's work, the following error occurred: {mode_desc} "
        "Please review your team's reasoning and provide a corrected final answer."
    )

    vague_reason = (
        "Your team's answer to this task was incorrect. The error occurred "
        f"somewhere in the {quarter_word} quarter of the interaction "
        f"(roughly steps {lo}-{hi} of {total}). The error was of this kind: "
        f"{mode_desc} Please review that part of your team's work and "
        "provide a corrected final answer."
    )

    if gold_agent:
        location_clause = f"at step {gold_step}, in {gold_agent}'s turn,"
    else:
        location_clause = f"at step {gold_step}"

    precise_reason = (
        f"Your team's answer to this task was incorrect. Specifically, "
        f"{location_clause} the following error occurred: {mode_desc} "
        "Please review that step and provide a corrected final answer."
    )

    precise_only = (
        f"Your team's answer to this task was incorrect. Specifically, "
        f"{location_clause} an error occurred. Please review that step "
        "and provide a corrected final answer."
    )

    return FeedbackConditions(
        reason_only=reason_only,
        vague_reason=vague_reason,
        precise_reason=precise_reason,
        precise_only=precise_only,
    )
