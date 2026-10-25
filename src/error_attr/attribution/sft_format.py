"""Build SFT (trajectory -> attribution) examples from Who&When Pro rows.

Reuses the official ``whowhen_eval`` harness (vendored under
``third_party/whowhen_pro_repo``, installed as the ``whowhen_eval``
package) for three things:

1. ``get_renderer(framework)`` -- turns the framework-specific release JSON
   into the same flat transcript text the paper's own LLM-judge baseline
   sees. Using the identical renderer (rather than reverse-engineering each
   of the 8 text frameworks' raw schemas myself) means the specialized
   attributor trained here and the frontier-LLM-judge baseline (Research
   Plan Sec. 9.B) are evaluated on byte-identical inputs -- a fair RQ1
   comparison by construction.
2. ``prompts.all_at_once`` -- the exact prompt template used for their
   judge baseline. We reuse it verbatim as the SFT input.
3. ``score.score`` -- the per-framework ground-truth/coordinate mapping
   (debate/dylan use round-only coordinates, mathchat folds round+position
   into a flat ordinal, etc.). Rather than reimplementing 8 different
   coordinate conventions, we brute-force scan the renderer's own
   ``step_index`` and keep the first coordinate that scores as correct
   under their official scorer -- this is exactly consistent with their
   own "any position within the round counts" leniency for debate/dylan.

Target format is a 3-line subset of their 4-field response schema (Agent
Name / Step Number / Error Mode) -- Reason is dropped because there is no
ground-truth reason text to supervise against (inventing one would be
fabricated supervision), and `score.py` never scores it anyway.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from whowhen_eval.prompts import Taxonomy, all_at_once, parts_to_text
from whowhen_eval.render import get_renderer
from whowhen_eval.run import release_from_row
from whowhen_eval.score import score as official_score

# Frameworks where the renderer hardcodes a fixed agent label regardless of
# `ground_truth.agent` (which is null for these -- true single-agent
# frameworks). Mirrors whowhen_eval/score.py's per-framework registry.
_FIXED_AGENT_LABEL = {
    "smolagents": "agent",
    "alfagent": "agent",
    "openai_cua": "computer_use_agent",
    "agentoccam": "web_agent",
    "gemini": "web_agent",
}


class UnattributableRow(Exception):
    """Raised when a row's gold agent/step can't be resolved."""


def gold_agent_label(framework: str, ground_truth: dict) -> Optional[str]:
    if framework in _FIXED_AGENT_LABEL:
        return _FIXED_AGENT_LABEL[framework]
    agent = ground_truth.get("agent")
    if agent:
        return agent
    agents = ground_truth.get("agents")
    if agents:
        return agents[0]
    return None


def _gold_step_coord(framework: str, rr, ground_truth: dict, gold_agent: str, gold_mode: str) -> Optional[str]:
    """Brute-force the renderer's own step_index against the official
    scorer to find a coordinate string that scores step=True. Using their
    scorer (rather than re-deriving each framework's round/position math)
    guarantees this matches the semantics `score.py` will later grade us
    against."""
    for coord_str, _native in rr.step_index:
        pred = {"agent_name": gold_agent, "step_coord": coord_str, "error_mode": gold_mode}
        result = official_score(pred, ground_truth, framework)
        if result["step"]:
            return coord_str
    return None


def build_sft_example(row: dict, split: str, data_root: Path, taxonomy: Taxonomy) -> dict[str, Any]:
    """Returns {id, framework, benchmark, prompt, target, gold_agent,
    gold_step, gold_mode}. Raises UnattributableRow if gold agent/step
    can't be resolved (should be rare; caller should count and report
    these rather than silently dropping them)."""
    release = release_from_row(row, split, data_root)
    framework = release["framework"]
    ground_truth = release["ground_truth"]

    renderer = get_renderer(framework)
    rr = renderer(release)

    gold_mode = ground_truth.get("mode")
    gold_agent = gold_agent_label(framework, ground_truth)
    if gold_agent is None:
        raise UnattributableRow(f"{release['id']}: no resolvable gold agent (framework={framework})")

    gold_step = _gold_step_coord(framework, rr, ground_truth, gold_agent, gold_mode)
    if gold_step is None:
        raise UnattributableRow(f"{release['id']}: no rendered step_index entry scores as gold (framework={framework})")

    prompt_parts = all_at_once(rr, problem=release["task"].get("query", ""), taxonomy=taxonomy)
    prompt_text = parts_to_text(prompt_parts)

    target_text = (
        f"Agent Name: {gold_agent}\n"
        f"Step Number: {gold_step}\n"
        f"Error Mode: {gold_mode}"
    )

    return {
        "id": release["id"],
        "framework": framework,
        "benchmark": release["benchmark"],
        "prompt": prompt_text,
        "target": target_text,
        "gold_agent": gold_agent,
        "gold_step": gold_step,
        "gold_mode": gold_mode,
        # Raw ground_truth (round/position/agents, framework-native schema)
        # is kept alongside the resolved gold_step/gold_agent above because
        # whowhen_eval.score's per-framework scorers (mathchat, debate,
        # dylan) need the ORIGINAL round/position fields, not the already-
        # rendered coordinate string -- passing a synthetic
        # {"step": gold_step} dict silently produces empty accept-sets for
        # those frameworks (see eval_attributor.py fix, 2026-09-08).
        "ground_truth": ground_truth,
    }
