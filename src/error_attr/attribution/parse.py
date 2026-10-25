"""Shared parser for the attributor's 3-line output.

Single source of truth for BOTH evaluation (scripts/eval_attributor.py) and
RL reward computation (scripts/train_attributor_grpo.py): if the reward
were parsed differently from the eval metric, GRPO could optimize a
quirk of one parser that the other doesn't reward -- a subtle form of
reward hacking that's cheap to rule out by construction.
"""
from __future__ import annotations

import re

_FIELD_RE = re.compile(
    r"(?:\*\*)?(?P<key>Agent Name|Step Number|Error Mode)(?:\*\*)?[ \t]*:[ \t]*(?:\*\*)?[ \t]*(?P<val>.*?)"
    r"(?=\n[ \t]*(?:\*\*)?(?:Agent Name|Step Number|Error Mode)(?:\*\*)?[ \t]*:|\Z)",
    flags=re.DOTALL,
)


# Taxonomy-shaped code: uppercase letters, dot, digits (P.1, PL.2, R.4 ...),
# bounded so e.g. "CR.7" inside a longer token doesn't match. Same regex as
# the official harness's parse.py.
_CODE_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]+\.\d+)(?![A-Za-z0-9])")


def _extract_mode(raw: str | None) -> str | None:
    """Lenient mode extraction, mirroring whowhen_eval.parse._extract_error_mode:
    take the first taxonomy-shaped code in the field. An un-fine-tuned model
    writes things like 'A.3: Premature Termination' or '**R.2** (Reasoning
    Error)'; the official scorer compares the normalized field against the
    bare code, so without this the mode axis scores 0 for every such output
    (observed: mode reward exactly 0.0 in the first base-model GRPO log).
    SFT outputs are bare codes already, so this is a no-op for them."""
    if not raw:
        return None
    codes = _CODE_RE.findall(raw)
    return codes[0] if codes else raw


def parse_prediction(text: str) -> dict:
    """-> {"agent_name", "step_coord", "error_mode"}; missing fields are None."""
    matches = {m.group("key"): m.group("val").strip() for m in _FIELD_RE.finditer(text or "")}
    return {
        "agent_name": matches.get("Agent Name") or None,
        "step_coord": matches.get("Step Number") or None,
        "error_mode": _extract_mode(matches.get("Error Mode")),
    }
