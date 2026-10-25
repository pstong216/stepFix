"""Trace-input ablation variants (TODO 2b/3), shared by
scripts/ablate_attributor_inputs.py (inference-time) and
scripts/prepare_attributor_data.py --variant (training-time, item 3).

variant_blocks(name, rr, seed) -> edited step blocks; build_prompt(...) ->
(judge prompt text, compression ratio, blocks). Same definitions as the
ablation script so training-time and inference-time variants are identical.
"""
from __future__ import annotations

import dataclasses
import random
import re

from whowhen_eval.prompts import all_at_once, parts_to_text
from whowhen_eval.render import TASK_ANCHOR

VARIANTS = ["full", "no_observations", "no_agent_ids", "no_task", "prefix_half", "suffix_half",
            "shuffled", "first_line", "extractive", "final_only", "llm_compressed"]

_OBS_RE = re.compile(r"\[observation\].*?\[/observation\]\n?", re.DOTALL)
_AGENT_RE = re.compile(r"(\| Agent: )([^\n]+)")
_KEEP_LINE_RE = re.compile(r"\d|\\boxed|FINAL ANSWER|=|answer", re.IGNORECASE)
_HEADER_RE = re.compile(r"^Step [^\n]*\n")


def step_blocks(rr):
    return [b for b in rr.blocks if b.coord is not None and b.coord != TASK_ANCHOR]


def rebuild(rr, blocks):
    others = [b for b in rr.blocks if b.coord is None or b.coord == TASK_ANCHOR]
    return dataclasses.replace(rr, blocks=others + blocks, step_index=[(b.coord, ()) for b in blocks],
                               trajectory_length=len(blocks))


def _map_body(block, fn):
    m = _HEADER_RE.match(block.text)
    header, body = (m.group(0), block.text[m.end():]) if m else ("", block.text)
    return dataclasses.replace(block, text=header + fn(body))


_CACHE = None


def body_key(body):
    import hashlib
    return hashlib.sha1(body.strip().encode()).hexdigest()


def _compress_cache():
    """{sha1(body): compressed body} from $ERROR_ATTR_COMPRESS_CACHE (jsonl rows {"h", "c"})."""
    global _CACHE
    if _CACHE is None:
        import json, os
        _CACHE = {}
        p = os.environ.get("ERROR_ATTR_COMPRESS_CACHE")
        if p and os.path.exists(p):
            for l in open(p):
                r = json.loads(l); _CACHE[r["h"]] = r["c"]
    return _CACHE


def variant_blocks(name, rr, seed=0):
    steps = step_blocks(rr)
    n = len(steps)
    if name == "full" or name == "no_task":
        return steps
    if name == "no_observations":
        return [dataclasses.replace(b, text=_OBS_RE.sub("", b.text)) for b in steps]
    if name == "no_agent_ids":
        return [dataclasses.replace(b, text=_AGENT_RE.sub(r"\1[hidden]", b.text)) for b in steps]
    if name == "prefix_half":
        return steps[: max(1, n // 2)]
    if name == "suffix_half":
        return steps[n - max(1, n // 2):]
    if name == "shuffled":
        out = list(steps); random.Random(seed).shuffle(out); return out
    if name == "first_line":
        def fl(body):
            lines = [l for l in body.splitlines() if l.strip() and not l.strip().startswith("[")]
            return (lines[0] + "\n") if lines else ""
        return [_map_body(b, fl) for b in steps]
    if name == "extractive":
        def ex(body):
            keep = [l for l in body.splitlines() if l.strip().startswith("[") or _KEEP_LINE_RE.search(l)]
            return "\n".join(keep) + ("\n" if keep else "")
        return [_map_body(b, ex) for b in steps]
    if name == "final_only":
        return []
    if name == "llm_compressed":   # no_observations, then each step body replaced by its LLM rephrasing (cache keyed by sha1)
        cache = _compress_cache(); miss = [0]
        def cp(body):
            c = cache.get(body_key(body))
            if c is None: miss[0] += 1; return body
            return c.strip() + "\n"
        return [_map_body(dataclasses.replace(b, text=_OBS_RE.sub("", b.text)), cp) for b in steps]
    raise ValueError(f"unknown variant {name!r}; choose from {VARIANTS}")


def build_prompt(name, release, rr, taxonomy, seed=0):
    blocks = variant_blocks(name, rr, seed)
    rr2 = rebuild(rr, blocks)
    problem = release["task"].get("query", "")
    if name == "no_task":
        problem = "[task description withheld]"
    if name == "final_only":
        problem = f"{problem}\n\n(The transcript is withheld. The system's final answer was: {rr.final_answer!r})"
    text = parts_to_text(all_at_once(rr2, problem=problem, taxonomy=taxonomy))
    full_chars = sum(len(b.text) for b in step_blocks(rr))
    kept_chars = sum(len(b.text) for b in blocks)
    return text, (kept_chars / full_chars if full_chars else 1.0), blocks
