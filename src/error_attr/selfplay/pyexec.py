"""Sandboxed-ish Python execution for the tool-using self-play frameworks
(mathchat's user_proxy, magentic-one's computerterminal). Mirrors what
AutoGen's MathUserProxyAgent does: run the model's ```python block in a
fresh interpreter, capture stdout/stderr, hard timeout. Runs on the compute
node under the same user -- acceptable for this research setting (the
original frameworks do exactly this), NOT a security boundary.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
MAX_OUTPUT_CHARS = 2000


def extract_python_blocks(text: str) -> list[str]:
    """All ```python ... ``` blocks in order (bare ``` blocks count too, like AutoGen)."""
    return [b.strip() for b in _CODE_BLOCK_RE.findall(text) if b.strip()]


def run_python(code: str, timeout_s: int = 30) -> tuple[str, bool]:
    """Execute `code` in a fresh interpreter in a temp dir. Returns (output, ok).
    Output is stdout (+ stderr tail on error), truncated to MAX_OUTPUT_CHARS."""
    with tempfile.TemporaryDirectory() as cwd:
        try:
            p = subprocess.run(
                [sys.executable, "-c", code], cwd=cwd, capture_output=True, text=True, timeout=timeout_s,
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "MPLBACKEND": "Agg", "HOME": cwd},
            )
        except subprocess.TimeoutExpired:
            return f"Error: execution timed out after {timeout_s} seconds.", False
    out = p.stdout
    if p.returncode != 0:
        err_tail = p.stderr.strip().splitlines()[-3:] if p.stderr.strip() else ["(no stderr)"]
        out = (out + "\n" if out else "") + "Error: " + " | ".join(err_tail)
        ok = False
    else:
        ok = True
    out = out.strip() or "(no output -- did you forget to print?)"
    if len(out) > MAX_OUTPUT_CHARS:
        out = out[:MAX_OUTPUT_CHARS] + f"\n...[truncated, {len(out)} chars total]"
    return out, ok
