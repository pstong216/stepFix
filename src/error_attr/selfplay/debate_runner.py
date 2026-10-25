"""Qwen-driven multi-agent debate that emits Who&When Pro `debate` release
schema, so self-generated trajectories flow through the EXISTING official
renderer -> attributor prompt -> Model A -> feedback-conditions pipeline
with no changes.

Why this exists (see docs/feedback_granularity_study.md, LIMITATION): the
feedback study's failed trajectories came from OTHER systems, so the
"team" receiving feedback was never the team that erred. Both fixes need
failures produced by the tested Qwen model itself -- route A (natural
failures, labeled by Model A) and route B (Who&When-style error injection
into Qwen's own successful runs). This runner serves both.

Schema emitted (matches whowhen_eval/render/debate.py exactly):
    release = {
      "id", "framework": "debate", "benchmark", "modality": "text",
      "agents": [{"id": "agent_1", "role": "debater"}, ...],
      "task": {"query", "answer"},
      "trajectory": [
        {"kind": "user", "content": <query>},
        {"kind": "round", "round": 0, "turns": [{"agent_id": "agent_1", "output": ...}, ...]},
        {"kind": "round", "round": 1, "turns": [...]},
        {"kind": "final_answer", "content": <aggregated answer>},
      ],
      "ground_truth": None,   # natural failures have no gold (route A); route B fills {agents, round, mode}
      "final_correct": bool,
    }

Protocol (standard multi-agent debate): round 0, each agent answers the
task independently (it cannot see peers -- matches the renderer's own
step_format_hint). Rounds >= 1, each agent sees all peers' previous-round
outputs and revises. Final answer = majority vote over the last round's
parsed answers (ties -> agent_1's). Correctness is graded with the same
parser/normalizer as scripts/run_feedback_study.py so results are directly
comparable to the external-transcript study on the SAME task pool
(data/selfplay/task_pool.jsonl).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass

import torch

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER:\s*(.+)", re.IGNORECASE)


def _last_boxed(text: str) -> str | None:
    """Content of the LAST \\boxed{...} with proper brace matching. The old regex
    `\\boxed\\{([^}]*)\\}` stopped at the first '}', so \\boxed{\\frac{1}{64}} parsed
    as '\\frac{1' -- a false failure for every nested-brace answer (fractions,
    roots). Found 2026-09-09 in the mathchat collection; see DECISIONS #25."""
    start = text.rfind("\\boxed{")
    while start != -1:
        i, depth = start + len("\\boxed{"), 1
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if depth == 0:
            return text[start + len("\\boxed{"): i - 1].strip()
        start = text.rfind("\\boxed{", 0, start)  # unbalanced: try an earlier one
    return None


def parse_task_answer(benchmark: str, text: str) -> str | None:
    """Same conventions as run_feedback_study.parse_answer: GAIA/SimpleQA
    tasks ask for 'FINAL ANSWER:', everything else for \\boxed{}."""
    if benchmark in ("gaia", "simpleqa_verified"):
        m = _FINAL_ANSWER_RE.findall(text)
        return m[-1].strip().split("\n")[0].strip() if m else None
    return _last_boxed(text)


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9.]", "", s.lower())


_FRAC_RE = re.compile(r"^\\d?frac\{([^{}]+)\}\{([^{}]+)\}$")
_SCI_RE = re.compile(r"^([-+]?\d*\.?\d+)\s*(?:\\times|\\cdot|x|×|\*)\s*10\^\{?([-+]?\d+)\}?$")
_LEAD_NUM_RE = re.compile(r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
NUMERIC_REL_TOL = 0.01  # 1%: stricter than SciBench's own 5% relative-error rule


def to_number(s: str) -> float | None:
    """Best-effort numeric value of an answer string: strips $, \\text{...}, units,
    thousands separators; handles \\frac{a}{b}, a \\times 10^{b}, 1.2e5, leading signs."""
    if s is None:
        return None
    t = s.strip().strip("$").replace(",", "").replace("\\!", "").replace("\\,", "")
    t = re.sub(r"\\text\{[^}]*\}", "", t).replace("\\%", "").replace("%", "").strip()
    t = re.sub(r"\\left|\\right|\\ ", "", t).strip()
    m = _FRAC_RE.match(t)
    if m:
        a, b = to_number(m.group(1)), to_number(m.group(2))
        return a / b if a is not None and b not in (None, 0) else None
    m = _SCI_RE.match(t.replace(" ", ""))
    if m:
        return float(m.group(1)) * 10 ** int(m.group(2))
    m = _LEAD_NUM_RE.match(t)
    if not m:
        return None
    rest = t[len(m.group(0)):]
    # Accept a trailing UNIT (" K/atm", "\mathrm{J}", "^\circ", "cm") but not more
    # math ("\times 10^{3}", "/2", "x", "^2", "+1") -- that would misread the value.
    if rest and not re.match(r"^\s+[A-Za-z\\]|^\\(mathrm|text|,|;|\s)|^\^\\circ|^°", rest):
        return None
    if re.match(r"^\s*(\\times|\\cdot|x\b|\*)", rest):
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def is_correct(benchmark: str, parsed: str | None, gold: str) -> bool:
    """Exact match after normalization, OR -- when both sides are numbers --
    agreement within NUMERIC_REL_TOL. The exact-only rule (until 2026-09-09)
    graded '87.79' vs gold '+87.8' and '0.5' vs '\\frac{1}{2}' as wrong,
    which hit scibench hardest and hit mathchat (told to answer in decimals)
    harder than debate. See DECISIONS #26."""
    if parsed is None:
        return False
    if normalize(parsed) == normalize(gold):
        return True
    a, b = to_number(parsed), to_number(gold)
    if a is None or b is None:
        return False
    return abs(a - b) <= NUMERIC_REL_TOL * max(abs(b), 1e-12) or (b == 0 and abs(a) < 1e-9)


ROUND0_SYSTEM = (
    "You are one of several independent expert agents solving a problem. Think step by step, "
    "then give your final answer in exactly the format the task specifies."
)
REVISE_SYSTEM = (
    "You are one of several expert agents in a debate. You previously answered the problem; you now "
    "see the other agents' answers from the last round. Reconsider carefully -- keep your answer if "
    "you are confident, change it only if another agent's reasoning is genuinely more convincing. "
    "Think step by step, then give your final answer in exactly the format the task specifies."
)


@dataclass
class DebateConfig:
    n_agents: int = 3
    n_rounds: int = 2
    max_new_tokens: int = 768
    temperature: float = 0.7
    top_p: float = 0.95


class QwenDebate:
    def __init__(self, model, tokenizer, cfg: DebateConfig):
        self.model, self.tok, self.cfg = model, tokenizer, cfg

    @torch.no_grad()
    def _chat_batch(self, systems: list[str], users: list[str]) -> list[str]:
        self.tok.padding_side = "left"
        texts = [
            self.tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": u}],
                                         tokenize=False, add_generation_prompt=True)
            for s, u in zip(systems, users, strict=True)
        ]
        inputs = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        out = self.model.generate(
            **inputs, do_sample=self.cfg.temperature > 0, temperature=self.cfg.temperature or None,
            top_p=self.cfg.top_p, max_new_tokens=self.cfg.max_new_tokens,
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
        )
        return self.tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def run(self, task: dict, trace_id: str) -> dict:
        n, query, bench = self.cfg.n_agents, task["query"], task["benchmark"]
        agent_ids = [f"agent_{i+1}" for i in range(n)]
        trajectory = [{"kind": "user", "content": query}]

        # Round 0: independent answers (agents cannot see each other).
        outputs = self._chat_batch([ROUND0_SYSTEM] * n, [query] * n)
        trajectory.append({"kind": "round", "round": 0,
                           "turns": [{"agent_id": a, "output": o} for a, o in zip(agent_ids, outputs)]})

        # Rounds >= 1: each agent sees all peers' previous-round outputs.
        for r in range(1, self.cfg.n_rounds):
            prev = trajectory[-1]["turns"]
            users = []
            for i, a in enumerate(agent_ids):
                peers = "\n\n".join(f"[{t['agent_id']}]:\n{t['output']}" for t in prev if t["agent_id"] != a)
                users.append(f"Problem:\n{query}\n\nYour previous answer:\n{prev[i]['output']}\n\n"
                             f"Other agents' answers (last round):\n{peers}\n\nGive your (possibly revised) final answer.")
            outputs = self._chat_batch([REVISE_SYSTEM] * n, users)
            trajectory.append({"kind": "round", "round": r,
                               "turns": [{"agent_id": a, "output": o} for a, o in zip(agent_ids, outputs)]})

        # Final answer: majority vote over last round's parsed answers; ties -> agent_1.
        last = trajectory[-1]["turns"]
        parsed = [parse_task_answer(bench, t["output"]) for t in last]
        votes = Counter(normalize(p) for p in parsed if p is not None)
        if votes:
            top_norm, _ = votes.most_common(1)[0]
            final = next(p for p in parsed if p is not None and normalize(p) == top_norm)
        else:
            final = parsed[0]
        trajectory.append({"kind": "final_answer", "content": final if final is not None else ""})

        return {
            "id": trace_id, "framework": "debate", "benchmark": bench, "modality": "text",
            "agents": [{"id": a, "role": "debater"} for a in agent_ids],
            "task": {"query": query, "answer": task["answer"]},
            "trajectory": trajectory,
            "ground_truth": None,
            "final_answer": final,
            "final_correct": is_correct(bench, final, task["answer"]),
            "per_agent_correct_last_round": [is_correct(bench, p, task["answer"]) for p in parsed],
        }


def to_release_row(release: dict) -> dict:
    """Serialize like data/text.jsonl rows (task/trajectory/ground_truth as JSON
    strings) so whowhen_eval.run.release_from_row + the renderers accept it."""
    return {
        "id": release["id"], "framework": release["framework"], "benchmark": release["benchmark"],
        "task": json.dumps(release["task"]), "trajectory": json.dumps(release["trajectory"]),
        "ground_truth": json.dumps(release["ground_truth"] or {}),
        "extras": json.dumps({"agents": release["agents"]}),
    }
