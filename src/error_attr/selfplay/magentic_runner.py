"""Qwen-driven Magentic-One-style orchestration emitting the Who&When Pro
`magentic-one` release schema (whowhen_eval/render/magentic.py). Third
collaboration topology for the self-play study: HIERARCHICAL -- a central
orchestrator plans, routes each round to one specialist, and alone states
the final answer. (debate = parallel peers; mathchat = reasoner + tool proxy.)

Specialists: `coder` (writes a ```python block) and `computerterminal`
(executes the most recent code block; deterministic, not an LLM turn), i.e.
the two Magentic-One specialists that make sense for self-contained
reasoning tasks. `websurfer` / `filesurfer` are omitted -- no web/file
access here -- and the agents list says so.

Schema:
    trajectory = [
      {"kind": "user", "content": <task>},
      {"kind": "agent", "agent": "orchestrator", "role": "Orchestrator", "round": r, "position": 0,
       "output": <instruction>, "next_speaker": <specialist>, "phase": ["live"]},
      {"kind": "agent", "agent": "coder", "role": "Coder", "round": r, "position": 1,
       "output": <analysis + code>, "tool_actions": [{"type": "code", "name": "code_block", "args": {...}}]},
      ... {"kind": "agent", "agent": "orchestrator", ..., "phase": ["live", "final_answer"], "output": <answer text>}
    ]
No `final_answer` entry: the renderer surfaces the orchestrator turn whose
phase contains "final_answer", exactly like the official traces.

Orchestrator turn format (parsed leniently):
    NEXT_SPEAKER: coder | computerterminal | FINAL
    INSTRUCTION: <what the specialist should do>      (or, for FINAL, the answer in the task's format)
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch

from error_attr.selfplay.debate_runner import is_correct, parse_task_answer
from error_attr.selfplay.pyexec import extract_python_blocks, run_python

AGENTS = [{"id": "orchestrator", "role": "Orchestrator"}, {"id": "coder", "role": "Coder"},
          {"id": "computerterminal", "role": "ComputerTerminal"}]

ORCH_SYSTEM = (
    "You are the Orchestrator of a small team solving a task. You do not solve it yourself; you plan, "
    "delegate one step at a time, check progress, and state the final answer. Your specialists:\n"
    "- coder: reasons about the problem and writes Python code (it cannot run code).\n"
    "- computerterminal: runs the most recent Python code block the coder wrote and reports the output.\n"
    "Each turn, reply in EXACTLY this format:\n"
    "NEXT_SPEAKER: <coder | computerterminal | FINAL>\n"
    "INSTRUCTION: <one clear instruction for that specialist>\n"
    "Choose FINAL only when the team has enough to answer; then INSTRUCTION must contain the final answer "
    "in exactly the format the task specifies (e.g. \\boxed{...} or 'FINAL ANSWER: ...'). Be decisive: "
    "do not repeat an instruction that already produced a result."
)
CODER_SYSTEM = (
    "You are the Coder on a team. The Orchestrator gives you an instruction. Reason briefly, then write ONE "
    "```python block that prints what is needed (use sympy/numpy if helpful; always print results). You "
    "cannot run code yourself; the computerterminal will run it. Do not state a final answer for the task."
)


@dataclass
class MagenticConfig:
    max_rounds: int = 6
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    exec_timeout_s: int = 30


_NEXT_RE = re.compile(r"NEXT_SPEAKER:\s*\**\s*(coder|computerterminal|final)", re.IGNORECASE)
_INSTR_RE = re.compile(r"INSTRUCTION:\s*(.*)", re.DOTALL | re.IGNORECASE)


def parse_orchestrator(text: str) -> tuple[str, str]:
    m = _NEXT_RE.search(text)
    nxt = m.group(1).lower() if m else ("final" if parse_task_answer("math", text) or "FINAL ANSWER" in text else "coder")
    mi = _INSTR_RE.search(text)
    instr = mi.group(1).strip() if mi else text.strip()
    instr = instr.lstrip("*: ").strip()  # tolerate markdown-bold labels ("**INSTRUCTION:** ...")
    return nxt, instr


def _transcript(traj: list[dict]) -> str:
    parts = []
    for e in traj:
        if e.get("kind") != "agent":
            continue
        parts.append(f"[{e['agent']} | round {e['round']}]\n{e['output']}")
    return "\n\n".join(parts) if parts else "(nothing yet)"


class QwenMagentic:
    def __init__(self, model, tokenizer, cfg: MagenticConfig):
        self.model, self.tok, self.cfg = model, tokenizer, cfg

    @torch.no_grad()
    def _chat_batch(self, systems: list[str], users: list[str]) -> list[str]:
        self.tok.padding_side = "left"
        texts = [self.tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": u}],
                                              tokenize=False, add_generation_prompt=True)
                 for s, u in zip(systems, users, strict=True)]
        inputs = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        out = self.model.generate(
            **inputs, do_sample=self.cfg.temperature > 0, temperature=self.cfg.temperature or None,
            top_p=self.cfg.top_p, max_new_tokens=self.cfg.max_new_tokens,
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
        )
        return self.tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def _orchestrate(self, tasks, trajs, idx, extra: list[str] | None = None) -> list[str]:
        users = []
        for k, i in enumerate(idx):
            u = (f"# Task\n{tasks[i]['query']}\n\n# Team transcript so far\n{_transcript(trajs[i])}\n\n"
                 + (extra[k] + "\n\n" if extra else "")
                 + "Decide the next step (NEXT_SPEAKER / INSTRUCTION).")
            users.append(u)
        return self._chat_batch([ORCH_SYSTEM] * len(idx), users)

    def _code(self, tasks, trajs, idx, instructions) -> list[str]:
        users = [f"# Task\n{tasks[i]['query']}\n\n# Team transcript so far\n{_transcript(trajs[i])}\n\n"
                 f"# Orchestrator's instruction to you\n{ins}" for i, ins in zip(idx, instructions)]
        return self._chat_batch([CODER_SYSTEM] * len(idx), users)

    @staticmethod
    def _terminal(traj: list[dict], timeout_s: int) -> tuple[str, list[dict]]:
        code_turns = [e for e in traj if e.get("agent") == "coder"]
        blocks = extract_python_blocks(code_turns[-1]["output"]) if code_turns else []
        if not blocks:
            return "No Python code block found in the coder's most recent message; nothing to execute.", []
        out, ok = run_python(blocks[-1], timeout_s=timeout_s)
        return f"Execution {'succeeded' if ok else 'failed'}. Output:\n{out}", [
            {"type": "shell", "name": "execute_python", "args": {"code": blocks[-1]}, "exit_ok": ok}]

    def _step_round(self, tasks, trajs, active, r, extra=None) -> tuple[list[int], dict[int, str]]:
        """One orchestrator turn + one specialist turn for every active task. Returns (still_active, finals)."""
        finals: dict[int, str] = {}
        orch_outs = self._orchestrate(tasks, trajs, active, extra)
        to_code, code_instr, still = [], [], []
        for i, out in zip(active, orch_outs):
            nxt, instr = parse_orchestrator(out)
            entry = {"kind": "agent", "agent": "orchestrator", "role": "Orchestrator", "round": r, "position": 0,
                     "output": instr, "next_speaker": nxt, "phase": ["live"]}
            if nxt == "final":
                entry["phase"] = ["live", "final_answer"]
                trajs[i].append(entry)
                finals[i] = instr
                continue
            trajs[i].append(entry)
            if nxt == "coder":
                to_code.append(i); code_instr.append(instr)
            else:
                text, actions = self._terminal(trajs[i], self.cfg.exec_timeout_s)
                trajs[i].append({"kind": "agent", "agent": "computerterminal", "role": "ComputerTerminal", "round": r,
                                 "position": 1, "output": text, "tool_actions": actions, "phase": ["live"]})
            still.append(i)
        if to_code:
            for i, out in zip(to_code, self._code(tasks, trajs, to_code, code_instr)):
                blocks = extract_python_blocks(out)
                trajs[i].append({"kind": "agent", "agent": "coder", "role": "Coder", "round": r, "position": 1,
                                 "output": out, "phase": ["live"],
                                 "tool_actions": [{"type": "code", "name": "code_block", "args": {"language": "python", "code": b}} for b in blocks]})
        return still, finals

    def run_many(self, tasks: list[dict], trace_ids: list[str]) -> list[dict]:
        n = len(tasks)
        trajs = [[{"kind": "user", "content": t["query"]}] for t in tasks]
        final: list[str | None] = [None] * n
        active = list(range(n))
        for r in range(self.cfg.max_rounds):
            if not active:
                break
            active, finals = self._step_round(tasks, trajs, active, r)
            for i, txt in finals.items():
                final[i] = parse_task_answer(tasks[i]["benchmark"], txt)
        # Out of rounds: force a final orchestrator turn (still a genuine orchestrator decision, flagged in phase).
        if active:
            outs = self._orchestrate(tasks, trajs, active,
                                     ["The round budget is exhausted. You MUST answer now: NEXT_SPEAKER: FINAL."] * len(active))
            for i, out in zip(active, outs):
                _nxt, instr = parse_orchestrator(out)
                trajs[i].append({"kind": "agent", "agent": "orchestrator", "role": "Orchestrator", "round": self.cfg.max_rounds,
                                 "position": 0, "output": instr, "next_speaker": "final", "phase": ["live", "final_answer", "forced"]})
                final[i] = parse_task_answer(tasks[i]["benchmark"], instr)
        return [{
            "id": tid, "framework": "magentic-one", "benchmark": t["benchmark"], "modality": "text",
            "agents": AGENTS, "task": {"query": t["query"], "answer": t["answer"]},
            "trajectory": trajs[i], "ground_truth": None,
            "final_answer": final[i], "final_correct": is_correct(t["benchmark"], final[i], t["answer"]),
        } for i, (t, tid) in enumerate(zip(tasks, trace_ids))]

    # ---- revision protocol for the feedback chain ----
    def revise_many(self, releases: list[dict], feedbacks: list[str], extra_rounds: int = 2) -> list[str | None]:
        """The orchestrator receives the coordinator feedback with the full transcript and
        gets up to `extra_rounds` more rounds (it may re-delegate) before it must answer."""
        tasks = [{"query": r["task"]["query"], "benchmark": r["benchmark"]} for r in releases]
        trajs = [list(r["trajectory"]) for r in releases]
        start = max((e.get("round", 0) for tr in trajs for e in tr if e.get("kind") == "agent"), default=0) + 1
        out_final: list[str | None] = [None] * len(releases)
        active = list(range(len(releases)))
        extra = [f"## Coordinator Feedback\n{fb}\n\nYour team's previous final answer was judged INCORRECT. "
                 "Decide how to proceed." for fb in feedbacks]
        for k in range(extra_rounds):
            if not active:
                break
            active, finals = self._step_round(tasks, trajs, active, start + k, [extra[i] for i in active])
            for i, txt in finals.items():
                out_final[i] = parse_task_answer(tasks[i]["benchmark"], txt)
        if active:
            outs = self._orchestrate(tasks, trajs, active,
                                     [extra[i] + "\nThe round budget is exhausted. You MUST answer now: NEXT_SPEAKER: FINAL." for i in active])
            for i, out in zip(active, outs):
                _n, instr = parse_orchestrator(out)
                out_final[i] = parse_task_answer(tasks[i]["benchmark"], instr)
        return out_final
