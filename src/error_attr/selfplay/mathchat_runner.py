"""Qwen-driven MathChat (AutoGen 2-agent: assistant <-> user_proxy with
Python execution) emitting the Who&When Pro `mathchat` release schema, so
self-generated trajectories flow through the official renderer -> Model A ->
feedback-conditions pipeline unchanged (see debate_runner.py for the
rationale; this adds a second, structurally different collaboration
topology: a single reasoning agent plus a tool-executing proxy, sequential
turns, instead of parallel peer debate).

Schema (matches whowhen_eval/render/mathchat.py):
    trajectory = [
      {"kind": "user", "content": <query>},
      {"agent_id": "user_proxy", "role": "user_proxy", "round": 0, "position": 0, "output": <framing prompt>},
      {"agent_id": "assistant",  "role": "assistant",  "round": 0, "position": 1, "output": ...},   # step 1
      {"agent_id": "user_proxy", "role": "user_proxy", "round": 1, "position": 0, "output": "(Result of the code run:\n...\n)"},  # step 2
      {"agent_id": "assistant",  "role": "assistant",  "round": 1, "position": 1, "output": ...},   # step 3
      ...
      {"kind": "final_answer", "content": <boxed answer>},
    ]

Protocol = AutoGen MathUserProxyAgent semantics, reproduced from the
official traces: the proxy's round-0 turn is the verbatim MathChat framing
prompt; after each assistant turn, if the turn contains a ```python block
the proxy executes it and replies "(Result of the code run:\n...\n)";
else if the turn contains \\boxed{} the dialog terminates; else the proxy
says "Continue. Please keep solving the problem until you need to query.
(If you get to the answer, put it in \\boxed{}.)". Final answer = the last
\\boxed{} in the assistant's last turn (AutoGen's rule).

Runs many dialogs in lockstep so each assistant turn is one batched
generate() call.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from error_attr.selfplay.debate_runner import is_correct, parse_task_answer
from error_attr.selfplay.pyexec import extract_python_blocks, run_python

FRAMING = """Let's use Python to solve a math problem.

Query requirements:
You should always use the 'print' function for the output and use decimal numbers for final numerical answers.
You can use packages like sympy to help you.
You must follow the formats below to write your code:
```python
# your code
```

First state the key idea to solve the problem. You may choose from three ways to solve the problem:
Case 1: If the problem can be solved with Python code directly, please write a program to solve it. You can enumerate all possible arrangements if needed.
Case 2: If the problem is mostly reasoning, you can solve it by yourself directly.
Case 3: If the problem cannot be handled in the above two ways, please follow this process:
1. Solve the problem step by step (do not over-divide the steps).
2. Take out any queries that can be asked through Python (for example, any calculations or equations that can be calculated).
3. Wait for me to give the results.
4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

After all the queries are run and you get the answer, put the answer in \\boxed{}.

Problem:

{query}"""

CONTINUE = "Continue. Please keep solving the problem until you need to query. (If you get to the answer, put it in \\boxed{}.)"
ASSISTANT_SYSTEM = "You are a helpful AI assistant."
AGENTS = [{"id": "assistant", "role": "assistant"}, {"id": "user_proxy", "role": "user_proxy"}]


@dataclass
class MathChatConfig:
    max_assistant_turns: int = 5
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    exec_timeout_s: int = 30


def proxy_reply(assistant_output: str, exec_timeout_s: int = 30) -> tuple[str | None, str | None]:
    """AutoGen MathUserProxy rule. Returns (proxy_message or None if terminated, final_answer or None)."""
    blocks = extract_python_blocks(assistant_output)
    if blocks:
        results = []
        for code in blocks:  # AutoGen runs every block, concatenating results
            out, _ok = run_python(code, timeout_s=exec_timeout_s)
            results.append(out)
        return "(Result of the code run:\n" + "\n".join(results) + "\n)", None
    boxed = parse_task_answer("math", assistant_output)  # \boxed{} convention for all non-GAIA tasks
    if boxed is not None:
        return None, boxed
    return CONTINUE, None


class QwenMathChat:
    def __init__(self, model, tokenizer, cfg: MathChatConfig):
        self.model, self.tok, self.cfg = model, tokenizer, cfg

    @torch.no_grad()
    def _chat_batch(self, message_lists: list[list[dict]]) -> list[str]:
        self.tok.padding_side = "left"
        texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in message_lists]
        inputs = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        out = self.model.generate(
            **inputs, do_sample=self.cfg.temperature > 0, temperature=self.cfg.temperature or None,
            top_p=self.cfg.top_p, max_new_tokens=self.cfg.max_new_tokens,
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
        )
        return self.tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def run_many(self, tasks: list[dict], trace_ids: list[str]) -> list[dict]:
        """Lockstep dialogs: one batched generate per assistant turn across all still-active tasks."""
        n = len(tasks)
        framing = [FRAMING.replace("{query}", t["query"]) for t in tasks]
        msgs = [[{"role": "system", "content": ASSISTANT_SYSTEM}, {"role": "user", "content": f}] for f in framing]
        trajs = [[{"kind": "user", "content": t["query"]},
                  {"agent_id": "user_proxy", "role": "user_proxy", "round": 0, "position": 0, "output": f}]
                 for t, f in zip(tasks, framing)]
        final: list[str | None] = [None] * n
        active = list(range(n))
        for r in range(self.cfg.max_assistant_turns):
            if not active:
                break
            outs = self._chat_batch([msgs[i] for i in active])
            still = []
            for i, out in zip(active, outs):
                trajs[i].append({"agent_id": "assistant", "role": "assistant", "round": r, "position": 1, "output": out})
                msgs[i].append({"role": "assistant", "content": out})
                proxy_msg, boxed = proxy_reply(out, self.cfg.exec_timeout_s)
                if proxy_msg is None:
                    final[i] = boxed
                    continue
                if r + 1 < self.cfg.max_assistant_turns:  # proxy speaks only if the assistant gets another turn
                    trajs[i].append({"agent_id": "user_proxy", "role": "user_proxy", "round": r + 1, "position": 0, "output": proxy_msg})
                    msgs[i].append({"role": "user", "content": proxy_msg})
                    still.append(i)
            active = still
        releases = []
        for i, (t, tid) in enumerate(zip(tasks, trace_ids)):
            if final[i] is None:  # ran out of turns: AutoGen would take the last boxed anywhere, else nothing
                last_assistant = [e for e in trajs[i] if e.get("agent_id") == "assistant"]
                final[i] = parse_task_answer("math", last_assistant[-1]["output"]) if last_assistant else None
            trajs[i].append({"kind": "final_answer", "content": final[i] if final[i] is not None else ""})
            releases.append({
                "id": tid, "framework": "mathchat", "benchmark": t["benchmark"], "modality": "text",
                "agents": AGENTS, "task": {"query": t["query"], "answer": t["answer"]},
                "trajectory": trajs[i], "ground_truth": None,
                "final_answer": final[i], "final_correct": is_correct(t["benchmark"], final[i], t["answer"]),
            })
        return releases

    # ---- revision protocol for the feedback chain (framework-native "extra round") ----
    def revise_many(self, releases: list[dict], feedbacks: list[str]) -> list[str | None]:
        """The user_proxy delivers the coordinator feedback as the next message of the
        SAME dialog; the assistant gets up to 2 more turns (one may be a code query).
        Returns the revised final answers (None if the assistant never boxes one)."""
        msgs, out_final = [], [None] * len(releases)
        for rel, fb in zip(releases, feedbacks):
            m = [{"role": "system", "content": ASSISTANT_SYSTEM}]
            for e in rel["trajectory"]:
                if e.get("agent_id") == "user_proxy":
                    m.append({"role": "user", "content": e["output"]})
                elif e.get("agent_id") == "assistant":
                    m.append({"role": "assistant", "content": e["output"]})
            m.append({"role": "user", "content": f"## Coordinator Feedback\n{fb}\n\nContinue solving. Put your corrected final answer in \\boxed{{}}."})
            msgs.append(m)
        active = list(range(len(releases)))
        for _turn in range(2):
            if not active:
                break
            outs = self._chat_batch([msgs[i] for i in active])
            still = []
            for i, out in zip(active, outs):
                msgs[i].append({"role": "assistant", "content": out})
                proxy_msg, boxed = proxy_reply(out, self.cfg.exec_timeout_s)
                if proxy_msg is None:
                    out_final[i] = boxed
                else:
                    msgs[i].append({"role": "user", "content": proxy_msg})
                    out_final[i] = parse_task_answer("math", out)  # provisional, in case no further turn
                    still.append(i)
            active = still
        return out_final
