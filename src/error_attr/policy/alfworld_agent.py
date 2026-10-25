"""LLM policy for ALFWorld: ReAct-style prompting over admissible commands.

Actions are constrained to the environment's own `admissible_commands` list
at every step (rather than free-form generation + parsing), so the policy
never wastes a step on an invalid action. This keeps rollouts clean for the
oracle-labeling step downstream (Sec. 8/10 of the research plan) -- the
trajectory's action sequence is always a sequence of real, replayable
environment commands.

Selection is done by scoring each admissible command's log-probability
under the model conditioned on the running prompt (a fixed instruction +
truncated interaction history), and picking the argmax (or sampling from
the resulting distribution when `temperature > 0`). This also happens to
be exactly the per-step distribution q_S(a) / q_T(a) the OPD loss (Sec. 10
Step 5) needs -- so this module doubles as both the rollout-time policy and
the distillation-time student/teacher scorer, called on the same logged
h_t during training.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

SYSTEM_PROMPT = (
    "You are an embodied household agent completing a task in a simulated "
    "environment. You will be given a task goal and a history of "
    "observations and actions. Choose the single best next action from the "
    "list of admissible actions. Respond with the exact text of one "
    "admissible action and nothing else."
)


@dataclass
class InteractionStep:
    observation: str
    admissible_commands: list[str]
    action: str | None = None


@dataclass
class ALFWorldHistory:
    goal: str
    steps: list[InteractionStep] = field(default_factory=list)

    def render_prompt(self, max_history_steps: int = 10) -> str:
        lines = [f"Task goal: {self.goal}", ""]
        recent = self.steps[-max_history_steps:]
        for i, step in enumerate(recent):
            lines.append(f"Observation: {step.observation}")
            if step.action is not None:
                lines.append(f"Action: {step.action}")
        lines.append("Admissible actions:")
        if self.steps:
            for a in self.steps[-1].admissible_commands:
                lines.append(f"- {a}")
        lines.append("")
        lines.append("Next action:")
        return "\n".join(lines)


class LLMPolicy:
    """Wraps a causal LM + tokenizer to select among admissible commands."""

    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device

    def _build_chat_prompt(self, history: ALFWorldHistory) -> str:
        user_content = history.render_prompt()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def score_admissible_commands(self, history: ALFWorldHistory, batch_size: int = 16) -> dict[str, float]:
        """Returns {command: log-probability} for every admissible command
        at the current step, under this policy's distribution. This is
        q(.|h_t) for OPD -- shared code path for rollout and training.

        All commands share the same prompt prefix, so they're scored in
        batches with a single forward pass per batch (right-padded, with an
        attention mask) rather than one full forward pass per command --
        matters once there are a dozen+ admissible commands at 7B scale.
        """
        prompt = self._build_chat_prompt(history)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        prompt_len = prompt_ids.shape[0]
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        commands = history.steps[-1].admissible_commands
        cmd_id_lists = [
            self.tokenizer(cmd, add_special_tokens=False).input_ids for cmd in commands
        ]

        logprobs = {}
        for start in range(0, len(commands), batch_size):
            chunk_cmds = commands[start : start + batch_size]
            chunk_ids = cmd_id_lists[start : start + batch_size]
            max_cmd_len = max(len(ids) for ids in chunk_ids)
            total_len = prompt_len + max_cmd_len

            input_ids = torch.full((len(chunk_ids), total_len), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(chunk_ids), total_len), dtype=torch.long)
            for i, ids in enumerate(chunk_ids):
                seq = torch.cat([prompt_ids, torch.tensor(ids, dtype=torch.long)])
                input_ids[i, : seq.shape[0]] = seq
                attention_mask[i, : seq.shape[0]] = 1

            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            token_logprobs = torch.log_softmax(out.logits[:, prompt_len - 1 : -1, :], dim=-1)

            for i, (cmd, ids) in enumerate(zip(chunk_cmds, chunk_ids)):
                ids_t = torch.tensor(ids, dtype=torch.long, device=self.device)
                chosen = token_logprobs[i, : len(ids), :].gather(1, ids_t.unsqueeze(1)).squeeze(1)
                logprobs[cmd] = chosen.sum().item()
        return logprobs

    @torch.no_grad()
    def score_admissible_commands_cached(self, history: ALFWorldHistory, batch_size: int = 32) -> dict[str, float]:
        """Same output as score_admissible_commands, much cheaper: the shared prompt is run once with a KV cache,
        the cache is replicated across a batch of commands and only the command tails are scored (2 forwards/step)."""
        import copy
        prompt = self._build_chat_prompt(history)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        plen = prompt_ids.shape[1]
        out = self.model(input_ids=prompt_ids, use_cache=True)
        cache = out.past_key_values
        first_lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        cmds = history.steps[-1].admissible_commands
        id_lists = [self.tokenizer(c, add_special_tokens=False).input_ids for c in cmds]
        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        logprobs = {}
        for start in range(0, len(cmds), batch_size):
            chunk = id_lists[start: start + batch_size]; n = len(chunk)
            tails = [ids[:-1] for ids in chunk]; maxlen = max(len(t) for t in tails)
            tl = None
            if maxlen > 0:
                c = copy.deepcopy(cache); c.batch_repeat_interleave(n)
                inp = torch.full((n, maxlen), pad, dtype=torch.long); att = torch.zeros((n, plen + maxlen), dtype=torch.long); att[:, :plen] = 1
                for i, t in enumerate(tails):
                    if t: inp[i, : len(t)] = torch.tensor(t); att[i, plen: plen + len(t)] = 1
                o = self.model(input_ids=inp.to(self.device), attention_mask=att.to(self.device), past_key_values=c, use_cache=True)
                tl = torch.log_softmax(o.logits.float(), dim=-1)
            for i, ids in enumerate(chunk):
                lp = first_lp[ids[0]].item()
                if len(ids) > 1:
                    tg = torch.tensor(ids[1:], device=self.device)
                    lp += tl[i, : len(tg)].gather(1, tg.unsqueeze(1)).sum().item()
                logprobs[cmds[start + i]] = lp
        return logprobs

    @torch.no_grad()
    def score_admissible_commands_cached_seq(self, history: ALFWorldHistory) -> dict[str, float]:
        """Same output as score_admissible_commands, ~10x cheaper: the shared prompt is run once with a KV
        cache, then each command's few tokens are scored against that cache (cache cropped back after each)."""
        prompt = self._build_chat_prompt(history)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        plen = prompt_ids.shape[1]
        out = self.model(input_ids=prompt_ids, use_cache=True)
        cache = out.past_key_values
        first_lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)  # distribution over the command's first token
        logprobs = {}
        for cmd in history.steps[-1].admissible_commands:
            ids = self.tokenizer(cmd, add_special_tokens=False).input_ids
            lp = first_lp[ids[0]].item()
            if len(ids) > 1:
                o = self.model(input_ids=torch.tensor([ids[:-1]], device=self.device), past_key_values=cache, use_cache=True)
                tl = torch.log_softmax(o.logits[0].float(), dim=-1)
                lp += tl.gather(1, torch.tensor(ids[1:], device=self.device).unsqueeze(1)).sum().item()
                cache.crop(plen)
            logprobs[cmd] = lp
        return logprobs

    def act(self, history: ALFWorldHistory, temperature: float = 0.0, cached: bool = True) -> str:
        logprobs = self.score_admissible_commands_cached(history) if cached else self.score_admissible_commands(history)
        commands = list(logprobs.keys())
        if temperature <= 0:
            return max(commands, key=lambda c: logprobs[c])
        logits = torch.tensor([logprobs[c] for c in commands]) / temperature
        probs = torch.softmax(logits, dim=0)
        idx = torch.multinomial(probs, 1).item()
        return commands[idx]
