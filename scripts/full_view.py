"""Oct 9 (user, autonomy note): FULL-information views for the hindsight self-teacher (ALFWorld). The standard teacher saw only the last
10 prefix steps (policy prompt), at most 25 future steps, and future observations cut to 160 characters. These wrappers remove all three
caps: the complete prefix, every remaining step with its complete observation, and the episode outcome. Used by
scripts/self_teacher_informed.py --full_view (default off, so existing chains are unchanged)."""


class FullPrefix:
    """Policy history rendered with EVERY prefix step (no max_history_steps window)."""
    def __init__(self, h):
        self.steps = h.steps; self._h = h
    def render_prompt(self, max_history_steps=10):
        return self._h.render_prompt(10 ** 6)


class FullHindsight:
    """Full prefix + everything that happened after step t in the failed episode (all steps, full observations)."""
    def __init__(self, h, ep, t):
        self.steps = h.steps; self._h = h
        fut = ep["steps"][t:]
        self._block = "\n".join(f"- step {t + i}: {s['action']} -> {s['next_obs']}" for i, s in enumerate(fut))
        self._n = len(ep["steps"])
    def render_prompt(self, max_history_steps=10):
        p = self._h.render_prompt(10 ** 6)
        note = ("Hindsight: this attempt FAILED. After this point the agent did the following "
                f"(episode length {self._n} steps, budget exhausted; complete record):\n{self._block}\n"
                "Using everything above, choose the best action for the CURRENT step.\n")
        return p.replace("Admissible actions:", note + "Admissible actions:", 1)
