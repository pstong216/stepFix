# StepFix

Code for **StepFix**: self-evolving language agents that learn from failures through environment-verified single-step corrections.

For each failed episode, a frozen failure attributor localizes a critical step, the current policy proposes alternative actions in hindsight, and each alternative is executed for one environment step and scored by observable goal progress. The best-verified corrections, together with KL-anchored replay of the agent's own successful decisions, update the policy each round. Experiments use ALFWorld and ScienceWorld with Qwen2.5-3B/7B-Instruct.

## Layout
- `src/` — environment wrappers and shared utilities
- `scripts/` — data collection, attribution, teacher proposals, one-step verification, training, evaluation, baselines, and analysis
- `slurm/` — cluster job templates
- `configs/` — configuration files
- `docs/` — experiment notes and status logs

Model weights and experiment outputs are not stored in this repository.
