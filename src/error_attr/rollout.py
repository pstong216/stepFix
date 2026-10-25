"""On-policy rollout collection (Research Plan Sec. 10, Step 1): the current
student pi_theta generates tau ~ pi_theta. Trajectories are logged in a
plain JSON-serializable format so oracle labeling (Sec. 8/14) and OPD
training (Sec. 10, Steps 2-5) can both consume them without re-running the
environment or the policy.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

from alfworld.agents.environment import get_environment

from error_attr.policy.alfworld_agent import ALFWorldHistory, InteractionStep, LLMPolicy


@dataclass
class LoggedStep:
    observation: str
    admissible_commands: list[str]
    action: str
    reward: float


@dataclass
class Trajectory:
    game_file: str
    goal: str
    steps: list[LoggedStep] = field(default_factory=list)
    won: bool = False
    total_step_budget: int = 0

    def to_json(self) -> dict:
        return asdict(self)


def collect_one_episode(
    env_class_config: dict,
    policy: LLMPolicy,
    game_index: int,
    total_step_budget: int,
    temperature: float = 0.0,
    train_eval: str = "train",
) -> Trajectory:
    EnvClass = get_environment(env_class_config["env"]["type"])
    env_wrapper = EnvClass(env_class_config, train_eval=train_eval)
    game_file = env_wrapper.game_files[game_index]
    env = env_wrapper.init_env(batch_size=1)

    obs, info = env.reset()
    goal = obs[0]
    history = ALFWorldHistory(goal=goal)

    traj = Trajectory(game_file=game_file, goal=goal, total_step_budget=total_step_budget)

    done = False
    step = 0
    current_obs = obs[0]
    while not done and step < total_step_budget:
        admissible = info["admissible_commands"][0]
        history.steps.append(InteractionStep(observation=current_obs, admissible_commands=admissible))

        action = policy.act(history, temperature=temperature)
        history.steps[-1].action = action

        obs, reward, dones, info = env.step([action])
        done = dones[0]
        traj.steps.append(
            LoggedStep(observation=current_obs, admissible_commands=admissible, action=action, reward=float(reward[0]))
        )
        current_obs = obs[0]
        step += 1

    traj.won = bool(info.get("won", [False])[0])
    env.close()
    return traj


def save_trajectories(trajectories: list[Trajectory], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for traj in trajectories:
            f.write(json.dumps(traj.to_json()) + "\n")


def load_trajectories(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]
