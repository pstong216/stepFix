"""Per-game ALFWorld text environments (deterministic game order, resumable collection)."""
from __future__ import annotations
import os
import yaml
import textworld
import textworld.gym
from alfworld.agents.environment import get_environment
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("ERROR_ATTR_ROOT", ROOT)
SPLIT_MAP = {"train": "train", "valid_seen": "eval_in_distribution", "valid_unseen": "eval_out_of_distribution"}


def load_config(path=None):
    path = path or os.path.join(ROOT, "configs", "base_config.yaml")
    with open(path) as f:
        raw = f.read()
    for key, val in os.environ.items():
        raw = raw.replace(f"${key}", val)
    return yaml.safe_load(raw)


def game_files(split="train", config=None, shuffle_seed=0):
    """Sorted game list, then deterministically shuffled (seed 0) so that any contiguous slice mixes the six task
    types (the sorted order groups games by task type). shuffle_seed=None returns the sorted order."""
    import random
    config = config or load_config()
    EnvClass = get_environment(config["env"]["type"])
    env = EnvClass(config, train_eval=SPLIT_MAP.get(split, split))
    gs = sorted(env.game_files)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(gs)
    return gs


def make_env(game_file: str, max_episode_steps: int):
    """One TextWorld gym env for a single game, no expert wrapper (fast)."""
    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
    env_id = textworld.gym.register_games([game_file], request_infos, batch_size=1, asynchronous=False,
                                          max_episode_steps=max_episode_steps, wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
    return textworld.gym.make(env_id)


def task_goal(initial_obs: str) -> str:
    for line in initial_obs.split("\n"):
        if "Your task is to:" in line:
            return line.split("Your task is to:")[1].strip()
    return initial_obs.strip().split("\n")[-1]


class NoisyEnv:
    """Stochastic ALFWorld: with probability p the intended command is replaced by a uniformly random OTHER admissible
    command (an action slip). The underlying game stays deterministic, so a logged trajectory is replayed exactly by
    feeding its EXECUTED commands through step_exact(); noise is applied only by step(). info["executed"] holds the
    command that actually ran, info["slipped"] whether it differs from the intended one."""

    def __init__(self, env, p, seed=0):
        import random
        self.env, self.p, self.rng, self.adm = env, p, random.Random(seed), []

    def reset(self):
        obs, info = self.env.reset(); self.adm = list(info["admissible_commands"][0]); return obs, info

    def step_exact(self, cmds):
        obs, rw, dn, info = self.env.step(cmds); self.adm = list(info["admissible_commands"][0])
        info["executed"], info["slipped"] = cmds[0], False; return obs, rw, dn, info

    def step(self, cmds):
        a = cmds[0]; others = [c for c in self.adm if c != a]
        ex = self.rng.choice(others) if (self.p > 0 and others and self.rng.random() < self.p) else a
        obs, rw, dn, info = self.step_exact([ex]); info["executed"], info["slipped"] = ex, ex != a
        return obs, rw, dn, info

    def close(self):
        self.env.close()


def make_noisy_env(game_file: str, max_episode_steps: int, p: float, seed: int = 0):
    return NoisyEnv(make_env(game_file, max_episode_steps), p, seed)
