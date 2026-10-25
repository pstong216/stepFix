"""Phase 1 smoke test: confirm the ALFWorld text environment loads and runs
one episode end-to-end with a random policy. No model, no training -- this
only verifies the environment substrate before anything else is built on it.
"""
import os
import random
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ERROR_ATTR_ROOT", ROOT)

from alfworld.agents.environment import get_environment  # noqa: E402


def load_config(path):
    with open(path) as f:
        raw = f.read()
    for key, val in os.environ.items():
        raw = raw.replace(f"${key}", val)
    return yaml.safe_load(raw)


def main():
    config_path = os.path.join(ROOT, "configs", "base_config.yaml")
    config = load_config(config_path)

    env_type = config["env"]["type"]
    EnvClass = get_environment(env_type)
    env = EnvClass(config, train_eval="train")
    env = env.init_env(batch_size=1)

    obs, info = env.reset()
    print("=== TASK GOAL ===")
    print(obs[0][:500])
    print()

    random.seed(0)
    done = False
    step = 0
    max_steps = 15
    total_reward = 0.0
    while not done and step < max_steps:
        admissible = info["admissible_commands"][0]
        action = random.choice(admissible)
        obs, reward, dones, info = env.step([action])
        done = dones[0]
        total_reward += reward[0]
        step += 1
        print(f"[step {step}] action={action!r} reward={reward[0]} done={done}")

    print()
    print(f"Episode finished after {step} steps. total_reward={total_reward} won={info.get('won', [None])[0]}")
    print("SMOKE TEST PASSED" if step > 0 else "SMOKE TEST FAILED")


if __name__ == "__main__":
    sys.exit(main())
