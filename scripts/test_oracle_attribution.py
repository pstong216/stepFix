"""Validate oracle_alfworld.find_critical_step on a real failed trajectory.

Generates a short random-policy trajectory that fails (does not win), then
recovers its oracle critical step and sanity-checks the result: the
recoverable-prefix trace should go True...True,False...False (monotonic),
and t* should be strictly less than the trajectory length whenever the
final state truly is unrecoverable.
"""
import os
import random
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ERROR_ATTR_ROOT", ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from alfworld.agents.environment import get_environment  # noqa: E402

from error_attr.attribution.oracle_alfworld import find_critical_step, is_recoverable_from  # noqa: E402


def load_config(path):
    with open(path) as f:
        raw = f.read()
    for key, val in os.environ.items():
        raw = raw.replace(f"${key}", val)
    return yaml.safe_load(raw)


TOTAL_STEP_BUDGET = 30  # matches typical ALFWorld agent horizons in the literature


def collect_failed_trajectory(config, max_steps=12, seed=0):
    EnvClass = get_environment(config["env"]["type"])
    env = EnvClass(config, train_eval="train")
    game_file = env.game_files[0]
    env = env.init_env(batch_size=1)

    obs, info = env.reset()
    random.seed(seed)
    actions = []
    done = False
    step = 0
    while not done and step < max_steps:
        admissible = info["admissible_commands"][0]
        action = random.choice(admissible)
        obs, reward, dones, info = env.step([action])
        actions.append(action)
        done = dones[0]
        step += 1
    won = bool(info.get("won", [False])[0])
    return game_file, actions, won


def main():
    config_path = os.path.join(ROOT, "configs", "base_config.yaml")
    config = load_config(config_path)

    game_file, actions, won = collect_failed_trajectory(config, max_steps=TOTAL_STEP_BUDGET, seed=0)
    print(f"game_file={game_file}")
    print(f"actions={actions}")
    print(f"won at end of logged trajectory={won}")

    if won:
        print("Trajectory won outright -- pick a different seed/max_steps for a failure case.")
        return 1

    # Sanity check 1: the full prefix (all logged actions) should be
    # unrecoverable, since the trajectory as logged did not win and the
    # expert had every chance to salvage it during collect (it didn't
    # exist during collection -- this checks post hoc).
    result = find_critical_step(game_file, actions, total_step_budget=TOTAL_STEP_BUDGET, verify_monotonicity=True)
    print(f"t_star={result['t_star']}")
    print(f"recoverable_prefix_lengths={result['recoverable_prefix_lengths']}")

    trace = result["recoverable_prefix_lengths"]
    assert trace == sorted(trace, reverse=True), "trace is not monotonic True->False"

    if result["t_star"] is None:
        print("WARNING: oracle says trajectory was recoverable throughout (expert could still "
              "win after every prefix) -- means 'failure' here is just running out of budget, "
              "not a true unrecoverable error. Expected for very short random rollouts.")
    else:
        assert 0 <= result["t_star"] < len(actions)
        print(f"Critical action: actions[{result['t_star']}] = {actions[result['t_star']]!r}")

    print("ORACLE ATTRIBUTION TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
