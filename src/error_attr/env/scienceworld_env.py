"""ScienceWorld interface in the same trajectory schema as ALFWorld (id, goal, initial_obs, steps[obs, action, next_obs],
won), plus the environment's own 0-100 score per step (used as a generic subgoal signal, never a planner)."""
from scienceworld import ScienceWorldEnv


def make(step_limit=150):
    return ScienceWorldEnv("", envStepLimit=step_limit)


def reset(env, task, var, gold=False):
    env.load(task, var, generateGoldPath=gold)
    obs, info = env.reset()
    return obs, info


def step(env, action):
    obs, reward, done, info = env.step(action)
    return obs, info["score"], done, info


def valid_actions(env):
    return env.get_valid_action_object_combinations()
