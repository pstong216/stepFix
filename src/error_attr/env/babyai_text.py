"""Text interface to BabyAI (minigrid) for LLM agents, plus the expert (BabyAIBot) used for demonstrations and as a
measurement ruler (steps the expert still needs from a state; never used as a training input in oracle-free variants)."""
import copy
import warnings

import gymnasium as gym
import minigrid  # noqa: F401  (registers BabyAI envs)
from minigrid.core.constants import COLOR_NAMES, IDX_TO_OBJECT
from minigrid.utils.baby_ai_bot import BabyAIBot

warnings.filterwarnings("ignore")
ACTIONS = ["turn left", "turn right", "move forward", "pick up", "drop", "toggle"]
IDX_TO_COLOR = {i: c for i, c in enumerate(COLOR_NAMES)}
DOOR_STATE = {0: "open", 1: "closed", 2: "locked"}


def make(level, seed):
    env = gym.make(level)
    obs, _ = env.reset(seed=seed)
    return env, obs


def _where(fwd, lat):
    parts = []
    if fwd: parts.append(f"{fwd} step{'s' if fwd > 1 else ''} forward")
    if lat: parts.append(f"{abs(lat)} step{'s' if abs(lat) > 1 else ''} {'left' if lat < 0 else 'right'}")
    return ", ".join(parts) if parts else "right here"


def describe(env, obs):
    """Egocentric 7x7 view -> text. Agent sits at view (3, 6) facing 'forward' (decreasing y)."""
    img = obs["image"]; lines = []
    for j in range(6, -1, -1):
        for i in range(7):
            t = IDX_TO_OBJECT[int(img[i, j, 0])]
            if t in ("unseen", "empty", "wall", "floor", "agent"): continue
            color = IDX_TO_COLOR[int(img[i, j, 1])]
            fwd, lat = 6 - j, i - 3
            name = f"a {DOOR_STATE[int(img[i, j, 2])]} {color} door" if t == "door" else f"a {color} {t}"
            lines.append(f"{name} {_where(fwd, lat)}")
    front = IDX_TO_OBJECT[int(img[3, 5, 0])]
    walls = []
    if front == "wall": walls.append("a wall right in front of you")
    if IDX_TO_OBJECT[int(img[2, 6, 0])] == "wall": walls.append("a wall on your left")
    if IDX_TO_OBJECT[int(img[4, 6, 0])] == "wall": walls.append("a wall on your right")
    carrying = env.unwrapped.carrying
    held = f"You are carrying a {carrying.color} {carrying.type}." if carrying else "You are carrying nothing."
    seen = ("You see " + "; ".join(lines) + ".") if lines else "You see no objects."
    return " ".join([seen] + ([("There is " + ", ".join(walls) + ".")] if walls else []) + [held])


def expert_steps_to_go(env, cap=300):
    """Steps the expert bot needs to finish from the current state (on a deep copy), and whether it succeeds."""
    e = copy.deepcopy(env)
    bot = BabyAIBot(e.unwrapped); n = 0
    try:
        while n < cap:
            a = bot.replan(); _, r, term, trunc, _ = e.step(a); n += 1
            if term or trunc: return n, r > 0
    except Exception:
        return cap, False
    return cap, False


def expert_action(env):
    """The expert's next action from the current state (bot is rebuilt so it plans from this exact state)."""
    e = copy.deepcopy(env)
    try:
        return ACTIONS[BabyAIBot(e.unwrapped).replan()]
    except Exception:
        return None
