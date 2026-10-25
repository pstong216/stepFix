"""Oracle failure attribution for ALFWorld.

Ground truth for "which step caused the failure" does not exist natively in
ALFWorld the way it does in Who&When Pro (no injected-failure construction).
Instead we build it directly from the environment's own dynamics, using the
same handcoded expert ALFWorld already ships (`alfworld.agents.expert
.HandCodedTWAgent`) as a certifying planner:

    t* = the index of the first action in the trajectory after which the
         PDDL-planner expert can no longer complete the task from the
         resulting state, within a fixed remaining-step budget.

Concretely: replay the trajectory's action prefix of length p in a fresh
instance of the same game, then hand control to the expert and see if it
still wins. The largest p for which this succeeds, plus one, is the
critical step -- the action at that index is what made the game
unrecoverable. This is a genuine causal/counterfactual label (it is
verified by re-solving the game from that state), not a heuristic guess,
so it is a legitimate "oracle" for the Experiment 0 feasibility check
(Research Plan Sec. 14, Sec. 25).

Expert choice: ALFWorld ships two `expert_type`s. `HANDCODED` follows a
fixed script derived from the game's original solution trajectory and does
NOT replan cleanly when resumed from an off-script state -- empirically
this produced NON-MONOTONIC recoverability traces (a later, more-exhausted
prefix reported recoverable after an earlier one reported unrecoverable),
which is a false-negative artifact of the script, not a real property of
the game. `PLANNER` asks TextWorld's PDDL solver for a fresh shortest plan
from the CURRENT state on every step, so it replans correctly regardless of
how the prefix was reached. We use PLANNER as the oracle's certifying
solver for exactly this reason.

Caveats (tracked as ablation targets, not swept under the rug):
  - The PDDL planner still assumes remaining budget is what determines
    solvability; ties or near-budget cases are sensitive to the exact
    total_step_budget chosen, which should match the horizon the policy
    is actually trained/evaluated under (see `total_step_budget` below).
  - Recoverability is still verified per-prefix (no monotonicity is
    assumed by default at scale); `verify_monotonicity=True` raises if a
    later prefix is recoverable after an earlier one was not, so any
    residual non-monotonicity under the planner is caught rather than
    silently producing a wrong t*.
"""
from __future__ import annotations

import textworld
import textworld.gym

from alfworld.agents.environment.alfred_tw_env import (
    AlfredDemangler,
    AlfredExpert,
    AlfredExpertType,
    AlfredInfos,
)


def _make_single_env(game_file: str, max_episode_steps: int, expert_type: str = AlfredExpertType.PLANNER):
    demangler = AlfredDemangler(shuffle=False)
    wrappers = [demangler, AlfredInfos, AlfredExpert(expert_type=expert_type)]
    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile", "expert_plan"])
    env_id = textworld.gym.register_games(
        [game_file],
        request_infos,
        batch_size=1,
        asynchronous=False,
        max_episode_steps=max_episode_steps,
        wrappers=wrappers,
    )
    return textworld.gym.make(env_id)


def _expert_next_action(info: dict) -> str:
    plan = info["extra.expert_plan"][0]
    return plan[0] if isinstance(plan, list) else plan


def is_recoverable_from(game_file: str, action_prefix: list[str], total_step_budget: int) -> bool:
    """Replay action_prefix in a fresh episode, then follow the handcoded
    expert for the STEPS REMAINING under the same total step budget the
    policy is actually trained/evaluated with. Returns True iff the expert
    wins within that remaining horizon.

    Tying the horizon to total_step_budget (rather than a fixed generous
    allowance) matters: "recoverable in principle, given unlimited steps"
    is a much weaker and less useful oracle than "recoverable within the
    same budget the policy operates under." The latter is what actually
    determines whether the trajectory could still have succeeded.
    """
    remaining = total_step_budget - len(action_prefix)
    if remaining < 0:
        return False
    env = _make_single_env(game_file, max_episode_steps=total_step_budget + 1)
    try:
        obs, info = env.reset()
        done = False
        for a in action_prefix:
            obs, reward, dones, info = env.step([a])
            done = dones[0]
            if done:
                return bool(info["won"][0])

        steps = 0
        while not done and steps < remaining:
            expert_action = _expert_next_action(info)
            obs, reward, dones, info = env.step([expert_action])
            done = dones[0]
            steps += 1
        return bool(info["won"][0]) if done else False
    finally:
        env.close()


def find_critical_step(
    game_file: str,
    actions: list[str],
    total_step_budget: int,
    verify_monotonicity: bool = False,
) -> dict:
    """For a completed FAILED trajectory (`actions` did not win), find the
    oracle critical step t*, under a fixed total_step_budget (the same
    horizon the trajectory itself was collected under).

    Returns a dict:
        t_star: int index (0-based, into `actions`) of the critical action,
                or None if the trajectory was recoverable throughout (i.e.
                the failure was not yet locked in by the last logged step --
                can happen if the episode was cut off by a step budget
                rather than truly failing).
        recoverable_prefix_lengths: list[bool], recoverable_prefix_lengths[p]
                is whether the game was still winnable after the first p
                actions within the remaining budget (only populated in full
                when verify_monotonicity=True or when the forward scan
                needs the full trace anyway).
    """
    trace = []
    last_recoverable = -1
    broke = False
    for p in range(0, len(actions) + 1):
        rec = is_recoverable_from(game_file, actions[:p], total_step_budget=total_step_budget)
        trace.append(rec)
        if rec:
            last_recoverable = p
            if broke and verify_monotonicity:
                raise AssertionError(
                    f"Non-monotonic recoverability in {game_file}: prefix {p} recoverable "
                    f"after an earlier prefix was not. trace={trace}"
                )
        else:
            broke = True
            if not verify_monotonicity:
                break

    if last_recoverable == len(actions):
        return {"t_star": None, "recoverable_prefix_lengths": trace}

    return {"t_star": last_recoverable, "recoverable_prefix_lengths": trace}


def progress_trace(game_file: str, actions: list[str], total_step_budget: int) -> dict:
    """Second oracle notion -- the ORIGIN of a failure rather than its point of no return.

    Replay the trajectory once in a PLANNER-expert env and record, after every prefix, the length of the
    planner's shortest completion (distance to goal). The first wasted action is the first index p whose
    action does not reduce that distance (plan_len[p+1] >= plan_len[p]); for a wandering policy that is the
    onset of the detour/loop, typically many steps before the point of no return t*.
    Returns {plan_len: [L_0, L_1, ..., L_n], t_first_waste: int|None, t_first_regress: int|None} where
    t_first_regress is the first action that INCREASES the distance (a strictly harmful action).
    """
    env = _make_single_env(game_file, max_episode_steps=total_step_budget + 1)
    try:
        obs, info = env.reset()
        plan = info["extra.expert_plan"][0]
        lens = [len(plan) if isinstance(plan, list) else (1 if plan else 0)]
        for a in actions:
            obs, reward, dones, info = env.step([a])
            if dones[0]:
                lens.append(0 if info["won"][0] else lens[-1]); break
            plan = info["extra.expert_plan"][0]
            lens.append(len(plan) if isinstance(plan, list) else (1 if plan else 0))
        waste = next((p for p in range(len(lens) - 1) if lens[p + 1] >= lens[p]), None)
        regress = next((p for p in range(len(lens) - 1) if lens[p + 1] > lens[p]), None)
        return {"plan_len": lens, "t_first_waste": waste, "t_first_regress": regress}
    finally:
        env.close()
