"""Expert (handcoded ALFWorld agent) trajectories on train games, for warm-start SFT of the student.
CPU only. Writes the same episode schema as alfworld_collect.py.
    python scripts/alfworld_expert_collect.py --n_games 1200 --out reports/alfworld/expert_train.jsonl
"""
import argparse, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("ERROR_ATTR_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import textworld, textworld.gym
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredExpert, AlfredExpertType, AlfredInfos
from error_attr.env.alfworld_env import game_files, task_goal


def make_expert_env(game_file, max_steps, expert_type):
    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile", "expert_plan"])
    env_id = textworld.gym.register_games([game_file], request_infos, batch_size=1, asynchronous=False, max_episode_steps=max_steps,
                                          wrappers=[AlfredDemangler(shuffle=False), AlfredInfos, AlfredExpert(expert_type=expert_type)])
    return textworld.gym.make(env_id)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n_games", type=int, default=1200); ap.add_argument("--game_offset", type=int, default=0)
    ap.add_argument("--budget", type=int, default=30); ap.add_argument("--out", required=True); ap.add_argument("--split", default="train")
    ap.add_argument("--expert", default="handcoded", choices=["handcoded", "planner"])
    a = ap.parse_args()
    et = AlfredExpertType.HANDCODED if a.expert == "handcoded" else AlfredExpertType.PLANNER
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists(): done = {json.loads(l)["game_file"] for l in open(out)}
    games = game_files(a.split)[a.game_offset:][: a.n_games]; todo = [g for g in games if g not in done]
    print(f"{len(games)} games, {len(done)} done, {len(todo)} to do", flush=True)
    t0 = time.time(); n_won = 0
    with open(out, "a") as f:
        for gi, gf in enumerate(todo):
            env = make_expert_env(gf, a.budget + 1, et)
            try:
                obs, info = env.reset(); init = obs[0]; goal = task_goal(init); steps = []; won = False
                for t in range(a.budget):
                    plan = info["extra.expert_plan"][0]; action = plan[0] if isinstance(plan, list) else plan
                    if not action: break
                    adm = list(info["admissible_commands"][0])
                    obs, reward, dones, info = env.step([action])
                    steps.append({"obs": steps[-1]["next_obs"] if steps else init, "admissible": adm, "action": action, "next_obs": obs[0]})
                    if dones[0]: won = bool(info["won"][0]); break
                f.write(json.dumps({"id": f"alf_expert_{a.split}_{Path(gf).parent.name}_{gi + a.game_offset:05d}", "game_file": gf, "split": a.split, "goal": goal,
                                    "initial_obs": init, "steps": steps, "won": won, "n_steps": len(steps), "budget": a.budget, "model": f"expert_{a.expert}"}) + "\n"); f.flush()
                n_won += won
                if (gi + 1) % 50 == 0: print(f"[{gi + 1}/{len(todo)}] expert success {n_won}/{gi + 1}  {(time.time() - t0) / (gi + 1):.1f}s/ep", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"error on {gf}: {e!r}", flush=True)
            finally:
                env.close()
    print(f"DONE expert success {n_won}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
