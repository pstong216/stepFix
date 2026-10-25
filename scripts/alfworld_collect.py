"""Collect student rollouts in ALFWorld (single agent, admissible-command ReAct policy).
Writes one JSON line per episode: {id, game_file, split, goal, initial_obs, steps:[{obs, admissible, action, logprobs?}], won, n_steps}
and, for failures, a Who&When-style alfagent release row (*.release.jsonl) for Model A.
    python scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --split train --n_games 2500 --budget 30 --temperature 1.0 --out reports/alfworld/student3b_train.jsonl
"""
import argparse, json, os, random, signal, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("ERROR_ATTR_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from error_attr.env.alfworld_env import game_files, make_env, task_goal
from error_attr.policy.alfworld_agent import ALFWorldHistory, InteractionStep, LLMPolicy


def to_release(ep):
    traj = [{"kind": "user", "content": ep["initial_obs"]}]
    for i, st in enumerate(ep["steps"]):
        traj.append({"kind": "action", "step_number": i + 1, "reasoning": None, "action": st["action"],
                     "observation": st["next_obs"], "is_final_answer": False})
    return {"id": ep["id"], "framework": "alfagent", "benchmark": "alfworld", "modality": "text",
            "task": {"query": ep["goal"], "answer": ""}, "trajectory": traj, "ground_truth": {},
            "extras": {"agents": [{"id": "agent", "role": "agent"}]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="train"); ap.add_argument("--n_games", type=int, default=0); ap.add_argument("--game_offset", type=int, default=0)
    ap.add_argument("--budget", type=int, default=30); ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--history", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True); ap.add_argument("--save_logprobs", action="store_true")
    ap.add_argument("--tag", default="student3b"); ap.add_argument("--episode_timeout", type=int, default=600, help="seconds; a game that exceeds it is skipped (some pick_cool/Lettuce games hang the engine)")
    ap.add_argument("--max_episodes_per_run", type=int, default=0, help="exit after this many episodes (host RAM grows ~45 MB per env; rerun to resume)")
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done_games = set()
    if out.exists():
        for l in open(out):
            try: done_games.add(json.loads(l)["game_file"])
            except Exception: pass
    games = game_files(a.split)[a.game_offset:]
    if a.n_games: games = games[:a.n_games]
    todo = [g for g in games if g not in done_games]
    print(f"split {a.split}: {len(games)} games, {len(done_games)} done, {len(todo)} to do", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model); tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload().eval()
    policy = LLMPolicy(model, tok)
    rel_out = open(str(out).replace(".jsonl", ".release.jsonl"), "a")
    t0 = time.time(); n_won = 0; n_done = 0
    with open(out, "a") as f:
        def _alarm(signum, frame): raise TimeoutError("episode timeout")
        signal.signal(signal.SIGALRM, _alarm)
        n_skipped = 0
        for gi, gf in enumerate(todo):
            env = make_env(gf, a.budget + 1)
            signal.alarm(a.episode_timeout)
            try:
                obs, info = env.reset()
                init = obs[0]; goal = task_goal(init)
                hist = ALFWorldHistory(goal=goal)
                hist.steps.append(InteractionStep(observation=init, admissible_commands=list(info["admissible_commands"][0])))
                steps = []; won = False
                for t in range(a.budget):
                    cur = hist.steps[-1]
                    if a.save_logprobs:
                        lp = policy.score_admissible_commands_cached(hist); cmds = list(lp)
                        if a.temperature <= 0: action = max(cmds, key=lambda c: lp[c])
                        else:
                            logits = torch.tensor([lp[c] for c in cmds]) / a.temperature
                            action = cmds[torch.multinomial(torch.softmax(logits, 0), 1).item()]
                    else:
                        lp = None; action = policy.act(hist, temperature=a.temperature)
                    cur.action = action
                    obs, reward, dones, info = env.step([action])
                    nxt = obs[0]
                    steps.append({"obs": cur.observation, "admissible": cur.admissible_commands, "action": action, "next_obs": nxt,
                                  **({"logprobs": lp} if lp else {})})
                    if dones[0]:
                        won = bool(info["won"][0]); break
                    hist.steps.append(InteractionStep(observation=nxt, admissible_commands=list(info["admissible_commands"][0])))
                ep = {"id": f"alf_{a.tag}_{a.split}_{Path(gf).parent.name}_{gi + a.game_offset:05d}", "game_file": gf, "split": a.split, "goal": goal,
                      "initial_obs": init, "steps": steps, "won": won, "n_steps": len(steps), "budget": a.budget, "temperature": a.temperature, "model": a.model}
                f.write(json.dumps(ep) + "\n"); f.flush()
                if not won:
                    rel_out.write(json.dumps(to_release(ep)) + "\n"); rel_out.flush()
                n_won += won; n_done += 1
                if a.max_episodes_per_run and n_done >= a.max_episodes_per_run and gi + 1 < len(todo):
                    print(f"CHUNK {n_done} episodes, success {n_won}/{n_done}; remaining {len(todo) - gi - 1}", flush=True); rel_out.close(); return
                if n_done % 10 == 0:
                    print(f"[{n_done}/{len(todo)}] success {n_won}/{n_done} = {n_won / n_done:.3f}  {(time.time() - t0) / n_done:.1f}s/episode", flush=True)
            except TimeoutError:
                n_skipped += 1; print(f"TIMEOUT: skipping {gf}", flush=True)
            finally:
                signal.alarm(0)
                try: env.close()
                except Exception: pass
    print(f"DONE {n_done} episodes, success {n_won}/{n_done}, skipped {n_skipped}", flush=True)


if __name__ == "__main__":
    main()
