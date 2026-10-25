"""Expert (gold-path) demonstrations for ScienceWorld in the ALFWorld trajectory schema. Replays each gold sequence,
recording observation, action, next observation and the environment score per step; keeps only paths that reach 100.
Per task at most --per_task variations from the requested split (balanced across the 30 task types)."""
import argparse, json, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from error_attr.env.scienceworld_env import make, reset, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "dev", "test"])
    ap.add_argument("--per_task", type=int, default=30)
    ap.add_argument("--step_limit", type=int, default=250)
    ap.add_argument("--out", default="reports/sciworld/gold_train.jsonl")
    a = ap.parse_args(); rng = random.Random(0); out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    env = make(a.step_limit); done_ids = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    f = open(out, "a"); t0 = time.time(); n_ok = n_bad = 0
    for task in env.get_task_names():
        env.load(task, 0)
        vs = {"train": env.get_variations_train, "dev": env.get_variations_dev, "test": env.get_variations_test}[a.split]()
        vs = list(vs); rng.shuffle(vs)
        for v in vs[: a.per_task]:
            i = f"sw_{task}_{v}"
            if i in done_ids: continue
            obs, info = reset(env, task, v, gold=True); gold = env.get_gold_action_sequence()
            steps = []; cur = obs; score = 0; dn = False
            for act in gold:
                nxt, score, dn, info = step(env, act)
                steps.append({"obs": cur, "action": act, "next_obs": nxt, "score": score}); cur = nxt
                if dn: break
            won = score >= 100
            n_ok += won; n_bad += not won
            if won:
                f.write(json.dumps({"id": i, "task": task, "variation": v, "split": a.split, "goal": info["taskDesc"], "initial_obs": obs,
                                    "steps": steps, "won": True, "n_steps": len(steps)}) + "\n"); f.flush()
        print(f"{task}: ok {n_ok} bad {n_bad} {time.time() - t0:.0f}s", flush=True)
    print("DONE gold", n_ok, n_bad, flush=True)


if __name__ == "__main__":
    main()
