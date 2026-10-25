"""WebShop text-environment server (runs in the `webshop` conda env, py3.8).  The student / teacher code lives in the error_attr env
(py3.11), so the environment is exposed over HTTP as JSON.  One WebAgentTextEnv per worker id (envs are cheap: 0.6 s init).
  POST /reset   {worker, session}          -> {obs, instruction, actions}
  POST /step    {worker, action}           -> {obs, reward, done, actions}
  GET  /goal?session=N                     -> goal dict (asin, attributes, goal_options, query, price_upper, ...)
  POST /oracle  {worker, session, budget}  -> scripted expert trajectory using the goal's hidden asin/options (for SFT demos)
Actions: 'search[keywords]' or 'click[value]'; `actions` = {'has_search_bar': bool, 'clickables': [...]} (click values, lower-case).
"""
import argparse, json, random, sys, threading
from flask import Flask, request, jsonify
sys.path.insert(0, "/scratch.global/pan00389/error-attribution/webshop/WebShop")
from web_agent_site.envs import WebAgentTextEnv

app = Flask(__name__); ENVS = {}; LOCK = threading.Lock(); NUM_PRODUCTS = 1000


GOAL_SEED = 0   # Oct 8: WebShop samples each goal's price cap with the global `random` BEFORE its own random.seed(233) shuffle, so every new env
                # (every worker id, every server restart) got different price caps for the same session. Seeding right before each env is built
                # makes the goals identical across workers and restarts (servers started before this fix used random caps).


def env_for(w):
    with LOCK:
        if w not in ENVS:
            if GOAL_SEED is not None: random.seed(GOAL_SEED)
            ENVS[w] = WebAgentTextEnv(observation_mode="text", num_products=NUM_PRODUCTS)
        return ENVS[w]


def page(env):   # plain page text (the step() return concatenates history with [SEP]; we keep only the current page)
    return env.observation


@app.post("/reset")
def reset():
    d = request.json; env = env_for(d["worker"]); env.reset(session=int(d["session"]))
    return jsonify(obs=page(env), instruction=env.get_instruction_text(), actions=env.get_available_actions())


@app.post("/step")
def step():
    d = request.json; env = env_for(d["worker"]); _, r, done, _ = env.step(d["action"])
    return jsonify(obs=page(env), reward=float(r or 0.0), done=bool(done), actions=env.get_available_actions())


@app.get("/goal")
def goal():
    env = env_for("goal"); g = env.server.goals[int(request.args["session"])]
    return jsonify({k: v for k, v in g.items() if k != "weight"})


def oracle_run(env, session, budget=30):
    """search the goal query, page through results until the target asin shows up, open it, pick the goal options, buy."""
    g = env.server.goals[session]; env.reset(session=session); traj = []; asin = g["asin"].lower(); instr = env.get_instruction_text()

    def do(action):
        obs = page(env); _, r, done, _ = env.step(action); traj.append({"obs": obs, "action": action, "next_obs": page(env), "reward": float(r or 0.0)})
        return done
    queries = [g["query"]] + [" ".join(g["name"].split()[:6])]
    found = False
    for q in queries:
        do(f"search[{q}]")
        for _ in range(5):
            cl = env.get_available_actions()["clickables"]
            if asin in cl: found = True; break
            if "next >" in cl and len(traj) < budget - 6: do("click[next >]")
            else: break
        if found: break
        do("click[back to search]")
    if not found: return traj, 0.0, False, instr
    do(f"click[{asin}]")
    for opt in (g.get("goal_options") or {}).values():
        cl = env.get_available_actions()["clickables"]
        if str(opt).lower() in cl: do(f"click[{str(opt).lower()}]")
    done = do("click[buy now]")
    return traj, traj[-1]["reward"], done, instr


@app.post("/oracle")
def oracle():
    d = request.json; env = env_for(d["worker"]); traj, r, done, instr = oracle_run(env, int(d["session"]), int(d.get("budget", 30)))
    return jsonify(steps=traj, reward=r, done=done, instruction=instr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=38300); ap.add_argument("--num_products", type=int, default=1000); a = ap.parse_args()
    NUM_PRODUCTS = a.num_products; env_for("goal"); print("WEBSHOP SERVER READY", flush=True)
    app.run(host="0.0.0.0", port=a.port, threaded=True)
