import sys, json, time; sys.path.insert(0, "src")
from error_attr.env.webshop_env import WebShopEnv
env = WebShopEnv("oracle_test"); ok = 0; rs = []; n = 0; t0 = time.time()
with open("reports/webshop/oracle_test.jsonl", "w") as f:
    for s in range(0, 500):
        r = env.oracle(s, 30); n += 1; ok += r["reward"] >= 1.0; rs.append(r["reward"])
        f.write(json.dumps({"session": s, "reward": r["reward"], "n_steps": len(r["steps"])}) + "\n")
print(f"DONE oracle test split: success {ok}/{n} = {ok/n:.3f}, mean reward {sum(rs)/n:.3f}, {time.time()-t0:.0f}s", flush=True)
