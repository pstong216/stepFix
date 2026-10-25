"""Approximate per-process GPU wall-clock (process-hours) per StepFix round, from the logs' own progress counters.
For the ARR compute-budget paragraph only (never compared across methods)."""
import glob, json, os, re
ROOT = "/users/8/pan00389/multi-agent/error-attribution"; SF = ROOT + "/reports/sciworld/stepfix"
OUT = os.path.dirname(os.path.abspath(__file__)) + "/parts"
PR = re.compile(r"^\[(\d+)/(\d+)\].*?(\d+)s\s*$")
LOAD = 60   # model-load overhead per attempt (s), assumed


def log_seconds(path):
    segs = re.split(r"^### attempt.*$", open(path, errors="ignore").read(), flags=re.M); tot = 0
    for s in segs:
        last = None
        for l in s.splitlines():
            m = PR.match(l.strip())
            if m: last = tuple(map(int, m.groups()))
        if last is None: continue
        k, n, sec = last; done = "DONE" in s
        tot += (sec * n / max(k, 1) if done else sec) + LOAD
    return tot


def stage(pattern): return sum(log_seconds(p) for p in glob.glob(pattern))


def train_seconds(d):
    tot = 0; prev = 0; seg = 0
    for l in open(f"{d}/model/train_log.jsonl"):
        e = json.loads(l)["elapsed_s"]
        if e < prev: tot += prev
        prev = e
    return tot + prev + LOAD


def modelA_seconds(d):
    return sum(max(0, os.path.getmtime(f"{d}/modelA_pred_s{i}.jsonl") - os.path.getmtime(f"{d}/rel_s{i}.jsonl")) for i in range(6))


R = {}
for name, col, ma, te, tr, ev in [("R1", "x1/col", "x1", "z1", "w1", "w1/eval"), ("R2", "w2/col", "w2", "w2", "w2k", "w2k/eval"), ("R3", "wk3/col", "wk3", "wk3", "wk3", "wk3/eval")]:
    d = {"collection": stage(f"{SF}/{col}/collect_s*.log"), "modelA_upper": modelA_seconds(f"{SF}/{ma}"), "teacher_verify": stage(f"{SF}/{te}/teacher_s*.log"),
         "train": train_seconds(f"{SF}/{tr}"), "eval848": stage(f"{SF}/{ev}/eval_*_w*.log")}
    d = {k: round(v / 3600, 2) for k, v in d.items()}; d["total_excl_eval"] = round(sum(v for k, v in d.items() if k != "eval848"), 2); R[name] = d
    print(name, d)
json.dump(R, open(f"{OUT}/walltime_stepfix.json", "w"), indent=1)
