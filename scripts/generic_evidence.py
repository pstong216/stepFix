"""ONE generic one-step evidence template, shared by every environment (user decision Oct 5 23:00).  Four terms, each filled with
the environment's native signal; no task parsing, no action vocabulary:
  information  n_new    = number of new content tokens in the resulting observation (relative to everything seen so far)
  relevance    overlap  = some new token occurs in the task description
  progress     progress = the environment's own reward/score increased, or it acknowledged a state change it did not show before
  regress      regress  = the environment's own reward/score decreased, or the episode ended without success
  null         no_info  = observation unchanged, an observation already seen earlier in the episode (a revisit/undo), or no new
                          token and no progress
  score = 2*relevance + 2*progress - 4*regress + min(n_new,5)/5 - 2*no_info
Per environment only `observe` changes: how to strip boilerplate (e.g. WebShop repeats the instruction on every page) and what the
native score is (ALFWorld: none -> 0; WebShop: reward on purchase; ScienceWorld: task score).
"""
import re
STOP = set("a an the of to in on at is are was be you your it its this that with and or for from into onto as by see also called there here now not no yes find me lower than dollars price put some two look under arrive closed open nothing then pick move using".split())


def toks(s):
    """content units: object instances ('soapbar 4', 'b09hkzs7lv') when numbered, else words; stopwords and short words dropped."""
    s = s.lower(); inst = set(re.findall(r"\b([a-z][a-z\-]+ \d+)\b", s)); heads = {x.split()[0] for x in inst}
    words = {w for w in re.findall(r"[a-z][a-z0-9\-]+", s) if w not in STOP and len(w) > 2 and w not in heads}
    return {x for x in inst if x.split()[0] not in STOP} | words


def body(obs, env):
    if env == "webshop":
        parts = obs.split("[SEP]"); return "[SEP]".join(parts[2:]) if len(parts) > 2 and parts[0].strip().lower().startswith("instruction") else obs
    return obs


def generic_evidence(goal, history_obs, cur_obs, obs_after, env="alfworld", score_before=0.0, score_after=0.0, done=False, success=False):
    """history_obs: list of observations seen before the current one (initial obs included)."""
    b = body(obs_after, env).strip(); seen = [body(o, env).strip() for o in history_obs] + [body(cur_obs, env).strip()]
    known = set().union(*[toks(o) for o in seen]) if seen else set()
    new = toks(b) - known; gw = toks(goal.replace("Instruction:", ""))
    overlap = any(x.split()[0] in gw for x in new)   # (a coverage-gain variant was tried Oct 6 01:25 and reverted: it did not fix WebShop and would change running ALFWorld chains)
    # environment-acknowledged state change ("You <verb> the X ...") counts as progress only if the changed object X is named in the
    # task description (generic: no verb list, no task parser; the object is read off the acknowledgement itself)
    # a second state change on the same object instance is not new progress (undo / take-back loops)
    # the SAME acknowledged change (same verb, same object instance) seen before is not new progress -- catches take-back loops
    # (pick up X, put X, pick up X again) while a chain of different changes on one object (pick up, clean, move) still counts
    m = re.match(r"you ((?!see|arrive|are )[a-z]+(?: [a-z]+)?) the ([a-z]+)( \d+)?", b.lower())
    repeat = bool(m) and any(re.match(rf"you {re.escape(m.group(1))} the {re.escape(m.group(2) + (m.group(3) or ''))}\b", o.lower()) for o in seen)
    progress = (score_after > score_before) or (done and success) or (bool(m) and m.group(2) in gw and not repeat)
    regress = (score_after < score_before) or (done and not success)
    no_info = b == seen[-1] or b in seen[:-1] or b.lower().startswith("nothing happens") or (len(new) == 0 and not progress)
    return {"n_new": len(new), "overlap": overlap, "progress": bool(progress), "regress": bool(regress), "no_info": bool(no_info),
            "score": 2.0 * overlap + 2.0 * progress - 4.0 * regress + min(len(new), 5) / 5.0 - 2.0 * no_info}


def alfworld_step(e, t, obs_after):
    hist = [e["initial_obs"]] + [s["obs"] for s in e["steps"][:t]] + [s["next_obs"] for s in e["steps"][:t]]
    return generic_evidence(e["goal"], hist, e["steps"][t]["obs"], obs_after, env="alfworld")
