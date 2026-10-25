"""TextCraft (Prasad et al. 2023, "ADaPT"; packaged as AgentGym agentenv-textcraft 0.0.1, commit 3ef9235) -- a pure-python port of
agentenv_textcraft/{utils,crafting_tree,environment}.py (MIT; identical logic to ADaPT's TextCraft/*.py) with three determinism fixes:
  (1) recipe files are read in sorted order (the original uses os.listdir order, which decides which recipe of a cycle is dropped);
  (2) traverse_recipe_tree works on a copy of the stored recipe list (the original extends the tree's own list in place, so every task
      created changes the recipe sets of later tasks and what craft() returns);
  (3) distractors use a local random.Random(seed) (AgentGym: global random.seed(42); ADaPT: random.seed(task seed)).
Recipe JSONs (Minecraft 1.16.5 data shipped with AgentGym) are only read as data, from RECIPE_DIR.

Tasks: every item with min crafting depth >= 1 (AgentGym's pool), a task = (goal item, distractor seed); id "tc_<goal>_s<seed>".
Splits (by GOAL ITEM, so test goals are never seen in training): see splits().
Our addition for the admissible-scoring policy: admissible() and the oracle solver expert_actions().
    python scripts/textcraft_env.py stats      # task pool, depth distribution, admissible-set sizes, expert lengths / success
"""
import copy, json, math, os, random, re, sys, collections
from dataclasses import dataclass
from typing import List, Dict

RECIPE_DIR = os.environ.get("TEXTCRAFT_RECIPES", "/scratch.global/pan00389/error-attribution/third_party_ro/agentgym_textcraft/repo/agentenv-textcraft/agentenv_textcraft/recipes")
GET_MULTIPLES = (1, 2)                # admissible 'get' counts = k * (a count with which the item appears in a shown recipe)
MAX_TAG_VARIANTS = 4                  # concrete substitutions per craft command (tag ingredients)


# ----------------------------------------------------------------------------- port of agentenv_textcraft.utils
@dataclass(frozen=True)
class ItemTag:
    tag: str = None
    item_id: str = None
    @property
    def name(self): return self.item_id or self.tag


@dataclass
class ItemTagWithCount:
    item_tag: ItemTag
    count: int


@dataclass(frozen=True)
class Recipe:
    input_items: List[ItemTagWithCount]
    output_item: ItemTagWithCount
    @property
    def recipe_str(self):
        s = "craft {} {} using ".format(self.output_item.count, item_id_to_str(self.output_item.item_tag.name))
        for it in self.input_items: s += "{} {}, ".format(it.count, item_id_to_str(it.item_tag.name))
        return s[:-2]


class ActionFailed(Exception):
    pass


def item_id_to_str(item_id): return item_id.replace("minecraft:", "").replace("_", " ")
def str_to_item_id(s): return "minecraft:" + s.replace(" ", "_")


# ----------------------------------------------------------------------------- port of agentenv_textcraft.crafting_tree
class CraftingTree:
    def __init__(self, recipe_dir=RECIPE_DIR):
        self.tag_recipes = {}; self.itemid_recipes: Dict[str, list] = {}; self.tag_set = set(); self.itemid_set = set()
        self.item_id_to_tag = {}; self.transitive_dependencies = {}; self.min_depth = {}
        self._load_recipes(recipe_dir); self.clean_up_recipes()

    def clean_up_recipes(self):
        new_items = set()
        for item, recipes in self.itemid_recipes.items():
            for recipe in recipes:
                for input_item in recipe.input_items:
                    input_tag = input_item.item_tag.tag
                    if input_item.item_tag.item_id is None:
                        assert input_tag is not None
                        item_list = list(self.get_items_with_tags(input_tag)); success = False
                        for item_id in item_list:
                            if item_id in self.itemid_recipes: success = True; break
                        if not success:
                            input_item.item_tag = ItemTag(item_id=input_tag); new_items.add(input_tag)
        for item in new_items:
            self.itemid_set.add(item); self.tag_set.remove(item)

    def _load_recipes(self, recipe_dir):
        for f in sorted(os.listdir(recipe_dir)):                                   # fix (1)
            with open(os.path.join(recipe_dir, f)) as fp: rd = json.load(fp)
            input_items = []
            if rd["type"] == "minecraft:crafting_shaped":
                pattern = rd["pattern"]
                for key, item in rd["key"].items():
                    count = 0
                    if isinstance(item, list): item = item[0]
                    for line in pattern: count += line.count(key)
                    if "item" in item: input_item = ItemTag(item_id=item["item"]); self.itemid_set.add(item["item"])
                    elif "tag" in item: input_item = ItemTag(tag=item["tag"]); self.tag_set.add(item["tag"])
                    else: raise ValueError("Unknown item type")
                    input_items.append(ItemTagWithCount(input_item, count))
            elif rd["type"] == "minecraft:crafting_shapeless":
                idx = {}
                for ing in rd["ingredients"]:
                    if isinstance(ing, list): ing = ing[0]
                    if "item" in ing: name = ing["item"]; input_item = ItemTag(item_id=name); self.itemid_set.add(name)
                    elif "tag" in ing: input_item = ItemTag(tag=ing["tag"]); name = ing["tag"]; self.tag_set.add(ing["tag"])
                    else: raise ValueError("Unknown item type")
                    if name not in idx: idx[name] = len(input_items); input_items.append(ItemTagWithCount(input_item, 1))
                    else: c = input_items[idx[name]].count; input_items[idx[name]] = ItemTagWithCount(input_item, c + 1)
            else:
                continue
            res = rd["result"]
            if isinstance(res, str): out_id, out_n = res, 1
            elif "item" in res: out_id, out_n = res["item"], res.get("count") or 1
            else: raise ValueError("Unknown item type")
            self.itemid_set.add(out_id)
            if len(input_items) == 1 and input_items[0].item_tag.name.endswith("_block"): continue
            out_tag = None
            if "group" in rd:
                out_tag = "minecraft:" + rd["group"]
                if out_tag != out_id: self.tag_set.add(out_tag); self.item_id_to_tag[out_id] = out_tag
            out_item = ItemTagWithCount(ItemTag(tag=out_tag, item_id=out_id), out_n); recipe = Recipe(input_items, out_item)
            if out_id not in self.transitive_dependencies: self.transitive_dependencies[out_id] = set()
            skip = False
            for it in input_items:
                nm = it.item_tag.name
                if nm in self.transitive_dependencies and out_id in self.transitive_dependencies[nm]: skip = True
            if not skip:
                rid = out_item.item_tag.item_id
                self.itemid_recipes.setdefault(rid, []).append(recipe)
                for it in input_items:
                    nm = it.item_tag.name; self.transitive_dependencies[out_id].add(nm)
                    if nm in self.transitive_dependencies: self.transitive_dependencies[out_id].update(self.transitive_dependencies[nm])
                if out_item.item_tag.tag is not None: self.tag_recipes.setdefault(out_item.item_tag.tag, []).append(recipe)

    def craft(self, recipe):
        if recipe.output_item.item_tag.item_id not in self.itemid_recipes: return None
        for target in self.itemid_recipes[recipe.output_item.item_tag.item_id]:
            success = True; clone = copy.deepcopy(recipe.input_items)
            for itc in target.input_items:
                m = self.find_matching_item(itc.item_tag, clone)
                if m is None or m.count != itc.count: success = False; break
                clone.remove(m)
            if len(clone): success = False
            if success: return target.output_item
        return None

    def find_matching_item(self, itemtag, items):
        for x in items:
            if itemtag.item_id is not None:
                if x.item_tag.item_id == itemtag.item_id: return x
            elif itemtag.tag is not None:
                if x.item_tag.tag == itemtag.tag or self.item_id_to_tag.get(x.item_tag.item_id) == itemtag.tag: return x
        return None

    def is_craftable(self, item): return item in self.itemid_recipes or item in self.tag_recipes
    def is_valid_item(self, item): return item in self.itemid_set
    def is_tag(self, x): return x in self.tag_set
    def get_items_with_tags(self, tag):
        for item_id, t in self.item_id_to_tag.items():
            if tag == t: yield item_id

    def traverse_recipe_tree(self, item_name, visited):
        if item_name in visited: return []
        recipes = list(self.itemid_recipes.get(item_name) or self.tag_recipes.get(item_name) or [])   # fix (2): copy
        for recipe in recipes:
            nv = copy.deepcopy(visited)
            for it in recipe.input_items:
                nv.add(item_name); recipes.extend(self.traverse_recipe_tree(it.item_tag.name, nv))
        return recipes

    def collect_item_uses(self):
        uses = {}
        for d in (self.itemid_recipes, self.tag_recipes):
            for _, recipes in d.items():
                for r in recipes:
                    for it in r.input_items: uses.setdefault(it.item_tag.name, []).append(r)
        return uses

    def get_min_depth(self, x):
        if x in self.min_depth: return self.min_depth[x]
        if x in self.itemid_recipes: self.min_depth[x] = self._min_depth_recipes(self.itemid_recipes[x])
        elif x in self.tag_recipes: self.min_depth[x] = self._min_depth_recipes(self.tag_recipes[x])
        else: self.min_depth[x] = 0
        return self.min_depth[x]

    def _min_depth_recipes(self, recipes):
        return min(max(self.get_min_depth(it.item_tag.name) + 1 for it in r.input_items) for r in recipes)

    def item_recipes_min_depth(self, d):
        for item in self.itemid_recipes:
            k = self.get_min_depth(item)
            if k >= d: yield item, k

    def create_recipe_set(self, item_name, rng):
        uses = self.collect_item_uses(); recipes = self.traverse_recipe_tree(item_name, set()); distractors = []
        for r in recipes:
            for it in r.input_items:
                nm = it.item_tag.name
                if nm in uses: distractors.extend(rng.sample(uses[nm], min(len(uses[nm]), 10)))
        return recipes, distractors


_TREE = None
def tree():
    global _TREE
    if _TREE is None: _TREE = CraftingTree()
    return _TREE


def all_goals(min_depth=1):
    """AgentGym order: items with min depth >= 1 sorted by depth (stable)."""
    t = tree(); return sorted(t.item_recipes_min_depth(min_depth), key=lambda x: x[1])


MAX_PLAN = 30   # goals whose oracle plan (seed 0) is longer are dropped (e.g. netherite block: 69 steps of 9-nugget gold ingots)
_SPLITS = None


def splits():
    """By goal item (seeded shuffle within each depth, so every split has the same depth mix), after dropping goals with plan > MAX_PLAN:
         test  = 20% of the goal items x distractor seeds 0,1,2 (seed-major)
         dev   = 10% of the goal items x seeds 0,1
         train = the remaining 70% of the goal items x distractor seeds 0..S-1 (S chosen so the pool has >= 1500 tasks), goals in a seeded
                 random order, seed-major (all goals with seed 0, then seed 1, ...), so any prefix of the pool is depth-representative."""
    global _SPLITS
    if _SPLITS is not None: return _SPLITS
    goals = all_goals(); by = collections.defaultdict(list); env = TextCraftEnv(); dropped = []
    for g, d in goals:
        env.reset((g, 0)); plan = env.expert_actions()
        if plan is None or len(plan) > MAX_PLAN: dropped.append(g); continue
        by[d].append(g)
    rng = random.Random(1234); te, dv, tr = [], [], []
    for d in sorted(by):
        gs = sorted(by[d]); rng.shuffle(gs); n = len(gs); a = round(0.2 * n); b = round(0.1 * n)
        te += gs[:a]; dv += gs[a:a + b]; tr += gs[a + b:]
    S = math.ceil(1500 / max(1, len(tr))); rng.shuffle(tr); rng.shuffle(te); rng.shuffle(dv)
    mk = lambda g, s: f"tc_{item_id_to_str(g).replace(' ', '_')}_s{s}"
    _SPLITS = {"test": [mk(g, s) for s in range(3) for g in te], "dev": [mk(g, s) for s in range(2) for g in dv], "train": [mk(g, s) for s in range(S) for g in tr],
               "dropped": [mk(g, 0) for g in dropped]}
    # final protocol (Oct 8, user): TEST = the 108 test + 54 dev goals x distractor seeds 0-4 = 810 tasks (no dev set any more);
    # train_r1 = the unchanged 1512-task TRAIN pool in one fixed seeded order (K is computed on the failures among its first 1000)
    _SPLITS["test810"] = [mk(g, s) for s in range(5) for g in te + dv]
    tr1 = list(_SPLITS["train"]); random.Random(20261008).shuffle(tr1); _SPLITS["train_r1"] = tr1
    return _SPLITS


def budget_for(n_expert):
    """per-task step budget = 2 x the oracle plan length (min 6): plan lengths range 2..30, so one fixed budget is either impossible for the
    deep goals or ~5x slack for the shallow ones; 2x the task's own optimum keeps the slack uniform (mean budget ~11 = 2x the mean)."""
    return max(6, 2 * n_expert)


def parse_task(tid):
    m = re.fullmatch(r"tc_(.+)_s(\d+)", tid); return "minecraft:" + m.group(1), int(m.group(2))


# ----------------------------------------------------------------------------- port of agentenv_textcraft.environment (+ admissible / oracle)
class TextCraftEnv:
    action_regexes = {"craft": r"craft (.*) using (.*)", "get": r"get ([0-9]+) (.*)", "inventory": r"inventory"}
    count_regex = r"([0-9]+) (.*)"

    def __init__(self):
        self.t = tree(); self.inventory = {}; self.goal = None; self.shown = []; self.commands = ""

    # --- reset: same construction as AgentGym's TextCraftEnv.reset (recipes of the goal tree + <= 10 distractors, shuffled)
    def reset(self, task):
        goal, seed = parse_task(task) if isinstance(task, str) else task
        rng = random.Random(seed); self.inventory = {}; self.goal = goal; self.task = task
        recipes, distractors = self.t.create_recipe_set(goal, rng)
        rs = {}; ds = {}
        for r in recipes: rs.setdefault(r.recipe_str, r)
        for r in distractors:
            if r.recipe_str not in rs: ds.setdefault(r.recipe_str, r)
        dl = sorted(ds); lst = sorted(rs) + rng.sample(dl, min(len(dl), 10)); rng.shuffle(lst)
        allr = {**rs, **ds}; self.shown = [allr[s] for s in lst]; self.tree_recipes = set(rs)
        self.commands = "\n".join(lst)
        self.obs0 = "Crafting commands:\n{}\n\nGoal: craft {}.".format(self.commands, item_id_to_str(goal))
        return self.obs0

    def step(self, action):
        obs = None; reward = 0; done = False
        try:
            for kind, rx in self.action_regexes.items():
                m = re.match(rx, action)
                if not m: continue
                if kind == "craft":
                    recipe = self.extract_recipe(m.group(1), m.group(2))
                    if not self.has_items(recipe.input_items):
                        raise ActionFailed("Could not find enough items to craft {}".format(recipe.output_item.item_tag.item_id))
                    out = self.t.craft(recipe)
                    if out is None: raise ActionFailed("Could not find a valid recipe for {}".format(recipe.output_item))
                    self.remove_items(recipe.input_items); self.add_item(out.item_tag, out.count)
                    obs = "Crafted {} {}".format(out.count, out.item_tag.item_id)
                    if out.item_tag.item_id == self.goal: reward = 1; done = True
                elif kind == "get":
                    item, amt = m.group(2), int(m.group(1)); io = self.item_str_to_obj(item)
                    if self.t.is_craftable(io.name) or self.t.is_tag(io.item_id) or io.item_id is None or not self.t.is_valid_item(io.item_id):
                        raise ActionFailed("Could not find {}".format(item))
                    self.add_item(io, amt); obs = "Got {} {}".format(amt, item)
                    if io.item_id == self.goal: reward = 1; done = True
                elif kind == "inventory":
                    obs = "Inventory: "
                    if not self.inventory: obs += "You are not carrying anything."
                    for it, amt in self.inventory.items(): obs += "[{}] ({}) ".format(item_id_to_str(it), amt)
            if obs is None: raise ActionFailed("Could not execute {}".format(action))
        except ActionFailed as e:
            obs = str(e.args[0]); reward = 0
        return obs, reward, done

    def has_items(self, items):
        return all(x.item_tag.item_id in self.inventory and self.inventory[x.item_tag.item_id] >= x.count for x in items)

    def add_item(self, tag, amt): self.inventory[tag.item_id] = self.inventory.get(tag.item_id, 0) + amt

    def remove_items(self, items):
        for x in items:
            self.inventory[x.item_tag.item_id] -= x.count
            if self.inventory[x.item_tag.item_id] == 0: del self.inventory[x.item_tag.item_id]

    def extract_recipe(self, out_s, in_s):
        m = re.match("([0-9]+) (.*)", out_s)
        out = ItemTagWithCount(self.item_str_to_obj(m.group(2)), int(m.group(1))) if m else ItemTagWithCount(self.item_str_to_obj(out_s), 1)
        ins = []
        for x in in_s.split(","):
            mm = re.match(self.count_regex, x.strip())
            if not mm: raise ActionFailed("Wrong item format: {}".format(x.strip()))
            ins.append(ItemTagWithCount(self.item_str_to_obj(mm.group(2)), int(mm.group(1))))
        return Recipe(input_items=ins, output_item=out)

    def item_str_to_obj(self, item):
        iid = "minecraft:" + item.replace(" ", "_")
        return ItemTag(tag=iid) if self.t.is_tag(iid) else ItemTag(item_id=iid)

    # --- state for one-step lookahead (the whole state is the inventory)
    def get_state(self): return dict(self.inventory)
    def set_state(self, s): self.inventory = dict(s)

    # --- our additions --------------------------------------------------------------------------------------------------------
    def members(self, tag):
        """concrete items that satisfy a tag ingredient (items whose recipe group is that tag)."""
        return sorted(self.t.get_items_with_tags(tag))

    def fetchable(self, iid):
        t = self.t; return (not t.is_craftable(iid)) and (not t.is_tag(iid)) and t.is_valid_item(iid)

    def craft_variants(self, r, inv=None):
        """the shown command verbatim (tag names as written -- fails when a tag is used) + versions with every tag ingredient replaced
        by a concrete member held in the inventory."""
        inv = self.inventory if inv is None else inv
        opts = []
        for x in r.input_items:
            nm = x.item_tag.name; o = [nm]
            if x.item_tag.item_id is None:   # tag ingredient
                o += [m for m in self.members(nm) if inv.get(m, 0) > 0]
            opts.append(o)
        outs = []
        def rec(i, cur):
            if len(outs) >= MAX_TAG_VARIANTS + 1: return
            if i == len(opts):
                outs.append("craft {} {} using {}".format(r.output_item.count, item_id_to_str(r.output_item.item_tag.item_id),
                                                         ", ".join(f"{x.count} {item_id_to_str(n)}" for x, n in zip(r.input_items, cur)))); return
            for n in opts[i]: rec(i + 1, cur + [n])
        rec(0, []); return outs

    def get_counts(self):
        """{item name: sorted admissible counts}: every FETCHABLE ingredient named in a shown command (a 'get' of a craftable item or a tag
        always fails, so -- like ALFWorld's admissible list -- it is not offered), with the counts k*c, c = a count it appears with in a
        shown command, k in GET_MULTIPLES."""
        cs = collections.defaultdict(set)
        for r in self.shown:
            for x in r.input_items:
                if self.fetchable(x.item_tag.name): cs[x.item_tag.name].add(x.count)
        return {nm: sorted({k * c for c in v for k in GET_MULTIPLES}) for nm, v in cs.items()}

    def admissible(self):
        acts = ["inventory"]
        for nm, cnts in sorted(self.get_counts().items()):
            for c in cnts: acts.append(f"get {c} {item_id_to_str(nm)}")
        for r in self.shown:
            for a in self.craft_variants(r):
                if a not in acts: acts.append(a)
        return acts

    # --- oracle solver (uses only the shown commands) ----------------------------------------------------------------------------
    def _recipe_for(self, iid):
        rs = [r for r in self.shown if r.output_item.item_tag.item_id == iid]
        if not rs: return None
        def cost(r): return (max(self.t.get_min_depth(x.item_tag.name) for x in r.input_items), r.recipe_str not in self.tree_recipes, r.recipe_str)
        return min(rs, key=cost)

    def _concrete(self, ing, inv):
        """item to use for an ingredient: itself, or for a tag the member already held (most), else the shallowest member with a shown recipe."""
        if ing.item_tag.item_id is not None: return ing.item_tag.item_id
        ms = self.members(ing.item_tag.tag); held = [m for m in ms if inv.get(m, 0) > 0]
        if held: return max(held, key=lambda m: (inv[m], m))
        ok = [m for m in ms if self._recipe_for(m) is not None or self.fetchable(m)]
        return min(ok or ms, key=lambda m: (self.t.get_min_depth(m), m))

    def _get_split(self, nm, total):
        """fewest admissible gets summing exactly to total (DP over the admissible counts); if no exact sum exists (possible from a
        mid-episode inventory, e.g. 7 more nuggets with counts {2, 9, 18}), the fewest gets reaching >= total, least overshoot first.
        From an empty inventory an exact split always exists, so demos / budgets are unaffected by the fallback."""
        allc = self.get_counts().get(nm, [])
        if not allc: return None
        top = total + max(allc); best = {0: []}
        for s in range(1, top + 1):
            for c in allc:
                if s - c in best and (s not in best or len(best[s - c]) + 1 < len(best[s])): best[s] = best[s - c] + [c]
        if total in best: return best[total]
        ok = [s for s in best if s >= total]
        return best[min(ok, key=lambda s: (len(best[s]), s))] if ok else None

    def expert_actions(self, max_steps=200):
        """plan from the CURRENT inventory: depth-first, per craftable item ensure its ingredients for all needed batches (one merged get per
        base ingredient), then craft the batches.  Returns the action list (executed on a copy of the inventory) or None if it fails."""
        inv = dict(self.inventory); acts = []
        def ensure(iid, n, depth=0):
            if depth > 12 or len(acts) > max_steps: raise ActionFailed("too deep")
            have = inv.get(iid, 0)
            if have >= n: return
            if self.fetchable(iid):
                split = self._get_split(iid, n - have)
                if split is None: raise ActionFailed(f"no admissible get for {iid} x{n - have}")
                for c in split: acts.append(f"get {c} {item_id_to_str(iid)}"); inv[iid] = inv.get(iid, 0) + c
                return
            r = self._recipe_for(iid)
            if r is None: raise ActionFailed(f"no recipe for {iid}")
            B = math.ceil((n - have) / r.output_item.count)
            for _ in range(4):   # re-check: ensuring one ingredient can consume another (shared intermediates)
                order = sorted(r.input_items, key=lambda x: -self.t.get_min_depth(x.item_tag.name))
                conc = {id(x): self._concrete(x, inv) for x in r.input_items}
                for x in order: ensure(conc[id(x)], x.count * B, depth + 1)
                if all(inv.get(conc[id(x)], 0) >= x.count * B for x in r.input_items): break
            for _ in range(B):
                conc = [(x, self._concrete(x, inv)) for x in r.input_items]
                for x, c in conc:
                    if inv.get(c, 0) < x.count: ensure(c, x.count, depth + 1)
                conc = [(x, self._concrete(x, inv)) for x in r.input_items]
                acts.append("craft {} {} using {}".format(r.output_item.count, item_id_to_str(iid), ", ".join(f"{x.count} {item_id_to_str(c)}" for x, c in conc)))
                for x, c in conc:
                    inv[c] -= x.count
                    if inv[c] == 0: del inv[c]
                inv[iid] = inv.get(iid, 0) + r.output_item.count
        try: ensure(self.goal, 1)
        except (ActionFailed, RecursionError): return None
        return acts


def cmd_stats(a):
    env = TextCraftEnv(); goals = all_goals(); sp = splits()
    dep = dict(goals); print(f"recipes dir: {RECIPE_DIR} ({len(os.listdir(RECIPE_DIR))} files)")
    print(f"goal items (min depth >= 1): {len(goals)}; depth distribution {sorted(collections.Counter(d for _, d in goals).items())}")
    print(f"dropped goals (oracle plan > {MAX_PLAN} steps or unsolved): {sp['dropped']}")
    for k, v in sp.items():
        if k == "dropped": continue
        ds = collections.Counter(dep[parse_task(t)[0]] for t in v); print(f"split {k}: {len(v)} tasks, {len({parse_task(t)[0] for t in v})} goals, depth {sorted(ds.items())}")
    rows = []
    for k in ("test", "dev", "train"):
        for tid in (sp[k] if k != "train" else sp[k][: a.n_train]):
            env.reset(tid); adm = env.admissible(); plan = env.expert_actions()
            ok = False; inval = 0
            if plan:
                for act in plan:
                    if act not in adm: inval += 1
                    o, r, d = env.step(act); adm = env.admissible()
                    if d: ok = True; break
            rows.append({"split": k, "id": tid, "depth": dep[env.goal], "n_adm0": len(env.reset(tid) and env.admissible()), "n_cmd": len(env.shown),
                         "plan_len": len(plan) if plan else None, "ok": ok, "not_adm": inval})
    for k in ("test", "dev", "train"):
        rs = [r for r in rows if r["split"] == k]; ok = [r for r in rs if r["ok"]]
        L = [r["plan_len"] for r in ok]; A = [r["n_adm0"] for r in rs]
        print(f"[{k}] expert solves {len(ok)}/{len(rs)}; plan len mean {sum(L) / len(L):.2f} max {max(L)}; not-admissible expert actions {sum(r['not_adm'] for r in rs)};"
              f" |admissible| at reset mean {sum(A) / len(A):.1f} p90 {sorted(A)[int(.9 * len(A))]} max {max(A)}; shown commands mean {sum(r['n_cmd'] for r in rs) / len(rs):.1f}")
        for d in sorted({r["depth"] for r in rs}):
            rd = [r for r in ok if r["depth"] == d]
            if rd: print(f"    depth {d}: {len(rd)} solved, plan len mean {sum(r['plan_len'] for r in rd) / len(rd):.2f}")
    bad = [r for r in rows if not r["ok"]]
    if bad: print("unsolved examples:", [r["id"] for r in bad[:10]])
    if a.out: open(a.out, "w").writelines(json.dumps(r) + "\n" for r in rows)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats"); s.add_argument("--n_train", type=int, default=400); s.add_argument("--out", default=None)
    s = sub.add_parser("show"); s.add_argument("task")
    a = ap.parse_args()
    if a.cmd == "stats": cmd_stats(a)
    else:
        e = TextCraftEnv(); print(e.reset(a.task)); print("\nADMISSIBLE:", e.admissible()); print("\nEXPERT:", e.expert_actions())
