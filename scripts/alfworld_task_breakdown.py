"""Pooled (774-game) success by ALFWorld task type for the student and chosen (arm, round) adapters.
Usage: python scripts/alfworld_task_breakdown.py attr_nofb:round1 oracle:round2 ..."""
import re, sys
from collections import defaultdict
sys.path.insert(0, "scripts")
from alfworld_pooled_report import splits, pooled

TYPES = "pick_and_place_simple|pick_clean_then_place_in_recep|pick_heat_then_place_in_recep|pick_cool_then_place_in_recep|look_at_obj_in_light|pick_two_obj_and_place"


def ttype(g):
    m = re.search(rf"/({TYPES})", g); return m.group(1) if m else "?"


def main():
    arms = [("student", None, None)] + [(a, a.split(":")[0], a.split(":")[1]) for a in sys.argv[1:]]
    tab = {}
    for name, arm, rnd in arms:
        d = defaultdict(lambda: [0, 0])
        for (sp, g), w in pooled(splits("student3b_low100", arm, rnd)).items():
            d[ttype(g)][0] += w; d[ttype(g)][1] += 1
        tab[name] = d
    print(f"{'task type':32s}" + "".join(f"{n:>22s}" for n, _, _ in arms))
    for t in sorted(tab["student"]):
        print(f"{t:32s}" + "".join(f"{100 * tab[n][t][0] / max(1, tab[n][t][1]):>16.1f}% n={tab[n][t][1]:<3d}" for n, _, _ in arms))


if __name__ == "__main__":
    main()
