"""Verified-fix rate vs pointer position for one Attr-OPD arm/round: where the chosen step falls relative to the progress
oracle's first harmful action, by chosen step, and by when the first harmful action occurs.
Usage: python scripts/alfworld_fixability_report.py <arm> <round> <progress_labels.jsonl> [steps_file]"""
import json, sys
A = "reports/alfworld/opd/student3b_low100/"


def main():
    arm, rnd, labf = sys.argv[1:4]; stepsf = sys.argv[4] if len(sys.argv) > 4 else (A + f"{arm}/steps.jsonl" if rnd == "round1" else A + f"{arm}/steps_{rnd}.jsonl")
    lab = {r["id"]: r for r in map(json.loads, open(labf))}; lab = {k: v for k, v in lab.items() if v.get("t_first_regress") is not None}
    steps = {r["id"]: r["steps"][0] for r in map(json.loads, open(stepsf))}
    ex = {json.loads(l)["id"] for l in open(A + f"{arm}/{rnd}/examples.jsonl") if json.loads(l).get("kind", "verified") == "verified"}
    sk = {json.loads(l)["id"] for l in open(A + f"{arm}/{rnd}/skipped.jsonl")}
    done = ex | sk | {json.loads(l)["id"] for l in open(A + f"{arm}/{rnd}/examples.jsonl")}
    ids = [i for i in lab if i in steps and i in done]
    rate = lambda sel: f"{100 * sum(i in ex for i in sel) / len(sel):.0f}% (n={len(sel)})" if sel else "--"
    d = {i: steps[i] - lab[i]["t_first_regress"] for i in ids}
    print(f"{arm} {rnd}: {len(ids)} labelled+processed states; verified-fix rate {rate(ids)}")
    print("  pointer vs oracle: earlier>2", rate([i for i in ids if d[i] < -2]), "| earlier 1-2", rate([i for i in ids if -2 <= d[i] < 0]), "| exact", rate([i for i in ids if d[i] == 0]), "| later 1-2", rate([i for i in ids if 0 < d[i] <= 2]), "| later>2", rate([i for i in ids if d[i] > 2]))
    print("  by chosen step: 0", rate([i for i in ids if steps[i] == 0]), "| 1-2", rate([i for i in ids if 1 <= steps[i] <= 2]), "| 3-6", rate([i for i in ids if 3 <= steps[i] <= 6]), "| 7-15", rate([i for i in ids if 7 <= steps[i] <= 15]), "| >15", rate([i for i in ids if steps[i] > 15]))
    print("  by first harmful step: 1-2", rate([i for i in ids if lab[i]["t_first_regress"] <= 2]), "| 3-6", rate([i for i in ids if 3 <= lab[i]["t_first_regress"] <= 6]), "| >6", rate([i for i in ids if lab[i]["t_first_regress"] > 6]))


if __name__ == "__main__":
    main()
