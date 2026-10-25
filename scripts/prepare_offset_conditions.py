"""Location-precision experiment for the EXTERNAL reader (researcher, 2026-09-18):
given the gold decisive step x, compare feedback pointing at x (existing
`precise_only`), x+2, x-2 (same template, shifted along the renderer's
step_index, agent taken from the block at the shifted step), and no step
(existing `incorrect_only`). Only traces where BOTH offsets exist are kept so
the four conditions are paired on identical traces. Records n_steps and the
gold index for length-bucket analysis.

Output: data/feedback_study/prompts_offsets.jsonl (conditions step_plus2_only,
step_minus2_only; same row schema as prompts.jsonl).
"""
import json, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from whowhen_eval.render import get_renderer  # noqa: E402
from whowhen_eval.run import release_from_row  # noqa: E402
DR = ROOT / "data" / "whowhen_pro_official"
TAIL = "\n\nRespond with only your corrected final answer, in the same format as the original task expects. Do not include any other commentary."


def agent_at(rr, coord):
    blk = next((b for b in rr.blocks if b.coord == coord), None)
    if blk and "| Agent: " in blk.text.split("\n")[0]:
        return blk.text.split("\n")[0].split("| Agent: ")[1].strip()
    return None


def main():
    base = {}
    for l in open(ROOT / "data" / "feedback_study" / "prompts.jsonl"):
        r = json.loads(l)
        if r["condition"] == "precise_only":
            base[r["id"]] = r
    rows_by_id = {}
    for l in open(DR / "data" / "text.jsonl"):
        r = json.loads(l)
        if r["id"] in base:
            rows_by_id[r["id"]] = r
    out, skipped, kept_fw = [], Counter(), Counter()
    for tid, b in base.items():
        rel = release_from_row(rows_by_id[tid], "text", DR); rr = get_renderer(rel["framework"])(rel)
        coords = [c for c, _ in rr.step_index]
        if str(b["gold_step"]) not in coords:
            skipped["gold step not in index"] += 1; continue
        x = coords.index(str(b["gold_step"])); n = len(coords)
        if x - 2 < 0 or x + 2 > n - 1:
            skipped["offset out of range"] += 1; continue
        p = b["prompt"]; k = p.find("## Coordinator Feedback")
        head = p[:k].rstrip()
        for name, xi in (("step_plus2_only", x + 2), ("step_minus2_only", x - 2)):
            coord = coords[xi]; ag = agent_at(rr, coord)
            loc = f"at step {coord}, in {ag}'s turn," if ag else f"at step {coord}"
            fb = (f"Your team's answer to this task was incorrect. Specifically, {loc} an error occurred. "
                  "Please review that step and provide a corrected final answer.")
            row = {k2: v for k2, v in b.items() if k2 not in ("prompt", "condition")}
            row.update({"condition": name, "prompt": head + "\n\n## Coordinator Feedback\n\n" + fb + TAIL,
                        "n_steps": n, "gold_index": x, "shifted_step": coord})
            out.append(row)
        kept_fw[rel["framework"]] += 1
    path = ROOT / "data" / "feedback_study" / "prompts_offsets.jsonl"
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"{len(out)} prompts ({len(out)//2} traces) -> {path}\nkept by framework: {dict(kept_fw)}\nskipped: {dict(skipped)}")


if __name__ == "__main__":
    main()
