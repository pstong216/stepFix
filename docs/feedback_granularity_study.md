# Feedback Granularity and Multi-Agent Collaboration

## Motivating question

Does the *type* of failure feedback handed back to a multi-agent team
change how well it recovers, and does it change collaboration dynamics
(blame attribution, overcorrection) independently of recovery success?

This is a variant of the main research plan's RQ2 (attribution
representation) and Sec. 20's teacher-information ablation (T0-T3),
reframed around **multi-agent collaboration** as the object of study
rather than downstream policy-training utility.

## The 4 conditions

| Condition | Location | Content (why) | = Plan's Sec. 20 label |
|---|---|---|---|
| `reason_only` | none | error type + short description | (new) |
| `vague_reason` | which quarter of the interaction | error type + short description | (new) |
| `precise_reason` | exact step + agent | error type + short description | T3 (failure critique) |
| `precise_only` | exact step + agent | none | T1 (critical-step indicator) |

Deliberately NOT included (for now): a no-feedback baseline and a
vague-location-only (no reason) cell -- the researcher scoped this to
these 4. Precise-location-only-WRONG (an intentionally incorrect step
number) would be a natural follow-up control for measuring "blind trust"
in stated location, but is out of scope for this first pass.

## Implementation

`src/error_attr/attribution/feedback_conditions.py`: `build_feedback_conditions(...)`
builds all 4 condition strings from one release row's gold (agent, step,
mode). Two design choices worth remembering:

- **Location coarsening is framework-agnostic.** Rather than trying to
  vaguely describe each framework's native round/position coordinate
  system, `vague_reason` buckets the gold step's ORDINAL POSITION within
  the renderer's own `step_index` into quarters (Q1-Q4) and reports it as
  "roughly steps N-M of T". Works identically whether the underlying
  coordinate is a flat step number or a round.position pair. (Edge case
  fixed: when `total < 4`, naive quarter-bucket math can produce `lo > hi`
  -- clamped so `lo <= hi` always holds.)
- **Mode description is name + one plain sentence, not the full taxonomy
  entry.** `taxonomy.yaml`'s descriptions are written for an annotator
  (multi-sentence, with "Distinguished from R.2..." disambiguation
  asides) -- correct register for the attributor's own judge/training
  prompt, wrong register for a coordinator's natural-language feedback to
  a team. `_mode_description` re-reads `taxonomy.yaml` directly (not the
  pre-formatted judge-prompt block) and keeps only `name` + first
  sentence of `description`.

## Scoping the gradable subset

Who&When Pro's multi-agent frameworks split by how hard `task.answer` is
to grade automatically:

- **Easily/reliably gradable (used here):** `debate`, `dylan` (multiple-
  choice letter, exact match), `mathchat` (numeric answer), `magentic-one`
  (short factual string, GAIA/SimpleQA-style). Total 2,328 rows.
- **Deferred:** `macnet`, `metagpt` (code generation -- grading needs
  executing the candidate code against the benchmark's own test cases,
  which we don't have a harness for yet).

## LIMITATION (raised by the researcher, 2026-09-08): the corrector is not the erring agent

The failed trajectories come from Who&When Pro -- i.e. from OTHER agent
systems (GPT-era frameworks: debate, mathchat, magentic-one, ...), not from
the Qwen2.5-7B model that receives the feedback. The feedback text's
framing ("Your team's answer was incorrect") is therefore a fiction: Model
B is a third-party reader given someone else's failed transcript plus a
hint, not a team correcting its own mistake.

What this study therefore measures: **the information value of different
feedback types for an external reader's correction of a failed transcript.**
What it does NOT measure, despite the section title above: multi-agent
collaboration dynamics or self-correction. A mundane reading of the main
result is available -- to someone reading another system's transcript,
"step 3 was wrong" carries little information while "the error type was X"
says what to avoid -- and the current design cannot distinguish that from
the stronger claim. Whether "why > where" holds when a team corrects ITS
OWN failure is an open question this data cannot answer. The paired design
and statistics are sound for the narrower claim; the collaboration framing
must be read as the narrower claim.

Model A (the attributor) is NOT affected by this: attribution trained on
one set of agents and applied to others is the intended transfer setting
(Research Plan Sec. 8, RQ3).

Fix = generate the failed trajectories from the tested Qwen model itself.
Two routes: (A) natural failures from a Qwen-driven agent (single-agent
first, then Qwen-driven debate), labeled by Model A's predictions -> this
is also the first true end-to-end use of the attributor
(attribution -> feedback -> correction); (B) replicate Who&When Pro's
error injection on Qwen's own successful trajectories for gold labels
(needs a Qwen multi-agent runner; the official repo ships only renderers).
The project's own ALFWorld environment (Qwen policy rollouts + PDDL oracle
for the step label) is a ready-made substrate for the same idea.

### Self-play chain pilot (route A, 40 GPQA failures, 2026-09-08 evening)

Pipeline validated end-to-end on Qwen's OWN failures (3x Qwen2.5-7B
debate, 2 rounds): Model A attributions parsed 40/40; 5 conditions
(4 + `incorrect_only` control) generated and graded; corrector = the same
Qwen2.5-7B. Results (n=40, NOT powered -- 0-3 discordant pairs per
comparison, all p >= 0.25): incorrect_only 17.5%, reason_only 12.5%,
vague_reason 15.0%, precise_reason 15.0%, precise_only 20.0%. No condition
beats the no-information control, and the external-transcript pattern
(reason >> precise_only) does not appear at this n.

Two things the full run must handle: (1) GPQA is 4-choice, so switching
away from a known-wrong letter yields ~33% by chance -- every condition is
BELOW that, i.e. the model mostly keeps its own wrong answer (anchoring).
This is precisely the self-correction difficulty the external-reader
design could not observe, and a chance-level re-guess reference must be
reported for MC benchmarks. (2) Answer-change rate per condition is the
direct anchoring metric (computed on the pilot; see STATUS.md).
Pilot outputs archived at reports/selfplay_chain_pilot_gpqa/.

### Self-play chain, FULL POOL, `one_shot` protocol (2026-09-09, ~00:30)

361 usable self-generated failures of the Qwen2.5-7B debate team
(scibench 163, math 116, gpqa 42, mmlu_pro 40); Model A parsed 361/361;
corrector = the same Qwen2.5-7B, one corrected answer per condition.

| condition | recovery | changed own answer | correct given changed |
|---|---|---|---|
| `incorrect_only` (no-info control) | 18/361 = 5.0% | 31.0% | 16.1% |
| `reason_only` | 15/361 = 4.2% | 27.7% | 15.0% |
| `vague_reason` | 14/361 = 3.9% | 29.1% | 13.3% |
| `precise_reason` | 18/361 = 5.0% | 31.9% | 15.7% |
| `precise_only` | 20/361 = 5.5% | 31.6% | 17.5% |

Per benchmark (recovery, chance-level re-guess in brackets): gpqa
12-21% [33%], mmlu_pro 5-10% [11%], math 2.6-5.2% [~0], scibench
1.2-1.8% [~0]. Paired McNemar: no condition beats `incorrect_only` (all
p >= 0.5); `reason_only` vs `precise_only` p=0.33 with the sign REVERSED
relative to the external-transcript study (6 vs 11 discordant).

**Reading:** on its OWN failures this team keeps its wrong answer ~70% of
the time regardless of what it is told, and when it does change it lands
on the right answer only 13-18% of the time (below MC chance for gpqa).
The external-reader effect ("why >> where") does not appear in self-
correction; anchoring, not information, is the bottleneck under
`one_shot`. This is exactly the distinction the researcher's objection
drew, now measured. Implication for route B: gold-vs-predicted label
quality is moot here -- even the label-free control matches every
condition. The `debate_round` protocol (feedback as an extra round to
the same three agents) is the remaining test; its pilot showed higher
answer-change rates (37-75%), so it is the protocol most likely to move
this. Outputs: reports/selfplay_chain_one_shot_full/.

## Protocol: one-shot corrective continuation (not full re-run)

We don't have the original frameworks' live orchestration engines (only
`whowhen_eval`'s read-only renderers), so a full multi-turn re-run of
debate/dylan's actual protocol isn't available. Instead: render the full
failed trajectory (via the same renderer used for attributor training),
append one of the 4 feedback conditions as a coordinator message, and ask
a single model to produce a corrected final answer in one shot. Compare
against `task.answer` for a recovery/success metric.

This is a real scope reduction from "study live collaboration dynamics
across multiple turns" -- it tests whether feedback TYPE changes
correction ABILITY, but not multi-round dynamics like blame-shifting
across agents in a live back-and-forth. Plan B (live intervention, sketched
in conversation) would be the fuller version if this first pass shows an
effect worth chasing further.

## Status

- [x] `feedback_conditions.py` built and validated (2,328/2,328 rows across
  the 4 target frameworks produce all-4 conditions with a valid location
  bucket; zero failures).
- [x] Data-prep script selecting the gradable subset + building the
  corrective-continuation prompt for each of the 4 conditions.
- [x] Grading logic per framework (exact-match for debate/dylan/mathchat
  via `\boxed{}`; magentic-one via `FINAL ANSWER:`, exact-match after
  normalization -- see results caveat below).
- [x] Generation run: full 9,312 rows (2,328 traces x 4 conditions),
  Qwen2.5-7B-Instruct, batched + length-sorted + OOM-safe fallback.
  Completed 2026-09-08.
- [x] Analysis (recovery rate by condition, per-framework breakdown,
  paired significance testing). C.3-taxonomy-flagged overcorrection check
  NOT yet done (would need per-example inspection of which agent's
  content changed, not just final-answer correctness).

## Results (2026-09-08)

**Overall recovery rate (n=2,328 traces per condition):**

| condition | recovery rate |
|---|---|
| `reason_only` | 32.39% |
| `vague_reason` | 31.06% |
| `precise_reason` | 32.30% |
| `precise_only` | 29.55% |

**Paired significance (McNemar's exact test, same traces across
conditions):**

| comparison | winner-wins / loser-wins | p |
|---|---|---|
| `reason_only` vs `precise_only` | 101 vs 35 | 1.3e-8 |
| `vague_reason` vs `precise_only` | 67 vs 32 | 5.6e-4 |
| `precise_reason` vs `precise_only` | 95 vs 31 | 9.6e-9 |
| `reason_only` vs `precise_reason` | 42 vs 40 | 0.91 (tie) |

**Reading:** `precise_only` (location, no diagnostic content) is
significantly worse than every condition carrying a reason -- highly
robust given n=2,328 and effect sizes this large. Once a reason is
present, adding precise location on top (`reason_only` -> `precise_reason`)
gives no further measurable benefit (p=0.91). This says diagnostic
CONTENT ("why") is the active ingredient for correction, not location
("where") -- directly relevant to the main research plan's Sec. 20
T1 (critical-step indicator, = our `precise_only`) vs T3 (failure
critique, = our `precise_reason`) framing: T3 beats T1, and in fact a
location-free reason alone matches T3.

**Per-framework breakdown** shows the same direction in 3/4 frameworks
(debate, mathchat clearly; magentic-one in the same direction but
compressed). `dylan` (smallest n=152) doesn't cleanly replicate the
pattern -- likely underpowered at that n, not a real reversal.

**magentic-one (GAIA) ceiling-effect caveat:** the one-shot corrective-
continuation protocol gives the model NO new tool/retrieval capability --
it only re-reasons over the SAME fixed transcript. For GAIA questions
whose correct answer depends on information the team's original attempt
never actually retrieved, no feedback of any kind can fix the answer,
which compresses the differences between conditions specifically for this
framework relative to the self-contained-reasoning frameworks
(debate/dylan/mathchat, where all needed information IS already in the
transcript). This caveat does not undermine the overall finding -- it
explains why the *effect size* is framework-dependent, and if anything
the self-contained-reasoning frameworks are the cleaner test of the
underlying hypothesis.

**Grading-strictness caveat:** the parser is exact-match-after-
normalization, applied identically across all 4 conditions -- so any
false negatives (e.g. magentic-one answering "6 crocodiles" against gold
"6") are symmetric across conditions and shouldn't bias the cross-
condition comparison, but they do mean the absolute recovery-rate numbers
understate true correction ability, particularly for magentic-one.

## Connecting the trained attributor (Model A) to this study (2026-09-08)

The results above all used GOLD (oracle) labels to build the feedback
paragraphs -- an idealized upper bound, not a test of the actual trained
attributor's usefulness. Closed that gap: reran the same 4-condition
protocol using Model A's (`reports/attributor_qwen2.5-3b-lora/final`)
OWN PREDICTED `(agent, step, mode)` instead of ground truth, restricted to
the 232 traces that are both (a) one of the 4 gradable frameworks and (b)
in Model A's held-out val split -- i.e. traces Model A did NOT train on
(using training-set predictions would just reflect memorized gold labels,
not genuine attributor accuracy).

`scripts/prepare_feedback_study_data_predicted.py` builds the predicted-
label prompts (feeds Model A's predictions into the same
`build_feedback_conditions()` used for gold labels -- the function's
graceful fallbacks, e.g. `_quarter_bucket`'s handling of a step string
that doesn't match any real coordinate, meant no code changes were needed
for this reuse). `scripts/run_feedback_study.py` gained a `--prompts_path`
flag so the same generation/grading pipeline could be reused as-is.

**Result: essentially no difference between oracle and real-predicted
feedback**, on the same 232 traces:

| condition | gold (oracle) | Model A (predicted) | diff | McNemar p |
|---|---|---|---|---|
| `reason_only` | 34.5% | 34.9% | +0.4pp | 1.00 |
| `vague_reason` | 32.3% | 33.2% | +0.9pp | 0.50 |
| `precise_reason` | 35.3% | 35.3% | 0.0pp | 1.00 |
| `precise_only` | 31.0% | 29.7% | -1.3pp | 0.25 |

Only 2-4 traces (out of 232) ever flip between correct/incorrect when
swapping oracle labels for Model A's real predictions, across any
condition. This is a meaningful, reassuring result given Model A is NOT
perfect on this held-out set (82.8% mode accuracy, 95.6% step accuracy) --
there was real room for its errors to hurt downstream correction, and
they largely didn't. Combined with the main finding (reason content
matters, location alone doesn't), this suggests the bottleneck for
practical deployment is which TYPE of feedback is given, not how
accurate the attributor generating it is -- at least at this attributor's
current accuracy level. (Caveat: n=232 is much smaller than the
2,328-trace main comparisons, so this test has less power; "not
significant" here means "no detected difference at this scale," not
"proven identical" -- but the tiny, inconsistent-direction effect sizes
make the qualitative read reasonably solid.)

## Not yet done

- C.3 (over-reliance/overcorrection) check: does an agent's actually-
  correct prior content get revised away under vague/no-location
  feedback more often than under precise feedback? Needs per-example
  diffing against the original (also-saved) trajectory content, not just
  final-answer correctness.
- macnet/metagpt (code generation) -- deferred, needs an execution
  harness to grade against benchmark test cases.
- Plan B (live intervention during rollout, not post-hoc) -- not started;
  would be the natural follow-up if a live multi-agent harness gets built
  for other reasons.
- Scale the oracle-vs-predicted comparison beyond n=232 -- would need a
  larger held-out validation split for the 4 gradable frameworks
  specifically (i.e. retraining Model A on a reshuffled split), since the
  current 232 is capped by how much of the original 90/10 split happened
  to land in these 4 frameworks.
- RL (GRPO) training of the attributor, in addition to SFT -- discussed
  in conversation, not yet started. Reward would reuse `official_score`
  directly (verifiable-reward RL); starting point would be the SFT
  checkpoint, not the base model. Open question worth testing: does RL
  generalize better than SFT on the ALFWorld transfer set, where SFT's
  in-distribution 81.6% joint accuracy drops to 20.3%?


## ADDENDUM 2026-09-15: the missing baselines change the headline

Three zero conditions were run on the same 2,328 traces with the same reader and
output constraint (data/feedback_study/prompts_controls.jsonl,
reports/feedback_study_results_controls.jsonl; current grader, DECISIONS #25/#26):
task_only 14.3% -> no_feedback 28.5% -> incorrect_only 42.4%. Against incorrect_only:
reason_only +1.2pp (78/50, p=0.017), precise_reason +1.1pp (p=0.036), vague_reason n.s.,
precise_only **-1.9pp** (34/79, p=3e-5). The original "why >> where" gap (3.1pp, p=2e-9)
is mostly the location-only condition falling BELOW the outcome-label baseline.
Corrected headline: the outcome label carries the effect; error type adds ~1pp; a
bare step pointer hurts. See STATUS.md "External-reader study: the missing baselines".
