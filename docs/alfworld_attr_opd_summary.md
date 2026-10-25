# ALFWorld Step-Localised, Verified Attr-OPD: Experiment Summary (2026-09-22, revised)

**Revision note.** This version corrects three verified errors and weakens several overclaims flagged by an
external methodological review (Sections 5, 6, 7, 10). The most important corrections: the `const1_r2` control is
not an end-to-end zero-attribution baseline (Section 5); the refreshed-vs-stale attributor comparison enters
round 2 from two different checkpoints (Section 4); a percentage was miscomputed (Section 6). A new,
better-supported finding is added in Section 7 (candidate-location union coverage). All comparisons in this
document are single-seed McNemar tests across tasks; they do not capture training-seed variability, and "not
statistically significant" is reported as "not distinguishable at this sample size," not as evidence of equality.

All experiments use the ALFWorld benchmark and a shared substrate: a ReAct-style policy over admissible commands
(`src/error_attr/policy/alfworld_agent.py`), a PDDL-planner oracle (`src/error_attr/attribution/oracle_alfworld.py`),
and a distillation pipeline (`scripts/alfworld_attr_opd.py`) that localises one training step per failed episode,
has a teacher propose corrections there, verifies them by student rollout, and updates the student with a KL loss
only on verified targets. Splits: `unseen` = 134 valid_unseen games, `seen` = 140 valid_seen games, `heldout` = 500
held-out train games (1500-1999, never touched by OPD or by the student's SFT -- **but see Section 11.4: the
teacher's SFT data contains 45 of them and the planner-refreshed attributor's data 22**), `pooled` = all 774 games combined (heldout carries
~65% of the pooled weight -- keep all three splits visible, do not read pooled significance as an unseen-
generalization claim). All paired comparisons use exact McNemar (`scipy.stats.binomtest`).

## 1. Setup

**Student (training target).** Qwen2.5-3B-Instruct, LoRA (r=32, all attention + MLP projections), fine-tuned by
standard next-action-token cross-entropy (prompt/observation tokens masked with `-100`) on 100 winning expert
(PDDL-planner) trajectories, 1 epoch. Greedy success on the 134 unseen games: **41.8% (56/134)**, pooled 774-game
baseline **53.7% (416/774)**.

**Teacher.** Same base model and training objective, but 549 winning expert trajectories, 2 epochs (SFT loss
0.66 -> 0.04). Greedy unseen success: **89.6% (120/134)**; all 14 remaining failures are budget exhaustion, not
wrong decisions.

**Verification mechanism (one training example).** At a chosen step *t* in a failed episode: (1) the teacher
scores every admissible command by its own log-likelihood at that state (SayCan-style) and proposes its top-2;
(2) the **current student** (not the teacher) replays the prefix, takes each candidate action, and continues
greedily for up to 20 steps; (3) the training target is built **only from candidates that led to a win** --
teacher-softmax-weighted if multiple win, one-hot if one wins; if none win, the state is either (a) skipped
entirely ("`_nofb`" / verified-only recipe) or (b) supervised with the teacher's raw (unverified) distribution
over all admissible commands ("with-fallback" recipe). The loss is `KL(target || student's own distribution at
that state)`, backpropagated only through the LoRA parameters.

**Coupling to flag as a confound (not yet controlled for).** The teacher is behavior-cloned on the ALFWorld
built-in PDDL planner's own winning trajectories; `t_first_regress` (Section 2) is computed by querying that
*same* planner's plan-length function at every step. This is a plausible reason `t_first_regress` and the teacher
work well together beyond "the position itself is better." A control not yet run: swap in a teacher never
fine-tuned on this planner's trajectories and see whether `t_first_regress`'s advantage shrinks.

## 2. Two oracle notions of "the error step," and a non-oracle pointer

- **t\* (point of no return)**, `find_critical_step`: binary search over prefixes for the last one from which the
  handcoded-expert planner can still win within the same step budget. This label depends on the planner actually
  being optimal and on a fixed remaining-step allowance; a planner timeout is missing evidence, not evidence of
  irrecoverability, and should not be over-read as ground truth.
- **t_first_regress (first harmful action)**, `progress_trace`: replay the trajectory, query the planner's
  shortest-remaining-plan length after every prefix; the first action that *increases* that length. This is a
  **planner-defined progress label**, not automatically "the" ground-truth first wrong decision -- it inherits
  whatever blind spots the planner's plan-length heuristic has. Mean position on round-1 failures: step 2.0.
- **Model A**, a real 7B attributor trained on human-annotated Who&When Pro failures (cross-domain, no ALFWorld
  exposure originally), later refreshed with 1,099 ALFWorld self-play failures labelled by the progress oracle. It
  reads only trajectory text, no privileged environment access.

**Important caveat on why Model A's round-1 numbers resemble the oracle's (Section 3): this is very likely a
floor effect, not evidence of cross-domain transfer.** A completely content-blind constant baseline ("always
point at step 1," no attributor at all) scores **59.8% exact / 90.4% within-2** on the exact round-1 failure
population used to build `attr_nofb` (n=450, first-harmful-action mean step 2.0) -- *higher* than the original
(pre-refresh) Model A's own 11% exact / 72% within-2 on the same population. The original attributor is at or
below this constant baseline on every ALFWorld failure population tested, including populations where errors
happen later (e.g. 44.8% within-2 for original Model A vs 67.8% for the constant baseline on round-2 failures,
mean error step 5.4).

**Directly measured mechanism (checked against Who&When Pro's actual training data,
`data/attributor_sft_no_observations/train.jsonl`, n=5,351 parsed examples).** Who&When Pro's trajectories are
short (mean 3.91 steps; framework medians of 2-7) and its gold step labels are concentrated at small *absolute*
step numbers (mean 2.51, median 2.0; 94.6% of labels fall at step 0-5). The labels are **not** biased toward early
*relative* position -- if anything the opposite (mean relative position 0.703, median 0.857; the two largest
frameworks, smolagents at 55% of the data and mathchat at 15%, have mean relative positions 0.818 and 0.849, i.e.
skewed *late*). The most parsimonious explanation for Model A pointing early on ALFWorld is therefore an
**absolute-small-integer output habit carried over from training on much shorter trajectories**, not a learned
"errors happen early" prior and not primarily a rendering-format-transfer effect: an ALFWorld episode runs up to
50 steps, roughly ten times longer than Who&When Pro's typical 2-7, so the same small-absolute-number habit,
calibrated on short trajectories, reads as "very early" once applied to a much longer one. This is closer to an
out-of-distribution trajectory-length generalization failure than to any transferred notion of where errors
occur, and it happens to be a good guess on round-1 ALFWorld failures only because those are almost all early and
salient. Read "attr_nofb = oracle_nofb" at round 1 (Section 3) as "fixing the first decision, verified, is enough
when the novice's errors are all early," **not** as evidence that human-annotated multi-agent attribution
knowledge transfers to embodied single-agent tasks. The real test of transfer is round 2 onward, where the
original attributor's weights fail (Section 4) and have to be refreshed on ALFWorld's own progress-oracle labels
to keep working -- i.e., whatever Model A learned from Who&When Pro did not robustly transfer once the easy
early-error floor effect (and the length mismatch that produces it) stops helping by coincidence.

## 3. Round-1 arm comparison (same 319 shared failures, one teacher call each)

| arm | pointer | fallback? | verified ex / 319 | unseen | seen | heldout | **pooled** | vs student (b/a, p) |
|---|---|---|---|---|---|---|---|---|
| student | -- | -- | -- | 41.8% | 57.1% | 56.0% | **53.7%** | -- |
| uniform | every step | yes | 40 (equal budget) | 8.2% | 3.6% | 6.0% | **5.9%** | collapse, p=2e-92 |
| last | last step | yes | 0 | 32.8% | 32.1% | 32.4% | **32.4%** | harm, p=1e-21 |
| random | random step | yes | 31 | 34.3% | 37.9% | 41.4% | **39.5%** | harm, p=2e-12 |
| attr_w1 | Model A step +-1 window | yes | 172 (769 states) | 16.4% (unseen only) | -- | -- | -- | harm, p=3e-6 |
| attr | Model A step | yes | 116 (36.4%) | 47.8% | 52.1% | 56.2% | **54.0%** | null, p=0.95 |
| oracle | t_first_regress | yes | 130 (40.8%) | 58.2% | 63.6% | 59.0% | **59.7%** | p=0.002 |
| **attr_nofb** | Model A step | **no** | 125 (39.2%) | 58.2% | 65.0% | 65.8% | **64.3%** | p=6e-10 |
| **oracle_nofb** | t_first_regress | **no** | 124 (38.9%) | 56.0% | 67.9% | 63.6% | **63.0%** | p=5e-8 |

**Reading.** With a fallback to the teacher's unverified opinion, only the oracle pointer helps and only mildly;
Model A's own pointer is statistically null; random/last/uniform actively hurt. Removing the fallback turns both
real pointers into strong, mutually indistinguishable wins (64.3% vs 63.0%, p=0.45).

**Open data-quality caveat.** The verified-example counts differ between the fallback and no-fallback runs of the
*same pointer on the same 319 states* (attr 116 vs attr_nofb 125; oracle 130 vs oracle_nofb 124), even though the
teacher's candidate proposals are confirmed byte-identical between the pairs (checked directly against
`targets.jsonl`). The fallback runs predate per-state `examples.jsonl`/`skipped.jsonl` logging (that logging was
added later in development), so a direct per-state diff of *which* candidate verified as a win is not available
for them, and the exact mechanical cause (a verification-code change between the two runs is the most likely
explanation, given how much this pipeline was iterated on) has not been pinned down. Treat "removing fallback" as
a comparison across runs that were not perfectly matched on every other dimension, not as a single isolated
variable change.

## 4. Iterating the recipe: does the pointer keep tracking the policy as it improves?

Each round recollects T=1 failures with the previous round's adapter, re-attributes, re-verifies, re-trains.

| chain | round 1 | round 2 | round 3 |
|---|---|---|---|
| oracle (with fallback) | 59.7% | **76.5%** | 74.0% (plateau) |
| attr (with fallback) | 54.0% | 55.0% (null) | -- |
| **oracle_nofb** (verified-only) | 63.0% | **76.0%** | -- |
| **attr_nofb** (verified-only, stale pointer) | 64.3% | **56.8% (REGRESSED)** | -- |
| **attralf_nofb** (verified-only, refreshed pointer) | 58.9% | **76.2%** | -- |

The planner-oracle chain compounds cleanly. The stale-attributor chain regresses at round 2 (64.3% -> 56.8%).
Refreshing the attributor each round on the previous round's own self-play failures (free progress-oracle labels)
recovers compounding (76.2% at round 2).

**Caveat on this comparison.** `attralf_nofb` and `attr_nofb` enter round 2 from **different round-1 checkpoints**
(58.9% and 64.3% pooled respectively) and therefore face different remaining-failure pools and different training
opportunities at round 2 -- this is not a controlled "same starting point, only the attributor differs"
experiment, it is a comparison of two independently iterated chains. The clean version of this claim requires
branching both a stale and a refreshed attributor from **one shared checkpoint, one frozen failure pool, and
identical teacher proposals** (queued as the top-priority next experiment, Section 10).

### Verification breadth ablation (k=2 vs k=3 candidates, round-1 attr_nofb recipe)

| verify_k | verified ex / 319 | env steps | unseen | pooled | vs k=2 (b/a, p) |
|---|---|---|---|---|---|
| 2 (default) | 125 (39.2%) | 14.3k | 58.2% | 64.3% | -- |
| 3 | 158 (49.5%) | 21.2k | 59.0% | 66.0% | 57/44, p=0.23 (not distinguishable at this n) |

A third candidate finds more verified fixes for more environment steps, but the extra fixes don't measurably
transfer to unseen games at this sample size; k=2 remains the default pending a larger comparison.

## 5. How much of the gain is attribution, vs the verified-OPD recipe itself?

`const1_r2`: same 575 round-2 failures, same teacher, same verified-only recipe as `attralf_nofb` round 2, and
**the same starting adapter** (`attralf_nofb/round1`) -- but the supervised step is always step 1, with no
attribution.

| adapter (round 2, same 575 failures, same starting adapter) | unseen | seen | heldout | **pooled** |
|---|---|---|---|---|
| student (for reference only, not the same starting point as the two rows below) | 41.8% | 57.1% | 56.0% | **53.7%** |
| **const1_r2** (no attribution, always step 1, from `attralf_nofb/round1`) | 73.9% | 73.6% | 71.4% | **72.2%** |
| **attralf_nofb round 2** (refreshed attributor, from the same `attralf_nofb/round1`) | 68.7% | 80.0% | 77.2% | **76.2%** |

**Correction from the previous version of this document.** `const1_r2` starts from `attralf_nofb/round1`, which
already benefited from one round of attribution-guided training. The 72.2% pooled result therefore does **not**
establish an 18.5pp gain from zero attribution throughout training -- the student row above is shown for reference
only and is not a valid comparison point for that claim. **The clean, controlled result is the 4.0pp difference
between the two round-2 branches that share the same starting checkpoint** (65/34, p=0.0024): attribution adds a
real but modest increment on top of whatever the verified-recipe-plus-that-particular-checkpoint already achieves.
Establishing the stronger "zero attribution from the original student, throughout" claim requires a `const1 ->
const1` chain launched from the original SFT student, not yet run.

**Do not dismiss the unseen-split reversal as noise without further evidence.** On the 134 unseen games alone,
`const1_r2` is numerically higher than `attralf_nofb` round 2 (73.9% vs 68.7%, p=0.09, not distinguishable at this
n). This could be sampling noise, but it could also reflect a real difference in how the two training signals
generalize to genuinely unseen environments versus the held-out-from-the-same-distribution training games (which
supply 500 of the 774 pooled games). Keep all three splits reported separately; the pooled number should not be
read as settling the unseen-generalization question.

## 6. Pointer-imprecision tolerance: is it symmetric, and does it depend on which pointer?

Same 319 round-1 failures, same teacher, verified-only recipe; only the supervised step is shifted by a fixed
offset from each pointer's own position (clamped to `[0, n-1]`).

**Shifting from `t_first_regress`:**

| offset | verified ex / 319 | unseen | pooled | vs exact (b/a, p) |
|---|---|---|---|---|
| -2 | 116 (36.4%) | 53.7% | 60.5% | 97/45, p=1.5e-5 vs student; 67/87 p=0.13 vs exact (not distinguishable) |
| 0 (exact) | 124 (38.9%) | 56.0% | 63.0% | -- |
| +1 | 97 (30.4%) | 53.0% | 62.1% | 64/71, p=0.61 (not distinguishable) |
| +2 | 84 (26.3%) | 41.8% | 52.2% | 66/78, p=0.36 vs student (no measurable effect at this n) |

**Shifting from Model A's own step:**

| offset | verified ex / 319 | unseen | pooled | vs exact (b/a, p) |
|---|---|---|---|---|
| 0 (exact) | 125 (39.2%) | 58.2% | 64.3% | -- |
| +1 | 107 (33.5%) | 50.0% | 56.3% | 41/103, p=2.4e-7 (clear drop) |
| +2 | 87 (27.3%) | 47.8% | 54.7% | 38/113, p=7.7e-10 (indistinguishable from student, p=0.62) |

**Shifting from t\* (233/319 failures with a valid label at the time this ran):**

| offset | verified ex / 233 | unseen | pooled |
|---|---|---|---|
| 0 (exact) | 20 (8.6%) | 53.0% | 59.2% |
| +1 | **0 (0%)** | = student (untrained adapter, 0 gradient steps) | **53.7%** |
| +2 | **0 (0%)** | = student | **53.7%** |

**Findings (verification yield vs downstream gain are different quantities).** Verified-fix rate falls smoothly
with |offset| in every direction tested, but downstream success does not track it smoothly for `t_first_regress`
and Model A's step: -2 and +1 lose little downstream despite losing verified examples, while +2 collapses all
downstream gain even though its fix rate (26-27%) is not dramatically lower than +1's. **This is the report's
single most useful observation: verification yield alone does not explain downstream performance.** Candidate,
not yet distinguished, explanations include example count, task-category coverage, the resulting state
distribution seen during training, and per-example learning difficulty -- Section 10's priority-3 experiment is
designed to separate these. The pointer-dependence (Model A's own step tolerates less forward offset than
`t_first_regress`, and t\* tolerates none) is consistent with, but does not prove, "how close the base pointer
already is to the true first-harmful-action determines its forward margin" -- an alternative reading is that the
pointers differ in other ways (e.g., which failure sub-population they tend to select) that happen to correlate
with offset-tolerance.

## 7. Cross-pointer agreement analysis

For the 319 shared round-1 failures, comparing Model A's chosen step against `t_first_regress`'s step
*per-failure*:

- **Location union coverage.** Model A's verified-only recipe alone repairs 125/319 = 39.2% of failures;
  `t_first_regress`'s verified-only recipe alone repairs 124/319 = 38.9%; the two sets overlap in 85 failures.
  By inclusion-exclusion, the **union of the two single-action locations repairs 125+124-85 = 164 failures =
  51.4%** of the shared set -- 12.2pp more than either pointer alone, using only two independent single-action
  interventions. This is the sharper, better-supported version of the earlier "location search helps" claim: it
  is direct evidence that **searching more candidate locations, not necessarily lengthening any one intervention,**
  captures a large share of the additional repairable failures. It does not by itself establish anything about
  whether multi-step windows are also needed for the remaining unrepaired failures (see below).
- Among the 85 jointly-fixable failures, the step distance `Model A - oracle` has mean 0.69, median -1.0 (Model A
  points earlier 61.2% of the time), tightly concentrated (68.2% within 1 step, 78.8% within 2 steps), with a long
  positive tail (14.1% differ by more than +2, up to +22) -- a minority of jointly-fixable failures have two
  genuinely different, independently valid repair points, not the same point measured with noise.
- 24.9% (with the t\* candidate included) / 24.8% (Model A vs oracle only, n=319) of failures are fixable by
  *exactly one* pointer -- real location-sensitivity, not redundancy.
- **43.3% are fixable by neither pointer (with t\* included).** This is *not*, by itself, evidence that a
  multi-step repair window is required -- it could equally reflect a missed location, insufficient teacher top-2
  coverage at the locations tried, an under-generous continuation budget, or a student too weak to execute an
  otherwise-correct continuation. Distinguishing between these explanations is exactly Section 10's priority-4
  diagnostic ladder; the current report does not have evidence to prefer "needs a longer window" over these
  alternatives.

## 8. Operational lessons (bugs found and fixed this session)

1. Never edit a running orchestration script while a queue invocation is reading it -- corrupted at least one run
   into evaluating a nonexistent adapter (now guarded against explicitly, see 3).
2. A default `--ids_from` file that silently grows across a session (from unrelated later CPU labelling jobs
   writing to the same path) is a scope bug -- always pin new arms to a frozen reference file.
3. A crashed pipeline stage must hard-fail (`|| exit 1` plus an explicit adapter-existence check), not silently
   fall through to evaluating whatever adapter path happens to exist.
4. A file-presence-based "already labelled" dedup check is not the same as "successfully labelled" -- a stored
   timeout error was silently treated as done and never retried until the error rows were explicitly deleted.
5. Teacher/pointer coupling (Section 1) is a live, unresolved confound.
6. 0 verified training examples means 0 gradient steps (PEFT's default LoRA has a zero `B` matrix), so such an
   adapter is byte-identical to its starting checkpoint and its evaluation is fully predictable in advance.
7. **Verification outcomes were not confirmed reproducible across separate runs of the identical (state, teacher
   candidates) input** (Section 3's caveat) -- this needs to be resolved (e.g., a same-input re-run test) before
   any future claim that isolates "fallback" or any other single flag as the only variable changed between two
   compared runs.

## 8b. Teacher-distribution-only ablation (no verification at all, 2026-09-22)

Same 319 round-1 failures, same teacher, same starting checkpoint as `oracle_nofb`/`attr_nofb`, but `verify_k=0`:
skip verification entirely and train directly on the teacher's full softmax distribution over admissible commands
at the attributed step (with the student's own failed action masked out and the remainder renormalized). No
environment rollout occurs during training (`env_steps=0` for all 319 states; none are skipped, since there is no
verification to fail).

| method (same oracle pointer, same 319 states) | pooled 774 | vs student (b/a, p) |
|---|---|---|
| student (no training) | 53.7% | -- |
| **oracle_teacheronly (pure teacher distribution, no verification)** | **53.4%** | 108/111, p=0.89 (no measurable effect) |
| oracle (fallback: verified when possible, else teacher distribution) | 59.7% | -- |
| oracle_nofb (verified-only) | 63.0% | -- |

`oracle_teacheronly` is statistically indistinguishable from doing nothing at all, and significantly worse than
both the fallback recipe (30/79, p=3.0e-6) and the verified-only recipe (62/137, p=1.1e-7) at the *identical*
localisation. This is the cleanest evidence in this report that **verification, not localisation, does essentially
all of the work**: even at the best available pointer position, imitating the teacher's opinion without an
environment check produces no measurable policy improvement. `attr_teacheronly` (Model A's pointer, same ablation)
is running to check whether this holds for the non-privileged pointer too.

**Result for the non-privileged pointer (`attr_teacheronly`, Model A's step, same no-verification recipe):
pooled 774 = 47.0%, significantly WORSE than the untrained student** (102/154, p=0.0014) **and significantly worse
than the privileged-pointer version of the same ablation** (`oracle_teacheronly`, 53.4%; 98/147, p=0.0021). With
the planner's privileged pointer, removing verification nets to zero measurable effect; with Model A's learned
(imperfect) pointer, removing verification is actively harmful. Reading: verification is not merely "nice to
have" on top of a good pointer -- it is what prevents an imprecise, learned pointer's unverified corrections from
actively degrading the policy. The combination "imprecise localisation + no verification" is worse than either
alone (an imprecise pointer with verification, `attr_nofb`, still reaches 64.3%; a precise pointer without
verification, `oracle_teacheronly`, is merely neutral at 53.4%).

## 8c. Retention check: does the recipe break tasks the student already solved? (2026-09-22)

All prior sections measure the *fix rate* on the pooled 774-game evaluation -- i.e. success on games the
pre-distillation student was failing. That leaves an obvious, unasked question: does the same training also
*break* games the pre-distillation student was already winning? This costs no new compute -- it only requires
joining the existing per-arm pooled-eval logs (unseen 134 + seen 140 + held-out-train 500) against the
pre-distillation baseline's log by `game_file`, restricted to the 416/774 games the baseline already won.

| arm (same 774-game pool, % of the 416 baseline wins still won after training) | retained |
|---|---|
| `uniform` (every step of every failed trajectory supervised, no localisation at all) | **7.0%** |
| `random` (one uniformly random step per failure, no localisation) | 57.0% |
| `attr` (Model A pointer, fallback to unverified teacher distribution allowed) | 72.4% |
| `oracle` (planner pointer, fallback allowed) | 79.8% |
| `attr_teacheronly` (Model A pointer, no verification at all, Sec. 8b) | 64.3% |
| `oracle_teacheronly` (planner pointer, no verification at all, Sec. 8b) | 72.9% |
| `attr_nofb` (Model A pointer, verified-only) | 88.5% |
| `oracle_nofb` (planner pointer, verified-only) | **87.7%** |

Two readings, both new to this report (not previously measured):

1. **`uniform` is not just inefficient -- it is actively destructive.** Treating every action in a failed
   trajectory as a positive training target (necessarily including the actions that led to failure) collapses
   93% of the student's prior competence. This is a much stronger argument for localisation than "it fixes more
   failures" (Sec. 3): naive full-trajectory imitation on failure data is actively harmful to the policy, not
   merely a wasted training signal.
2. **Verification protects retention, independently of localisation quality.** At matched localisation (compare
   `attr`/`oracle` vs `attr_teacheronly`/`oracle_teacheronly` vs `attr_nofb`/`oracle_nofb`), adding real
   environment verification and dropping unverified fallback targets monotonically improves retention alongside
   the already-known fix-rate improvement (Sec. 8b). This is an independent line of evidence for the same
   conclusion as Sec. 8b ("imprecise localisation + no verification is worse than either alone"), but on a
   different, previously unmeasured axis (forgetting rather than fixing) -- i.e. verification is not only about
   *how much you gain*, it is also about *how little you lose*.

Caveat: `uniform` and `random` are deliberately weak controls (see `scripts/alfworld_attr_opd.py` `cmd_choose`),
not competitive baselines, so their extreme retention numbers are somewhat expected in direction; the useful new
result is the *magnitude* (93% collapse for `uniform`) and the *monotonic ordering by verification* among the
real recipes, which was not measured or claimed anywhere in this report before now.

## 11. Part II (2026-09-23 to 09-28): frontier shift, planner-free refresh, uncertainty, clean pairs, retention

All evaluation below is greedy (T=0) on the same 774 games unless stated. Planner labels (`t_first_regress`) are
used as a *ruler* for pointer agreement only; Section 11.6 shows agreement is a poor predictor of training value.

### 11.1 Two corrections to earlier results
- **`const1_nofb` (always step 1).** The first rerun used the 14B zero-shot default teacher of
  `run_alfworld_opd_arm.sh` instead of the canonical 3B SFT teacher (579-tensor model in the log). Rerun with the
  correct teacher: verified 114/319, unseen 51.5%, **pooled 55.2%** = untrained student 53.7% (94/83, p=0.45), far
  below `attr_nofb` 64.3% (49/120, p=4.7e-8) -- although step 1 equals the round-1 planner label 62% of the time.
- **Teacher identity.** Every canonical arm uses the 3B SFT teacher (549 expert trajectories), not a 14B model;
  the `self_nofb` comparison is "stronger same-size SFT teacher vs the student itself".

### 11.2 The error frontier moves, and a stale attributor falls behind it
On games that fail in both rounds (184-221 per chain), the planner's first harmful action moves later in 71-78%
of games after one round (median +2-3 steps; relative position 0.02 -> 0.08-0.10). Untrained games shift *more*
often than games whose state was trained (73-82% vs 62-65%, n=34-41 trained): the shift looks like global
generalization, not per-game repair. Pointer agreement with the planner (within +-2): round-1 original Model A
65.1% (constant "step 1": 84.3%); round-2 original 43.5% (constant "round median": 67.8%); round-2
planner-refreshed 69.2%. Visualised in the "Attribution Drift" artifact.

### 11.3 The four attributors (all Model A = Qwen2.5-7B LoRA, agent-text-only input)
| name | extra ALFWorld fine-tuning (from the original) | label source |
|---|---|---|
| original | none (Who&When Pro only) | human annotation |
| verify-refreshed (VL) | 187 round-1 failures | earliest tried step where a replacement action was verified to win (no planner) |
| PL450 | 450 round-1 failures | planner |
| v1 | 1,099 failures (round-1 450 + a sibling chain's round-2 440 + zero-shot 112 + strong student 97) | planner |
All refreshes use ~1.1k training samples. Agreement on attr_nofb's 382 planner-labelled round-2 failures:
original 20.4% exact / 44.5% within-2; VL 13.9% / 47.1%; PL450 77.5% / 81.9%; v1 78.0% / 82.2%. PL187 (planner
labels on VL's exact 186 failures) 72.8% / 77.0% -> VL's low agreement is caused by the label definition
("earliest repairable" is not "where it went wrong"; with only early steps tried it degenerates into an
early-position prior), not by data size; v1's extra round-2 rows add nothing over PL450. Memorisation check on
games never seen in refresh data: v1 on attralf failures 52.3% exact vs best constant 15.4% (within-2 ties at 63.1%).

### 11.4 Leakage audit
OPD training pools (all rounds/chains) and the student's SFT data have zero overlap with unseen/seen/heldout.
The teacher's SFT data (`expert_all_train.jsonl`) contains **45 heldout games**; v1's refresh data contains **22
heldout-game rows** (`zeroshot` source, collected with `--n_games 2500`). unseen and seen are clean everywhere.
Removing these 67 games changes no conclusion (e.g. VL round 2: 73.8% -> 73.6%).

### 11.5 Planner-free refresh: poor agreement, strong student
Round 2 from `attr_nofb/round1` (same 528 failures, same teacher; only the pointer differs):

| round-2 pointer | unseen | seen | heldout | pooled |
|---|---|---|---|---|
| start (attr_nofb round 1) | 58.2% | 65.0% | 65.8% | 64.3% |
| original Model A | 49.3% | 60.7% | 57.8% | 56.8% |
| **verify-refreshed** | **70.9%** | **76.4%** | **73.8%** | **73.8%** |
| reference: oracle_nofb R2 (different start) | 73.1% | 80.7% | 75.4% | 76.0% |
| reference: attralf_nofb R2 (v1, different start) | 68.7% | 80.0% | 77.2% | 76.2% |

VL vs original: 162/31, p=1.3e-22; vs its start: 114/41, p=3.8e-9; vs the planner chains p=0.086 / 0.053 (not a
controlled comparison). Verified 231/528 (44%) vs 155/528 (29%). **216 of VL's 231 trained states are steps 0-1**;
77% lie before the planner's first harmful action. The stale pointer put 33% of its choices at step >= 10, where
verification rarely succeeds. Open question (not run): does a constant/position-matched early-step pointer from the
same start match VL? The same-start planner-refreshed branch was cancelled by the researcher.

### 11.6 Planner agreement does not predict training value
`const1_nofb` agrees with the round-1 planner label 62% (exact) but gives no gain; the original Model A agrees 15%
but gives +10.6pp; VL agrees 15% but gives +9.5pp at round 2. Use downstream student success as the metric.

### 11.7 Round-2 pointer-offset tolerance (oracle pointer, unseen only, single seed)
exact 98/134; -2: 85; -1: 68; +1: 83; +2: 77. All offsets cost 11-30 games vs exact (round 1: -2 and +1 cost
little), but the curve is non-monotonic (-1 worst) and n=134, so read only as "tolerance is not wider at round 2".

### 11.8 Verification success rates
| pointer | states | >=1 of 2 candidates wins | candidate 1 | candidate 2 | steps 0-2 | steps >= 3 |
|---|---|---|---|---|---|---|
| R1 Model A | 319 | 39.2% | 28.2% | 30.1% | 43.9% | 27.5% |
| R1 planner | 319 | 38.9% | 24.5% | 28.8% | 39.4% | 36.8% |
| R1 t* | 233 | 8.6% | 6.0% | 2.6% | -- | 8.6% |
| R1 Model A, self candidates | 319 | 27.3% | 18.2% | 20.4% | 32.5% | 14.3% |
| R2 original Model A | 528 | 29.4% | 23.3% | 21.4% | 36.2% | 22.2% |
| R2 verify-refreshed | 528 | 43.8% | 35.4% | 30.7% | 45.6% | 26.0% |
| R2 planner | 360 | 46.7% | 39.2% | 30.3% | 58.1% | 42.0% |
The teacher's top-2 ordering barely separates the candidates; early steps are more repairable.

### 11.9 Cheap signals for predicting verification (none usable)
Teacher entropy (r=-0.019, recomputed with a saved script; the earlier r=0.052 is superseded), LESS-style gradient
similarity (r=0.018), retrieval verb-typicality (within-2 9.3%), and nine teacher/student distribution features
(AUC 0.44-0.59, best: student entropy 0.594).

### 11.10 Uncertainty: attributor and student
- **Attributor** (full distribution over the blamed step via teacher-forced "Step Number: k"): confidence predicts
  planner agreement only for planner-trained attributors -- AUC(p_max -> exact) PL450 0.87, v1 0.86 (most confident
  third 97% exact), original 0.59, VL 0.46 (VL is confident, median p_max 0.72, but its confidence is uninformative).
- **Student** (the generating policy, every step of the 528 failures, 26,400 states, median 32 commands): action
  choice is genuinely uncertain -- normalized entropy 0.56, effective choices 6.6, p_top 0.40; openings are the most
  uncertain (0.63). The top summed-score command is also a shortest command at only 1% of steps (length is not
  driving choice); per-token-mean scoring flattens the distribution to 0.95 (it dilutes the one or two
  discriminative tokens) and is not a meaningful policy measure. At the planner's harmful step the student is not
  more uncertain than elsewhere (normalized entropy 0.53 vs 0.55; AUC 0.48); 53% of harmful actions were the
  student's own top choice. Student-uncertainty pointers reach within-2 11-44%, below constants.
- **Relation:** per-failure correlation between the attributor's step distribution and student uncertainty is
  ~0.01-0.03; planner-trained attributors blame steps where the student was slightly *more* confident than usual.
- Neither side predicts verification success well (AUC 0.54-0.62).

### 11.11 How many "errors" are real? Counterfactual (contrastive) re-verification
For each verified state, continue greedily from the round's starting student after (a) the original action and (b)
the verified alternative:

| data | verified states | original action also wins | alternative still wins | clean pairs (b wins, a loses) |
|---|---|---|---|---|
| R1 attr_nofb | 125 | 22% | 74% | 69 |
| R1 oracle_nofb | 124 | 30% | 77% | 65 |
| R1 attralf_nofb | 106/122 | 30% | -- | 53 |
| R2 verify-refreshed | 231 | 43% | 78% | 97 |
| R2 original Model A | 155 | 46% | 81% | 61 |

Training trajectories are collected by sampling every step at T=1 from a very uncertain student, so many failures
come from later unlucky samples, not from the blamed step; at round 2 almost half. Only 74-81% of "verified"
alternatives still win on re-check, because in-training verification uses the student as it is being updated.
Implications: many KL targets are not corrections; any preference-pair method must use only clean pairs.

### 11.12 Greedy retention of previously solved games
| method | before -> after | kept | newly broken | newly fixed |
|---|---|---|---|---|
| R1 Model A, verified only | 53.7 -> 64.3% | 88.5% | 48 | 130 |
| R1 planner, verified only | 53.7 -> 63.0% | 87.7% | 51 | 123 |
| R1 self candidates | 53.7 -> 60.2% | 91.1% | 37 | 87 |
| R1 always step 1 | 53.7 -> 55.2% | 80.0% | 83 | 94 |
| R1 planner, with fallback | 53.7 -> 59.7% | 79.8% | 84 | 130 |
| R1 Model A, with fallback | 53.7 -> 54.0% | 72.4% | 115 | 117 |
| R1 random step | 53.7 -> 39.5% | 57.0% | 179 | 69 |
| R1 every step | 53.7 -> 5.9% | 7.0% | 387 | 17 |
| R2 planner-refreshed (attralf) | 58.9 -> 76.2% | 95.0% | 23 | 157 |
| R2 planner | 63.0 -> 76.0% | 94.5% | 27 | 127 |
| R2 verify-refreshed | 64.3 -> 73.8% | 91.8% | 41 | 114 |
| R2 original Model A | 64.3 -> 56.8% | 78.7% | 106 | 48 |
Forgetting is 5-12% per round for verified-only recipes and largest where training signal is unverified or
misplaced; verification is itself the main guard against forgetting.

### 11.13 Using success data, planner-free pointers and untrained teachers (round 1, same start, 774 games)
Offline arms are compared with M0 (the same 125 verified KL targets, trained offline; online `attr_nofb` = 64.3%).
| arm | what it adds | pooled | kept | vs M0 / note |
|---|---|---|---|---|
| M0 offline KL | -- | 60.7% | 86.1% | reference |
| M2 + KL anchor on own successes | 500 success states, KL(old || new) | 59.9% | 90.1% | 59/65, p=0.65 (less forgetting, fewer fixes) |
| M3 rejection-sampling FT only | SFT on 1,000 states of own wins, no teacher | 57.2% | 86.8% | 67/94, p=0.04 (worse) |
| M4 DPO only | 69 clean pairs (Sec. 11.11) | 60.2% | 88.7% | 55/59, p=0.78 (same with 55% of the data) |
| M6 + continuation SFT | 555 states of the winning greedy continuations | 63.3% | 87.3% | 90/70, p=0.13 |
| **M5 progress-estimator pointer** (online verify, like attr_nofb) | pointer = first action that raises a remaining-steps estimator trained only on own wins | **65.5%** | **91.8%** | vs attr_nofb 61/52, p=0.45; verified 149/319 |
| **constant step 0** (online verify, same start) | always the first action; control for M5's early pointers | 59.3% | 87.5% | verified 135/319; M5 vs const0 84/36, p=1.4e-5; attr_nofb vs const0 82/43, p=6e-4 |
| untrained 7B teacher + hint | teacher never trained on ALFWorld | 59.3% | 85.1% | vs attr_nofb 47/86, p=0.0009; verified 103/319 |
| untrained 7B teacher, no hint | same, no attributor note | 58.1% | 84.6% | hint vs no hint 50/41, p=0.40 |
M5 is fully planner-free (its estimator only sees the student's own successful trajectories) and matches the
Model-A pointer; its pointer agrees with the planner only 7.5% exact (85.5% within 2; 86% of pointers at steps 0-1)
-- another instance of Sec. 11.6. An untrained teacher works but costs ~5 points; the attributor's note to the
teacher adds nothing measurable.

### 11.14 Relaxing "success" for discarded states (diagnostic, planner as ruler only)
For the 194 round-1 states where no candidate won, continuations after each single teacher alternative end closer
to the goal than the original action's continuation in 25-28% of cases and farther in 33-39% (median gain when
closer: 1 step). Taking the better of two alternatives looks better (41% closer) only by selection. These states are
mostly ones where the student cannot make progress whatever it does (median 6 steps from the goal after 20 more
steps), not partial fixes; relaxing success to "closer" would mostly admit noise, consistent with the fallback arms.

### 11.15 Multi-agent debate: more discussion rounds after feedback (Qwen2.5-7B, 118 true failures)
| condition | correct after round 1 / 2 / 3 / 4 | ever correct | answer changed r1 -> r4 |
|---|---|---|---|
| no feedback, just more rounds | 2.5 / 5.9 / 5.9 / 5.9% | 10 | 21% -> 58% |
| "your answer was incorrect" | 5.1 / 9.3 / 11.0 / 11.0% | 21 | 35% -> 70% |
| Model A location + reason | 5.1 / 6.8 / 11.0 / 8.5% | 18 | 40% -> 65% |
| expert-framed "incorrect" | 8.5 / 7.6 / 15.3 / 13.6% | 25 | 48% -> 74% |
| expert-framed location + reason | 8.5 / 7.6 / 11.9 / 11.0% | 22 | 38% -> 72% |
More rounds make teams change answers far more (58-74% by round 4) but rarely to the right one (<=14%), and correct
answers do not stick (they flip back between rounds). Round 4 vs round 1 is not significant in any condition (best
10/3, p=0.09). Only authority framing beats no feedback (expert "incorrect" at round 4: 13/4, p=0.049); diagnostic
content still adds nothing (location+reason vs plain "incorrect" at round 4: 7/10, p=0.63).

## 9. What this report's evidence does and does not establish

| Observation | What it supports | What remains open |
|---|---|---|
| Removing fallback improves both attribution arms | Selecting supervision by verified student continuation is valuable here | Whether the benefit is target quality, fewer/cleaner updates, or a different training distribution (confounded further by the unresolved run-to-run discrepancy in Section 3) |
| Attribution beats `const1_r2` by 4.0pp (same starting checkpoint) | Location selection adds value on top of the verified recipe, at round 2 | Whether it beats a fully attribution-free chain run from scratch, and whether the gain generalizes to unseen environments specifically |
| Refreshed reaches 76.2%, stale reaches 56.8% | Adapting the pointer to the evolving student is promising | Causal effect of refreshing per se, since the two chains enter round 2 from different checkpoints |
| Later offsets lose downstream gain while retaining substantial verified-fix rate | Verification yield alone does not explain training benefit | Whether the gap is explained by example count, coverage, state distribution, or difficulty |
| Two independent single-action pointers jointly cover 51.4% vs 39% each | Searching more locations captures real additional value | Whether any residual failures specifically need a longer intervention, vs simply more/different locations |

## 10. Revised next steps (reordered per external review; supersedes the previous pilot design)

The most defensible framing is not "does a fixed t+N window help," but a three-way distinction the current
verification procedure only partially measures:

- **Attribution:** which past action explains the failure (what Model A / `t_first_regress` / t\* each answer).
- **Repairability:** can *some* intervention at a state let *this* student finish -- approximated by
  `Q^pi_B(h,a) = P(success within remaining budget B | history h, action a, then follow pi)`, which the current
  verified-target construction weights teacher proposals by (restricted to the tested candidates).
- **Training utility:** does distilling from that intervention improve *independent* held-out performance --
  not guaranteed by high `Q^pi_B` alone, and the thing actually measured in Sections 5-6's downstream numbers.

Priority-ordered next experiments (each answers a specific question this report currently cannot):

1. **Shared-checkpoint, shared-failure-set pointer comparison.** From one round-1 checkpoint and one frozen
   failure pool, with identical teacher, verification rule, optimizer settings, and eval tasks, compare: always
   step 1, a depth-matched random location, the stale attributor, the refreshed attributor, and the planner
   pointer as a privileged reference. Report both an equal-acquisition-budget comparison and an equal-verified-
   example-count comparison (they answer different questions; equalizing only teacher-call count is insufficient
   when continuation lengths differ).
2. **Multiple locations vs. more actions at one location.** Compare (a) four candidate actions at one location,
   (b) two candidate actions at each of two locations, (c) the same allocation at depth-matched random locations
   -- measured in real environment steps and model compute, not nominal candidate count. Section 7's 51.4% union
   number is the motivating evidence that location search may dominate action-count search.
3. **Matched-size training on exact vs. offset/later verified repairs.** Control for number of unique tasks,
   verified states, optimizer updates, supervised tokens, and (where feasible) the same underlying failures.
   Report newly-solved tasks and previously-solved tasks that now fail *separately* -- a flat net rate could hide
   real learning offset by forgetting. Also check whether the trained policy's probability of the verified action
   at the training states actually increased (distinguishes "failed to learn the target" from "target didn't
   transfer").
4. **Diagnostic ladder for "none of three pointers repaired this failure."** On 50-100 stratified failures, test:
   more candidate locations (still one action); more candidate actions at the same location; a planner-proposed
   action followed by the student; 2-4 teacher actions then student continuation; teacher continuing to
   completion. Keep the *same total remaining budget* across all of these (a fresh 20-step allowance per restart
   is not the same problem as the original remaining budget an oracle label like t\* assumes). Include an
   unchanged-action replay control (should reproduce the original failure exactly; an unexpected success flags a
   state-restoration or budget mismatch).
5. **Adaptive selection and a second environment**, only after 1-4 clarify whether the bottleneck is location
   selection, training-example selection, or intervention length.

**Revised go/no-go criterion.** Do not require a strong correlation between a repair-feasibility proxy and
downstream training gain before continuing -- a weak correlation could mean the interesting thing is true
(repairability is necessary but not sufficient for useful supervision), and a handful of aggregated arm results
cannot establish a reliable per-example selection rule regardless. The better test: **at matched acquisition and
training budget, does selecting examples by the proposed feasibility score outperform random selection from the
same verified-candidate pool?** If not, investigate learning utility, coverage, or redundancy before concluding
the direction fails.

**Novelty risk (unverified by direct reading of the source papers; flagged as reported by external review, not
independently confirmed here).** Prior work said to already combine candidate-critical-step localisation, teacher
alternatives, student-verification, and localized distillation/DPO includes CSO and, more directly, "From Scoring
to Acting" (OVCSD: selects intervention states, backtracks on repair failure, retains and imitates successful
teacher continuations on ALFWorld and WebShop). ADWIN and TurnOPD are said to cover adaptive rollout-length
allocation. If accurate, "localise + correct + verify + distill" and "adaptive window length" are both
substantially occupied; the more defensible contribution is the attribution/repairability/training-utility
mismatch as the student evolves, and a selector trained on actual repair outcomes with a demonstrated budget
advantage over simple location-search baselines -- not a variable-N mechanism or a refreshed classifier alone.
These citations should be independently read and checked against our method before being treated as settled.

On theory: a local first-order expansion `F(theta - eta g_I) = F(theta) - eta <grad F(theta), g_I> + O(eta^2)`
explains *why* verification (which does not measure gradient alignment with the true objective) might fail to
predict training benefit, and is a reasonable motivation for a learned selection criterion -- but it is a standard
Taylor expansion, not a novel theorem, and should not be presented as one.

### 11.16 M5 round 2: does the refreshed estimator follow the student? (522 round-2 failures of M5's round-1 student)
Same failures, same pointer rule (first rise > 1 in expected remaining steps); only the estimator differs. Action categories
from the log-only rules of `scripts/analyze_pointer_action_types.py` (A explore, B redundant, C regress / Cu unrecovered, D progress).
| estimator | median step | at step 0 | explore | redundant | regress (unrecovered) |
|---|---|---|---|---|---|
| stale (trained on round-1 student's wins) | 0 | 68% | 75% | 20% | 4% (2%) |
| refreshed (trained on round-2 student's own wins, no oracle) | 2 | 27% | 48% | 13% | 36% (25%) |
The refreshed estimator moves the pointer later in 307/522 games (same 190, earlier 25) and toward regressions, although
regressions are a smaller share of all round-2 failure steps (13% vs 23% in round 1). M5 R2 (refreshed): verified
264/522, unseen 90/134 (R1 79/134).
M5 R2 pooled: **70.3%** (unseen 67.2%), up from R1 65.5% (fixed 112 / broke 75, p=0.008; keeps 85.2% of R1-solved games).
For reference (different round-1 starts): verify-refreshed Model A R2 73.8%, planner R2 76.0%, stale Model A R2 56.8%.
The same-start control (constant step 0 at round 2) is running.

## 12. Step selection, environment-evidence filters, and 7B self-evolution (2026-09-29/30)

### 12.1 Step selection, 3B student (same 100 round-1 failures, 3B SFT teacher, verified-only, unseen 134)
| step chosen by | median step | verified | unseen after training (student 56) |
|---|---|---|---|
| Model A (external attributor) | 0 | 40/100 | **70** (15/1 vs student, p=0.001) |
| max student entropy | 9 | 29/100 | 60 |
| random | 24 | 14/100 | 61 |
Model A vs entropy 12/2 (p=0.013), vs random 11/2 (p=0.022); entropy vs random n.s. Confound: Model A also yields more examples.

### 12.2 Teacher-action filters without rollout (experiment C; labels = full verification; scripts/env_evidence_*.py)
- Single-signal AUC (3B teacher, 638 candidates | 7B SFT teacher, 200): goal object revealed after one step 0.68 | 0.76;
  hindsight (failed log does not contradict) 0.65 | 0.69; teacher p 0.55 | 0.60; KL 0.52-0.54 | 0.49-0.54; student
  entropy is INVERTED (high entropy -> more likely to win: 0.59 | 0.68).
- Within the same state (one candidate wins, the other loses): env model picks the winner 0.82 (3B, 73 pairs) / 0.87 (7B, 26);
  teacher's first choice 0.49 / 0.54; state-level entropy/KL 0.50 by construction. State salvageability: student entropy
  AUC 0.61 / 0.67. => entropy finds WHERE correction is possible, environment evidence finds WHICH action.
- State-disjoint 3B->7B transfer of the env+dist filter: AUC 0.800 [0.73, 0.87].
- Hindsight veto (drop candidates the failed log contradicts): R1 removes 28% / 23%, keeps 85% / 93% of winners, also
  within 'go' actions; round 2 (stronger student): removes 42%, keeps only 67% -> unsafe.
- Frozen round-1 filter on round 2: env+dist 0.675 (refit 0.692); env signals weaken, teacher preference strengthens;
  ~45% of round-2 wins are also won by the original action.
- Budgeted selection (offline): entropy-ranked states + env-best candidate find +54% (3B) / +64% (7B) winning corrections at
  1/4 of the current rollout budget; active verification simulation +77% (R1) / +94% (R2) wins per 160 rollouts vs random.

### 12.3 Why teacher corrections work (357 verified R1 states)
Wrong pickups (131): goal object absent in 125 -> student "takes something" instead of searching; teacher picks locations
with ~2x the expert prior of the student's; 1/3 of corrections are the teacher's SECOND choice; 17-23% of verified targets
are 'look' (a no-op: look raises success 11% -> 22%, a placebo/perturbation effect); student was confident on wrong pickups
(p 0.51, correction rank<=2 only 18%) but hesitant on search (p 0.18, correction rank<=2 47%).

### 12.4 Attributor variants
500-row Model A: original vs LLM-compressed step bodies (58% length): val step acc 0.890 vs 0.815, all-3 0.585 vs 0.450;
downstream OPD on 100 failures: unseen 63 vs 64 (n.s.). The 7B SFT teacher (80.6%) is worse than the 3B one (89.6%):
78% of its demos are PDDL-planner paths that walk straight to the object (privileged), vs 52% handcoded search demos.

### 12.5 7B self-evolution from an untrained Qwen2.5-7B (student = teacher with the full failed log; zero-shot 110/274)
- Collection: 300 games, 122 won; 178 failures. Six step selectors + placebo.
- Updating the untrained 7B is fragile: argmax flips on 150 own held-out states predict damage (6% flips already -13
  games; DPO lr 2e-5 x2: 60-82% flips -> 29 / 1 per 134). Selection by env evidence alone picked ~100% 'go' -> collapse.
- Calibrated recipe (veto + hindsight ranking + verb cap, DPO lr 1e-6, 1 epoch): placebo 109/274 (= zero-shot); all six
  selectors 106-115, none significant; Model A best (+6 vs placebo, 16/10).
- Rollout audit of self-proposed corrections (30 per arm, real fix = correction wins, logged action loses): Model A 30%,
  self-attribution 23%, self-trained progress attributor 23%, random-early 17%, entropy 10%, hindsight-KL 10%, placebo 10%.
  => step selection decides correction quality; the same self-teacher is 3x better at attributor-chosen steps.
- Verified-only (28 real fixes among 178 Model A states, 16%): 114/274 vs matched placebo 110 (17/13). Proxy-selected (23%
  real fixes): 115 vs 109. Pooled corrections vs placebo 33/23, p=0.23: consistent but small -> the bottleneck is the
  NUMBER of corrections, not their quality. Scale-up (1500 games) in progress.
- Scale-up (1500 games, 914 failures, Model A step, self-teacher candidates, all states rollout-verified): 141 real fixes
  (15%; 29% among states passing the env veto vs 9% among vetoed). DPO lr 1e-6, 1 epoch, 2 seeds, 274 games:
  real fixes 107 / 114 vs matched placebo 107 / 107 (pooled 30/23, p=0.41); zero-shot 110. Seed-to-seed spread of the
  same recipe ~7 games. All four matched corrections-vs-placebo comparisons pooled: 63/46 (p~0.13).
  => In round 1 from an untrained 7B, verified single-step corrections roughly cancel the update's perturbation but give no
  detectable net gain; 86% of real fixes are game-specific 'go to X' (where the object happens to be), which plausibly
  does not transfer to unseen games.
