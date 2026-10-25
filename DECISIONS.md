# DECISIONS

## 12. GRPO on the attributor: design choices, and why the first run was mostly wasted compute

Researcher asked for RL in addition to SFT ("Start to do it", 2026-09-08).
Choices:
- **SFT-then-RL, not RL-from-base (first).** Start GRPO from the SFT LoRA
  merged into Qwen2.5-3B (`reports/attributor_qwen2.5-3b-sft-merged`),
  fresh LoRA on top. With PEFT, trl's KL reference is the base with
  adapters disabled -- i.e. the SFT policy itself -- so beta=0.04 anchors
  RL to SFT rather than to the raw instruct model. (trl 1.12 default is
  beta=0.0; set explicitly.)
- **Three separate 0/1 reward functions (agent/step/mode), equal weight,
  instead of one combined scalar.** Same total reward, but trl logs each
  function's mean -> per-axis accuracy curves during training for free.
  Mode is the weak axis after SFT (82.8% val), so this is the curve that
  matters.
- **One shared parser for reward AND eval** (`error_attr/attribution/
  parse.py`, imported by both `eval_attributor.py` and the GRPO script).
  If reward and eval parsed differently, RL could learn a quirk one
  rewards and the other doesn't -- cheap to rule out by construction.
- **Verifiable reward via the official scorer**, `whowhen_eval.score`,
  against the RAW ground_truth (JSON-string column, since per-framework
  schemas differ and HF datasets would otherwise coerce them into one
  struct). Smoke-tested before launch: gold targets -> 1.0, garbage -> 0.0
  on all three axes.
- **Prompts filtered to <= 4096 tokens** (3,639/5,353) so 8 generations
  per prompt fit one 46GB L40S; completions capped at 64 tokens (the
  answer is ~18).

**Finding from the first run (step 5 log):** `frac_reward_zero_std=0.85`,
agent/step reward = 1.0 with std 0, mode = 0.91, KL ~ 1e-6, loss ~ 1e-7.
The SFT model has memorized these training prompts (mode is 0.91 on train
vs 0.83 on val), so 85% of prompt groups produce 8 identical-reward
samples -> zero group-relative advantage -> zero gradient. RL on the same
prompts SFT already fit is mostly wasted rollout compute. Kept the run
going as a baseline (~15% informative groups, all on the mode axis, is not
literally nothing), and started a second variant in parallel:
**hard-example mining** -- one offline sampling pass with the SFT model,
keep only prompts whose 8-sample group has reward variance
(`scripts/mine_grpo_hard_prompts.py`), then GRPO on those so ~100% of
groups are informative. Known caveat: mined against the t0 policy; prompts
get easier as it learns, so periodic re-mining is the natural extension.
A third arm worth running if the transfer question stays open: GRPO from
the BASE instruct model (no SFT) -- high initial variance, and it tests
"RL vs SFT" as alternative paradigms rather than "SFT then RL".

**Update, same day:** stopped the SFT-start baseline after 2 log points
(steps 5 and 10): frac_reward_zero_std went 0.85 -> 0.95 and grad_norm
0.28 -> 2.7e-05, i.e. the adapter was staying at identity. Two points were
enough to establish "SFT-start GRPO on memorized prompts has ~no signal";
190 more steps of it would only have burned the L40S. Log kept at
`reports/attributor_grpo_sftstart_baseline.log`. Freed GPU went to the
base-model-start arm (`--model Qwen/Qwen2.5-3B-Instruct`, output
`reports/attributor_qwen2.5-3b-grpo-basestart`), run in parallel with
hard-example mining on the A40. Completion cap stays 64 tokens: the base
model will also emit the 4th "Reason:" line the prompt asks for, which gets
truncated, but the 3 scored fields come first and the parser doesn't need
Reason -- truncated completions are still scored (mask_truncated_completions
is False).

## 13. Base-model GRPO v1: mode reward exactly 0.0 -> parser strictness, not model failure

First base-start GRPO log: agent 0.94, step 0.54, **mode 0.00**, 99.4% of
completions clipped at 64 tokens. Same diagnostic instinct as entry 10 --
an exact 0.0 on one axis is a grading-path signal, not a "model is bad"
signal. The logged completions showed the un-fine-tuned model writing
`Error Mode: A.3: Premature Termination` (code + name, sometimes bold);
the shared parser passed the whole field through and the scorer's exact
normalized compare against `A.3` failed every time. The official harness's
own parser (`whowhen_eval/parse.py`) is lenient here -- it extracts the
first taxonomy-shaped code `[A-Z]+\.\d+` from the field. Mirrored that in
`error_attr/attribution/parse.py` (`_extract_mode`). Because the parser is
shared, this changes reward AND eval identically. Checked before relaunch:
gold targets 300/300; re-scoring the saved SFT val predictions with the new
parser reproduces 0.976/0.956/0.828/0.816 exactly, so nothing already
reported moves (SFT emits bare codes). Also raised the base arm's
completion cap 64 -> 160 (`--max_completion_length`): the base model
rambles inside the Step field ("not applicable (The task is to ...)") and
appends the Reason line, so the Error Mode line was at risk of truncation.
v1 log kept: `reports/attributor_grpo_basestart_train_v1_strictparser.log`.
Note the v1 zero-std fraction was 0.5 (vs 0.85-0.95 for SFT-start) -- the
base arm does have real learning signal, which is why it was worth fixing
rather than dropping.

**v2 follow-up (steps 5-25):** mode reward swung 0.19 -> 0.31 -> 0.06 ->
0.00 -> 0.16 while mean completion length and clipping crept up
(122 -> 129 tokens, 15% -> 23%). Hypothesis: outputs lengthening and the
Error Mode line getting truncated. Checked directly against trl's dumped
rollouts (`<output_dir>/completions/*.parquet`, which also carry per-sample
rewards -- far better than parsing the rich console table): the Error Mode
line is present and parses in ~100% of completions in every batch, so the
swings are real per-batch variance (each log = 4 prompts; batch 20 was
simply hard -- its step reward also fell to 0.19). Hypothesis rejected, no
intervention. Kept the `--mask_truncated_completions` flag as an available
contingency. Lesson: with 4 prompts per optimizer step, single-log reward
values are too noisy to act on; judge trends over >= 5 logs or via the
checkpoint evals.

## 14. The feedback study's corrector was never the erring agent -- fix via Qwen self-play

Researcher's objection (2026-09-08): the failed trajectories are Who&When
Pro's (other systems, GPT-era backends), so Model B (Qwen-7B) was a third
party reading someone else's transcript, and "Your team's answer was
incorrect" was a fiction. Accepted: the study measures the information
value of feedback types for an EXTERNAL reader's correction, not
self-correction or collaboration dynamics -- and a mundane reading of
"why > where" exists (a step pointer into someone else's transcript is
nearly uninformative). Docs relabeled accordingly. Model A is unaffected
(cross-agent transfer is its intended setting).

Fix design, built so that EITHER route the researcher picks can start
immediately:
- **Debate as the self-play protocol.** The only Who&When Pro multi-agent
  framework reproducible with pure Qwen (no Python execution like mathchat,
  no browser like magentic-one). Runner emits the official `debate` schema
  so the renderer -> attributor -> feedback pipeline is reused unchanged
  (verified on a hand-built trajectory).
- **Same task pool as the external study** (862 unique tasks with gold
  answers, pulled from the 4 gradable frameworks' rows) so the two studies
  are directly comparable.
- **Debaters = corrector = Qwen2.5-7B-Instruct**, so the corrected system
  is the system that erred.
- **Route A** (natural failures, labels from Model A's predictions) is
  recommended first: cheapest, and it is the first true end-to-end use of
  the attributor. **Route B** (Who&When-style error injection into Qwen's
  own successful runs) gives gold labels but needs an injection step; the
  same collection run supplies its successful trajectories.
- **Added an `incorrect_only` control** ("your answer was incorrect",
  nothing else) to the 4 conditions: with genuine self-correction, this
  separates "any feedback" from "which feedback type" -- a question the
  external-transcript design could not pose.
- A third GPU (A100-40GB, `a100-4` partition -- no per-account job cap) was
  requested and used for collection rather than waiting hours for the L40S
  /A40.

## 15. Known artifact: Model A always predicts position .0 for round-scored frameworks

Seen in the self-play chain pilot: Model A's step predictions on debate
traces were only `0.0` / `1.0`, never `.1`/`.2`. Cause is my SFT target
construction (sft_format.py): for debate/dylan the official scorer is
round-only, and the brute-force scan keeps the FIRST coordinate that scores
correct -- always position 0 -- so the model learned "R.0" regardless of
which agent it names. Scoring is unaffected (round-only). But the
`precise_*` feedback text can become internally inconsistent ("step 0.0,
in agent_2's turn" when agent_2 sits at 0.1). Fixed downstream in the
chain (`consistent_step`: re-derive the coordinate from the predicted round
+ the named agent's real position). The upstream fix -- use the gold
agent's own turn coordinate as the SFT target for round-scored frameworks
-- requires re-running data prep and retraining SFT; deferred, noted.

Pilot's other lesson (n=40 GPQA, route A): every condition changed the
model's own wrong answer only 30-42% of the time, incl. the no-info
control, while correct-given-changed was 42-47% (> 33% MC re-guess). On
self-generated failures the bottleneck looks like willingness to revise,
not what to revise to -- a mechanism the external-reader design could not
see. The chain now reports changed / correct-given-changed per condition.

## 16. Stale stored results: after fixing a scorer, rewrite the artifact, don't just re-score in memory

Follow-up to entry 10. After the ground_truth scoring fix I recomputed the
SFT numbers in a one-off script and reported them, but left
`eval_val_finetuned.jsonl`'s stored `result` fields as the buggy originals.
Hours later a per-framework comparison read those fields and silently
produced SFT step = 0.00 for mathchat/debate/dylan/metagpt again -- the
same impossible zero as the original bug, which is what caught it. Fixed by
re-scoring both SFT eval files in place (backups kept as
`*.prefix_bug_backup`); totals unchanged (val 0.816, transfer 0.203).
Rule: a scorer fix must be applied to every stored artifact that downstream
analysis might read, not only to the number in the chat.

## 17. Never run multi-hour training on `preempt-gpu`

Arm 3 (SFT-start GRPO on hard-mined prompts) was launched on the A40
allocation in `preempt-gpu` because `interactive-gpu` caps this account at
one job. It was preempted at step 40 of 250 (~21:01), before the first
`save_steps=50` checkpoint -- nothing saved, ~1.5h of A40 time lost, and
Slurm requeued the hold job as a useless `sleep`. The sibling project's
memory already warns that `preempt-gpu` "can be preempted by higher-
priority jobs"; the mining pass and short evals got away with it, the
9-hour training run did not. Rules going forward: (1) multi-hour training
only on non-preemptible partitions -- `interactive-gpu` (1 job) or
`a100-4`/`msigpu` (no per-account cap); `preempt-gpu` only for short,
idempotent/resumable jobs (mining, evals). (2) `--save_steps 25` (new CLI
flag) on anything that could be killed. Relaunched arm 3 on a fresh
`a100-4` allocation with save_steps=25.

## 18. Syntax-check scripts with `py_compile`, not `ast.parse`

`ast.parse` passed a chain-script edit that `python` refused to compile
(`global OUT_DIR` declared after `OUT_DIR` was already referenced in the
same function). The symptom was silent: `--help` printed nothing, and a
grep for the new flags simply came back empty -- easy to misread as a
grep problem. `python -m py_compile` catches this class (symtable-level
errors) and costs the same. All recent scripts re-checked with it.

## 19. Arm 2 verdict: GRPO from the base model does not transfer; SFT does

200 GRPO steps from Qwen2.5-3B-Instruct (0.22 epochs, verifiable reward
from the official scorer) moved in-distribution step localization
(val 0.395 -> 0.575, p=5e-19 paired vs untuned) and nothing else -- mode
stayed at the untuned 0.17 (p=1) -- and on the held-out ALFWorld transfer
set it is statistically identical to the untuned base on every axis
(step 0.064 vs 0.084; mode 0.206 vs 0.186; all-3 0.019 vs 0.029). SFT
transfers far better (step 0.534, all-3 0.203). Mid-run checkpoints (50,
100) told the same story early: training reward rose while held-out
accuracy stayed flat. Reading: with ~800 prompts seen and a 0.04 KL anchor
to the base, RL sharpened the base model's existing tendency to point at
plausible steps but did not teach the 17-mode taxonomy -- SFT's dense
token-level supervision on 5,353 examples did. Not extending this arm;
the remaining RL question is arm 3 (SFT-start on hard-mined prompts),
where the signal is concentrated on mode.

## 20. Arm 3 (SFT-start GRPO on hard-mined prompts): degrades held-out mode via a mining selection effect

Hard mining did what it was for -- 80-90% informative groups vs 5-15%,
real gradients, training mode reward 0.56 -> 0.69 -- yet held-out mode
accuracy fell monotonically: SFT 0.828 -> ckpt-25 0.818 -> ckpt-50 0.809,
with **0 held-out mode errors fixed and 11 broken** (p=0.001). Mechanism,
from the 11 broken rows: no distribution collapse; predictions drift
within the reasoning family toward R.3 (numerical) / R.2 (local reasoning)
and away from R.1 (hallucination) / R.4 (task misunderstanding). Cause is a
selection effect of mining: prompts where the SFT policy is uncertain
over-represent R.3-labeled mathchat cases (R.3 is 43% of mathchat hard
prompts vs 33% of all mathchat), and mathchat is 49% of the hard set --
so the reward taught a framework prior ("mathchat -> R.3") rather than
mode discrimination. Together with arm 2 (RL-from-base never learned mode
at all), the consistent finding is: **at these scales the 17-way mode
taxonomy is learned by SFT's dense supervision, not by a sparse 0/1 GRPO
reward, and RL on top of SFT only erodes it.** Step and agent are
unaffected (RL neither helps nor hurts them here). Checkpoint-100 was the pre-declared
early-stop decision point: mode 0.811 (2 fixed / 12 broken vs SFT,
p=0.013), all-3 0.799 -- plateaued at the degraded level, no recovery
from ckpt-50. Stopped the run there (~step 115 of 250; ~6h of A100
saved). ALFWorld transfer evaluated on ckpt-25 (identical to SFT) and
ckpt-100 (for the record). Not tried (future work, if RL on the
attributor is revisited): mode-balanced hard mining (cap per gold mode),
or a mode-specific reward weight -- both target the prior-learning
failure directly.

## 21. Self-correction protocol: `debate_round` (same multi-agent system continues), 150 first

Researcher's call after the pilots: `one_shot` gives the feedback to the
same MODEL but not the same multi-agent PROCESS (a single Qwen-7B reads the
transcript and emits one corrected answer); `debate_round` injects the
feedback as an extra round to the same three agents, who revise and vote
-- i.e. the erring system itself continues, which is what the original
collaboration-dynamics question needs. Pilot hint (n=8): answer-change
rates 37-75% under debate_round vs 30-42% under one_shot. Decision: run
debate_round on a seeded random 150-trajectory sample of the reasoning-
benchmark failures first (5 conditions incl. the no-info control), drawn
from the COMPLETE self-play pool so scibench/mmlu_pro tails are
represented; extend if the signal warrants. one_shot on the full pool is
deprioritized.

## 22. Self-correction under `one_shot` is a null: anchoring dominates, not information

Full-pool result (361 self-generated failures): 3.9-5.5% recovery in every
condition, none beating the label-free "your answer was incorrect"
control; the external-reader "why >> where" effect is absent (sign
reversed, n.s.). The team keeps its wrong answer ~70% of the time
regardless of feedback, and lands right only 13-18% of the time when it
changes (below the 33% chance of merely switching MC options on gpqa).
Two consequences: (1) the researcher's objection is now a measured
result -- the earlier study's effect was a property of external readers,
not of self-correcting teams; (2) route B (gold labels via injection) is
deprioritized: if even the no-label control matches every condition,
label quality is not the binding constraint. The `debate_round` protocol
(researcher's preferred; 150-sample running) is the remaining test, and
the mechanistically plausible one -- its pilot roughly doubled the
answer-change rate. If it also nulls, the honest conclusion is that this
7B team's self-correction is anchoring-bound and feedback TYPE is not the
lever; the lever would be the revision mechanism (e.g. forcing
re-derivation, or a stronger/other model as corrector).

## 23. Input-ablation results decide the item-3 retrain ladder; 32B judge deferred (2026-09-09)

Inference-time ablations (TODO 2b, 20/20, tables in STATUS.md) answered
the advisor's "do we really need the full trace?" with a split verdict:
in-distribution, compression costs accuracy (first_line 10% chars: -40pp
zero-shot; extractive 65%: -19pp mode, because it drops the prose lines
that carry the "why"); on the held-out ALFWorld, dropping observations or
first-line compression IMPROVES step localization (+6-7pp) -- the model
never learned to weight ALFWorld's room descriptions, so they are OOD
noise. Decisions that follow:
- **Retrain ladder = first_line (running) + extractive + no_observations**,
  in that launch order. Extractive first because it is the moderate-
  compression point with the largest mode penalty: a retrain that recovers
  it proves the penalty was format shift, not information loss. The
  no_observations retrain is the transfer-gap test: if training-time
  removal of observations preserves val and lifts ALFWorld above 0.534
  step / 0.203 all-3, it is the first concrete lever on RQ3 we have.
- **Confound to control:** compressed data keeps more rows under 8,192
  tokens (first_line 5,353; extractive 4,953; full 4,767). Any variant
  that matches/beats full is re-run with `--keep_ids` on the full run's
  4,767 ids before the result is attributed to the input format.
- **Prefix/suffix results are not a retrain target.** Step accuracy equals
  the gold-visibility cap in all four cells (val: 24%/69% visible; ALFWorld:
  82%/18%), so the finding is about WHERE errors occur (late in
  debate-style traces, early in embodied ones), not about the model. It
  goes into the write-up as the answer to "does attribution need the
  completed failure?" -- framework-dependent, and predictable from error
  position -- with no further compute.
- **32B LLM-judge deferred.** 7B untuned judge: all-3 9.6% (step 0.626,
  mode 0.160) vs 3B untuned 4.9% and SFT-3B 81.6%. Scale 3B->7B bought
  step but left mode flat, so a 32B point is unlikely to change the
  conclusion ("the 17-way taxonomy is learned from supervision, not from
  the prompt"), and it does not fit an L40S in bf16 beside the running
  chain. Revisit only if a reviewer asks.

## 24. `debate_round` also nulls; feedback type and protocol are both non-levers (2026-09-09)

The researcher's preferred protocol (feedback as an extra round to the same
three agents), 150 seeded failures, 5 conditions + 3 wrong-content
controls = 8 x 150 paired cells. Recovery 1.3-4.0% everywhere; no
condition beats the no-info control; wrong location / placebo reason /
random attribution are indistinguishable from the correct versions; and
debate_round is indistinguishable from one_shot on the same traces (it
raises the answer-change rate by ~4pp without converting any of it).
What this rules out and what it leaves:
- Ruled out as levers for THIS team: feedback precision (where), feedback
  content (why), their combination, and the injection protocol. The
  external-reader result ("why >> where", n=2,328, p<1e-8) is therefore a
  fact about a fresh reader given hints, not about a team revising itself.
- Not ruled out: the revision mechanism. The team sees its own previous
  answer in context and keeps it ~68% of the time regardless of what it is
  told; when it does change, it is right 4-10% of the time (below MC
  re-guess chance). The two backlog experiments that test this directly
  are (a) hide the team's previous answer / force re-derivation, and (b)
  give the same feedback to a fresh instance or a different (stronger)
  corrector. Either would separate "the information is useless" from "the
  recipient cannot act on it". Recommended next on the now-idle L40S.
- Route B (gold labels via error injection into Qwen's successes) stays
  deprioritized: label quality cannot be the binding constraint when random
  labels perform identically to correct ones.
- Do not scale this null beyond 150: 8 conditions x 150 paired traces
  already puts every pairwise difference at 0-3 discordant traces; more of
  the same design will not change the conclusion.

## 25. Answer-parser bug (nested braces) found via the mathchat collection; scope and repair (2026-09-09)

Symptom: mathchat self-play math accuracy 31/101 looked low; the failure
list showed finals like `\frac{512` vs gold `\frac{1}{64}`. Cause: the
shared `\boxed{}` regex `\\boxed\{([^}]*)\}` stops at the first `}`, so any
nested-brace answer (fractions, roots) is truncated and graded wrong. The
same regex lived in run_feedback_study.py (external-reader study) and
debate_runner.parse_task_answer (self-play collection, chain grading, AND
the debate_round protocol, which re-wraps the parsed vote as `\boxed{...}`
at generation time). Fixed with brace matching in both places.

Measured impact (re-grading stored raw text, CPU only):
- External-reader study (n=2,328 x 4): every condition +~1pp (e.g.
  reason_only 32.4 -> 33.5%, precise_only 29.6 -> 30.5%); all four McNemar
  conclusions unchanged (p = 5.8e-9 / 2.1e-4 / 3.3e-9 / 0.91).
- Debate self-play pool: 14/862 math trajectories were FALSE failures
  (actually correct); 8 of them sit in the 150-trace sample.
- one_shot_full (raw responses stored): recovery rises uniformly from
  4-5.5% to 7-9% per condition -- the ordering/null is unchanged (the
  gain is the false failures re-answering correctly).
- debate_round runs (main / controls / framing): 24/150 traces (all math)
  have nested-brace golds; for those the protocol truncated the vote at
  generation time, so NO recovery could ever register -- and the stored
  raw cannot be re-graded. Paired comparisons among conditions remain
  valid (same 24 traces in every cell), but absolute rates are floors.
Repair: `--ids_file` added to the chain; the 24 traces will be re-run for
all 11 debate_round conditions with the fixed parser on the next free GPU
and spliced in with scripts/merge_corrections.py; the 8 false failures are
then reported separately (they are not failures). Collections started
before the fix (debate; the running mathchat job) are re-graded from
stored outputs, not re-collected. Lesson (same family as #10): a low number
on a slice is only a bug signal once you look at the raw strings.

## 26. Grading was exact-string; switched to exact-OR-numeric-within-1% (2026-09-09)

Trigger: mathchat self-play scibench at 1/32 vs debate's 35/225 on the same
pool. The raw pairs were things like `87.79` vs gold `+87.8`, `0.5` vs
`\frac{1}{2}`, `-0.710 \text{ K/atm}` vs `0.71`. `is_correct` compared
strings after stripping non-alphanumerics, so any extra digit, sign glyph,
LaTeX form or unit was a "failure". This penalised scibench (numeric,
arbitrary precision) most, and mathchat (whose framing prompt demands
decimals) more than debate. New rule in `debate_runner.is_correct`, shared by
collection, chain and (now imported) run_feedback_study: exact match after
normalization OR both sides parse as numbers (fractions, a x 10^b, units
stripped) and agree within 1% relative (SciBench's own rule is 5%).

Re-grading everything from stored text (no GPU):
- Debate self-play pool: correct 208 -> 294 / 862 (scibench 35 -> 84, math
  69 -> 90, simpleqa 8 -> 22). In the 150-trace sample, 32 were FALSE
  failures (20 scibench, 12 math) -> 118 true failures.
- Every debate_round cell, restricted to the 118 true failures: 4-8/118
  (3.4-6.8%) for all 11 conditions, wrong_location (8) and the two
  expert-framed conditions (8) at the top -- the null and the absence of
  any ordering are unchanged. one_shot_full on 291 true failures: 6-9%.
- External-reader study: every condition +~11pp (reason_only 32.4 -> 43.6%,
  precise_only 29.6 -> 40.5%); dylan 17 -> 52% (it is scibench-heavy). All
  four McNemar conclusions unchanged (p = 1.7e-9 / 4.2e-3 / 1.3e-9 / 0.92);
  same for the predicted-label rerun (n=232).
Consequences: (1) absolute numbers in STATUS/docs are being restated on true
failures; (2) the debate 150-set stays as drawn (paired design), with the 32
false failures excluded from analysis; (3) mathchat/magentic pools are
re-graded with the new rule BEFORE their 150-samples are drawn, so the three
frameworks are compared under one grader; (4) `scripts/regrade_selfplay.py`
rewrites `final_correct` in a pool file and keeps the old label as
`final_correct_v1` for the record.

## 27. Compression-ladder verdict: drop observations at training time; never drop agent prose (2026-09-09)

Three retrains of Model A on compressed inputs, same recipe, evaluated on
their own prompts, paired McNemar vs the full-trace SFT:
- no_observations (0.75 of chars): val identical (0.815 vs 0.816, p=1);
  ALFWorld step 0.534 -> 0.643 (51 vs 17 discordant, p=4.5e-5), all-3
  0.203 -> 0.228 (p=0.17). The zero-shot ablation predicted this (+7pp
  step) and training on the format delivered more (+11pp).
- extractive (0.65): val -9pp all-3 (p=1e-8), nearly all mode; ALFWorld
  -7pp all-3 (p=5e-4), step 0.37. Half the zero-shot penalty was format,
  half is the discarded prose.
- first_line (0.10): val -32pp; ALFWorld collapses to positional priors
  (step "1" for 82% of traces). Real information loss + destroyed transfer.
Decision: the attributor's canonical input becomes "agents' own text, no
environment/tool observations". The `--keep_ids` control (same 4,767
training ids as the full SFT) REPRODUCED the gain (08:50): val 0.820 vs
0.816 (p=0.89); ALFWorld step 0.627 vs 0.534 (46/17, p=3.4e-4); identical
to the 5,016-row run (p=0.53). So the effect is the input format, not the
extra rows the shorter prompts let under the 8,192 cap. Reasons: (1) it is the first intervention that moves the RQ3 transfer
gap, and it moves the axis (step) that transfer had preserved best; (2)
it costs nothing in-distribution; (3) it shortens prompts 25%, which also
raises the row count under the length cap. What NOT to do: any compression
that filters agent prose by surface features -- mode classification lives
in those lines. The advisor's "do we really need the full trace?" thus gets
a precise answer: we need the full AGENT trace; we do not need the
environment's half of the conversation, and out of distribution it hurts.

## 28. The one lever that moved self-correction is authority framing, not diagnostic content (2026-09-09)

Researcher's question ("would saying it is an expert diagnosis make them
change their minds more?") tested as three conditions on the 150-set, then
replicated on the 173 remaining fresh debate true failures, pooled n=291
paired: "an independent expert reviewer has verified your answer is WRONG"
raises recovery 6.5 -> 10.7% with NO diagnosis attached (16/4 discordant,
p=0.012) and 6.2 -> 10.0% with the full precise_reason diagnosis attached
(14/3, p=0.013); the diagnosis adds nothing on top of the authority (9/11,
p=0.82) and nothing on its own (10/11, p=1). Answer-change rate rises from
~0.37 to ~0.50. Decisions:
- Treat this as the study's headline about the RECIPIENT: what changes
  behaviour is how confidently it is told it is wrong, not what it is told
  about the error. This is the mechanistic complement to DECISIONS #22/#24
  (anchoring-bound) -- a way to loosen the anchor exists, and it is
  social/presentational, not informational.
- Do not over-claim: two pairs each p~0.012 at n=291, found after a
  borderline result on the first 118 (the replication was pre-specified on
  disjoint traces, so it is not a garden-of-forking-paths result, but it is
  still one team on one model). Cross-framework replication (expert_* on
  the mathchat and magentic-one 150-sets) is running and decides whether
  this becomes a claim or a footnote.
- **Cross-framework replication outcome (same day, 06:55):** mathchat
  +3pp (8/3, p=0.23), magentic-one +4pp (9/3, p=0.15) for authority without
  content -- same sign and size as debate, each underpowered at n=150.
  Pooled over 591 paired traces: authority-no-content vs neutral 33/10,
  p=0.0006; authority+content vs content 29/13, p=0.02; content on top of
  authority 16/23, p=0.34; content alone 21/21, p=1. The framing effect is
  now a claim (three topologies, one direction, p<0.001 pooled); the content
  null is unanimous. Note the replication was decided before seeing the
  mathchat/magentic main-chain results and run on their pre-drawn 150-sets.
- **Scale-up outcome (19:10):** all remaining fresh true failures added (mathchat
  150 -> 326, magentic 150 -> 415; total 1,032 paired traces). Authority-no-content vs
  neutral: 60/22, p=3.2e-5 (debate 15/4, mathchat 21/6, magentic 24/12); authority +
  content vs content: 47/25, p=0.013; content on top of authority 30/40 (p=0.28,
  negative direction); content alone 41/35 (p=0.57). Claim status: the framing effect
  is a robust, replicated finding across three topologies; the content null is
  unanimous at n>1,000. Wording for the paper: "a 7B team's recovery depends on how
  authoritatively it is told it is wrong, not on what it is told about the error".
- Implication for the research plan: for a 7B team, Model A's value is not
  as a feedback CONTENT source for self-correction; it remains valuable as
  an external judge / training-signal source (RQ1-3), where its accuracy
  matters. The self-correction track's next experiments should manipulate
  the recipient (fresh instance, stronger corrector, hidden previous
  answer), not the message.

## 29. Scale x input-format 2x2 for the attributor: both matter OOD, they stack, neither matters in-distribution (2026-09-10)

Researcher asked to test the no_observations result on a larger model. Ran the
full 2x2 (Qwen2.5-3B/7B x full-trace/no_observations) on identical 4,767
training rows. ALFWorld all-3: 20.3 / 23.2 / 32.8 / 38.9. Scale alone +12.5pp
(p=4e-8); removing observations at 7B +6.1pp (p=9e-4) -- larger than at 3B
(+2.9pp, p=0.15) and now significant on mode as well (23/5, p=9e-4). val: all
four within 2pp (scale p=1; format at 7B p=0.09). Decisions:
- Canonical Model A becomes the 7B trained on agent-text-only input. It is
  the same in-distribution as everything else (~82-84%) and far better OOD
  (step 91%, mode 40%, all-3 39% on ALFWorld). Downstream uses (chain
  feedback, OPD attribution weights) should switch to it when re-run; the
  feedback-side conclusions will not change (Model A's labels were already
  ~= gold for the recipients that ignore them), but OPD's W will be sharper.
- The observation-removal effect is a property of the task, not a crutch for
  small models: it grows with capacity. Hypothesis for the write-up: a
  stronger model extracts more from agent prose OOD, and observations
  compete for attention it could spend there; removing them frees that
  capacity. Testable at 14B if a 2-GPU allocation is available.
- Mode is now the limiting axis OOD (40%) and the only axis where scale
  helps in-distribution (+2pp, borderline). If any further training budget
  goes to Model A, it goes to mode: more frameworks, mode-balanced data, or
  a 14B -- not to step, which is solved (96% in, 91% out).

## 30. Three training paradigms for the attributor: SFT works, GRPO and OPD do not move it (2026-09-10)

Researcher asked for OPD on Model A after SFT and GRPO. Ran GKD-style on-policy
distillation 7B -> 3B (same agent-text-only format, in-distribution prompts, no
labels, half epoch, uniform and mode-weighted token weights). Result: identical
to the student's SFT init on val (0.809/0.804 vs 0.820, n.s.) and on ALFWorld
(0.225/0.222 vs 0.232, n.s.) while the teacher is at 0.389. Why: per-token KL
between student and teacher on training prompts is ~0.01 nats -- both fit the
training set, so on-policy samples in-distribution carry no disagreement to
learn from; the 7B's OOD advantage is invisible there. Decisions:
- Close the attributor-training track at SFT. Across GRPO (3 arms) and OPD (2
  arms), nothing beat SFT; what improved Model A was data (agent-text-only
  input) and scale (7B). Spend future Model-A effort on data/scale, not on the
  objective.
- If OPD is ever revisited, the precondition is a prompt distribution where
  student and teacher disagree: perturbed/synthetic traces or held-out-style
  frameworks. Distilling where they agree is provably a no-op.
- The policy-side Attr-OPD (item 6) was a misreading of the request and is
  filed as a by-product; its conclusion (sign set by teacher correctness at the
  attributed step) stands and motivates the rollback idea in the AGR proposal.

## 31. The external-reader "why >> where" result, re-read against its missing baselines (2026-09-15)

The 2026-09-08 study had four conditions and no zero. Researcher asked for the
zeros; ran task_only / no_feedback / incorrect_only on the same 2,328 traces.
Result: 14.3 -> 28.5 -> 42.4%, then reason_only 43.6 (+1.2, p=0.017), precise_reason
43.5 (+1.1, p=0.04), vague 41.8 (n.s.), precise_only 40.5 (**-1.9, p=3e-5**).
Decisions:
- Restate the claim everywhere as: "for an external reader, the outcome label is
  the active ingredient (+14pp); the error type adds a small, significant +1pp;
  a bare step pointer without a reason is harmful (-2pp)". Drop "diagnostic
  content drives correction" as the headline -- it was 1pp riding on a 14pp
  baseline we had not measured.
- Methodological lesson for the paper: every feedback condition needs the "told
  it is wrong, nothing else" zero; without it, any content condition inherits
  the outcome label's effect. The self-play chain had this control from the
  start, which is why its null was legible immediately.
- It unifies the two studies: reader and team both respond to the outcome
  signal, not the diagnosis; the team additionally needs authority to act on it.
  Model A's value therefore lies in (a) judging/attributing for training signals
  and (b) choosing WHERE to intervene (rollback), not in what to tell anyone.
- The precise_only harm is the strongest single argument for the AGR design: a
  location pointer handed to a reader is a distraction, but a location used to
  truncate and regenerate never has to be interpreted by the reader.

## 32. From rollback to Attr-GRPO: attribution as credit assignment, and how the earlier GRPO failure is avoided (2026-09-18)

Researcher's direction: efficiency; the rollback/cut-point plan was judged too heavy;
"only train the specified step" preferred. Decision: Attr-GRPO -- standard RLVR on the
debate team (G team rollouts per task, 0/1 outcome reward, group-normalised advantage,
KL to reference) with ONE change: for failed rollouts the gradient lands only on the
turn Model A blames. Controls: uniform (standard GRPO), random turn, last round.
Why the attributor-GRPO failure (#12-20) does not carry over: that was a saturated
SFT policy (85-95% zero-variance groups) on a sparse 17-way label reward with a
mining selection bias; here the policy is far from saturated (train acc 32-43%), the
reward is answer correctness (the standard RLVR regime, where arm 2 already showed
GRPO learns when groups have variance), and the mask changes where gradient lands,
not the reward, so it cannot create shortcuts. What DOES carry over: zero-variance
groups, now from the hard side (62% of train tasks are all-wrong for the 7B team by a
within-team proxy; 14%/22% mixed for 7B/3B). Mitigation: dynamic sampling (drop
zero-variance groups, resample to >=4 informative), T=1 with G=8, a measured
group-variance probe BEFORE committing days of compute (queued), optional curriculum
on pass-rate 0.1-0.9, and as a method-specific fallback a small-weight negative
gradient on the attributed step for all-wrong groups (off by default). Also decided:
Attr-Distill with a single strong-teacher call per failure is the cheap complement
if a frontier API or a multi-GPU 32B/72B teacher is available -- researcher to decide.

## 33. Step-pointer exactness is irrelevant to every message recipient (2026-09-18)

Researcher's question: is x vs x+2 vs x-2 vs no step different, for the erring MAS and
for a third-party reader, on long agentic traces? Answer: no, for both. Model B: magentic
x-2 vs x 0/0 discordant on 50 traces, debate offsets within noise, mathchat too short.
External reader (434 gold-step traces): x+2 vs x 16/16 (p=1), x-2 vs x 11/10, all pointers
vs none p >= 0.31, no dependence on trace length (5-7 / 8-11 / 12+ steps) or framework.
Decisions:
- Stop investing in location precision for message-based feedback; report it as a
  closed question with the "why" effect (+1-3pp, small but replicated) as the only
  content signal for readers.
- For the efficiency framing this is good news: a coarse or upper-bound step index is
  free to use wherever the location is only communicated. The open question moves
  entirely to mechanical uses of the location -- replay cut points and, per the
  researcher's choice, Attr-GRPO (gradient localised at the attributed turn) -- where
  the random-turn control will show whether precision matters there.
- Note for the paper: precise_only < none on the full 2,328 (-1.9pp) but = none on the
  434 long-trace subset; the small harm of bare pointers is not about being wrong (off-
  by-two pointers are identical), so it is best described as "a pointer without a
  reason distracts some readers" rather than "misdirection".

## 34. Attr-GRPO run protocol: equal-compute arms, sampled primary metric (2026-09-18)

- All four arms get the identical configuration and seed (G=8, 8-16 tasks/step, 48 steps,
  beta 0.04, lr 2e-5); the arms differ ONLY in the turn mask on failed rollouts. Rollout
  cost is identical across arms by construction, so "equal compute" holds without tuning.
- 48 steps is a budget decision, not a convergence claim: one step is ~16 min on an A100-40
  (rollouts dominate), the holds have ~22 h, and every arm must finish inside its hold.
  If curves are still rising at step 48 the runs are resumable from the saved adapters.
- Primary metric = mean@4 at T=1 on all 238 test tasks with per-task records (paired
  Wilcoxon vs base / vs uniform); greedy accuracy is reported but secondary, because greedy
  decoding on this stack depends on batch composition (2/8, 4/8, 3/8 on the same 8 tasks).
- Base reference is computed by the same code path (attr arm, `--full_eval_at_start`) so
  that base vs arm comparisons are not confounded by decoding differences.
- Rejected: pre-filtering train tasks by pass rate (would cost ~8 h of rollouts) and
  merging/unmerging LoRA for faster generation (bf16 drift over 48 steps).

## 35. Attr-GRPO follow-ups chosen from the mechanism, not from the scoreboard (2026-09-19)

Mid-run diagnostics showed the masked arms (attr / random / last) losing accuracy on greedy
decoding through MORE no-answer teams while uniform GRPO gained through FEWER, and the
attr_committed arm's outputs lengthening step by step (52% -> 74% of turns at the cap). The
mechanism: in a 6-turn team rollout with a 0/1 outcome reward, uniform GRPO's negative
gradient on every turn of a failed rollout is also the only length regulariser; a hard mask
removes 5/6 of it, so the positive gradient from (long) successful rollouts wins and outputs
run past the budget. Decisions: (1) do not fix this by raising the token cap alone (it
relaxes the symptom, 2x cost); (2) add soft arms that keep a uniform floor and redistribute
the SAME total budget towards the blamed turn (attr_soft), with a random-turn twin
(random_soft) so that any gain can be attributed to location rather than to the floor;
(3) keep attr_committed running as the hard-mask-on-committed-failures reference; (4) the
primary metric stays mean@4 on the 238 test tasks; greedy numbers are reported with the
no-answer share because greedy repetition loops dominate them for this 3B team.

## 36. Why every location-based use of attribution came out null: the failures are not localised (2026-09-20)

Diagnostic: per-round, per-agent correctness on all self-play debate teams. 88-92% of failed
teams had all three debaters wrong already in round 0; only 0.5-2.3% had a correct majority that
was lost in round 1. Attribution ("which step caused the failure") presupposes a decisive step;
in a capability failure there is none, so any consumer of the location -- a reader, the erring
team, or a gradient mask -- receives noise. This single fact explains the whole chain of nulls
(precise_only <= none for readers, Model B ignores pointers, x+2 = x, attr = random in Attr-GRPO)
without appealing to the recipients' inability to use feedback. Decisions:
- Report it as the paper's organising finding: attribution quality is learnable and transfers
  (Model A), but its VALUE is conditional on failure localisability, which must be measured, not
  assumed. Who&When Pro is localisable by construction (annotators picked a decisive step);
  self-play debate on hard STEM is not.
- Next algorithmic step is triage, not masking: a failure is first classified as localised vs
  diffuse (Model A with an abstain/"no decisive step" class, trainable from self-play data where
  the label is computable), then localised failures get step-targeted treatment (message, replay
  from the step, or masked gradient) and diffuse failures get capability supervision (teacher /
  OPD / uniform RL). The efficiency argument survives in this form: spend the expensive targeted
  machinery only where a step exists.
- The right test bed for step-targeted methods is one where failures ARE localised: induced
  single-agent errors (the peer-induced-error setting), heterogeneous teams (one weak agent),
  or tool-using pipelines with a single failing call -- not symmetric debate on hard problems.
- Before any further RL comparison, the substrate must first show a moving uniform baseline
  (format/termination reward or 1,024-token cap, more steps); a null uniform arm makes every
  contrast uninformative.

## 37. Pivot the training side to a long-horizon single-agent harness with step-localised Attr-OPD (2026-09-20)

Researcher's direction (2026-09-20 01:00): the debate framework is unsuitable (short, symmetric,
non-localisable failures); use a single-agent harness with long trajectories and combine error
attribution with OPD -- at the attributed step, explore alternative actions under the guidance
of a stronger model. Decision: this is the research plan's Method I (Sec. 10 Attribution-Guided
OPD) plus its Sec. 14 verified-replay variant, so we return to the plan's benchmark A, ALFWorld,
where the substrate already exists in this repo (ReAct policy over admissible commands with
per-action log-probs = the OPD q_S/q_T; PDDL-planner oracle t*; 3,553 train games load) and where
Model A already transfers (7B agent-text-only: 91.3% step accuracy on the held-out ALFWorld
split). Why ALFWorld first: (1) failures there ARE localisable by construction -- the oracle t*
is the first action after which the certified planner cannot finish, so the localisability
question that killed the debate experiments has a ground-truth answer per trajectory; (2) the
action space is a small discrete admissible set, so the teacher/student KL is exact and cheap
(no full-vocab token KL, no length drift); (3) replay to t* is a deterministic re-execution of
a text prefix, so "explore alternatives at the wrong step" is affordable and environment-
verified. WebShop is the planned second benchmark (via AgentGym or verl-agent environment
servers). Order of work: A. student failure collection + Model A vs oracle on OUR student's
failures (localisation accuracy, localisability stats, where t* falls); B. step-localised,
verified Attr-OPD with a privileged stronger teacher, arms = Model-A step / oracle step /
random step / uniform OPD at equal teacher budget / untrained; C. success rate vs teacher calls
and env steps as the efficiency read-out. Teacher size (14B on one GPU vs 32B on an H100 vs API)
is the researcher's call; 14B first unless its own ALFWorld success is too low to teach.

## 38. Close Method II on the debate team; keep the mechanism, not the method (2026-09-20)

Seven arms, identical budgets and seeds, 238 held-out tasks: uniform GRPO = base (+0.2pp); every hard
mask below base (attr -3.2, random -2.5, last -6.1, attr_committed -5.3pp); soft masks at best neutral
(attr_soft -0.2, random_soft -1.9pp); attr vs random indistinguishable under both hard (-0.6pp, p=0.74)
and soft (+1.7pp, p=0.19) masking. Decisions: (1) report this as a complete negative result with its
mechanism (a hard mask removes the uniform negative gradient that keeps outputs inside the token budget;
a floor restores length control but adds nothing) and its root cause (98% of failures have no decisive
step); (2) do not spend more compute on masking variants for this team -- the remaining knobs (longer
cap, 7B team, more steps) change the substrate, not the finding; (3) the location question moves to
ALFWorld, where the progress oracle certifies a first harmful action and Model A already finds it 46% of
the time; (4) keep the seven adapters and per-task records for the paper's appendix.

## 39. ALFWorld Attr-OPD round 1: the location IS the signal when failures are localisable (2026-09-20)

Result: one verified teacher call per failure at the oracle's first harmful action lifts a 41.8% student to
58.2% on unseen games; at Model A's step to 47.8% (beats random p=0.005); at a random step 34.3% and at
the last step 32.8% -- both BELOW the untrained student. Decisions: (1) this is the paper's constructive
half -- the same attribution-as-location idea that was null on debate works when the failure has a first
harmful action, and the difference is measurable in advance by localisability and by the verified-fix
rate; (2) the verified-fix rate is adopted as the cheap, training-free proxy for "is this attributor
useful here" (oracle 41 > Model A 36 > random 10 > last 0); (3) unverified teacher targets are dangerous
-- distilling them at unfixable states caused the harm in random/last, so `--fallback_unverified` should
be off (or down-weighted) in later rounds, and "skip when nothing verifies" becomes the default;
(4) follow-ups in priority order: attr_w1 (window +-1 around Model A's step), iterated round 2 for oracle
and attr, then a Model A fine-tuned on ALFWorld self-play labels (progress oracle gives free labels) to
close the attr-oracle gap; (5) report costs: 319 teacher calls and ~14k env steps per arm vs ~1,000
calls for 20 failures under uniform OPD.

## 40. Verified-only distillation is the method; the fallback was the bug (2026-09-21)

Evidence: across seven round-1 arms the number of unverified fallback targets predicts unseen success with
r = -0.91 (uniform 960 -> 8%, attr_w1 597 -> 16%, last 319 -> 33%, random 288 -> 34%, attr 203 -> 48%,
oracle 189 -> 58%); removing the fallback from the attr arm (attr_nofb, 125 verified examples) lifts it
from 47.8% to 58.2%, equal to the oracle arm, with the same 319 teacher calls. Decisions: (1) all further
arms and rounds run without the fallback (`*_nofb`); (2) the story for the paper: attribution picks the
state, the teacher proposes k candidates, the environment verifies, and only verified corrections are
distilled -- attribution's value is measured by the verified-fix rate it achieves, which Model A (Who&When
Pro-trained, ALFWorld-untrained) already delivers at oracle level once fallbacks are gone; (3) the round-1
fallback arms stay in the paper as the ablation that exposes the failure mode of unverified OPD targets;
(4) the extended evaluation (774 games) is the reporting standard from now on -- the attr-with-fallback arm's
+6pp on unseen did not carry to the seen split (-5pp), while the oracle arm's did (+6pp pooled, p=0.002).

## 41. Attributor curriculum from the environment's own progress signal (2026-09-21)

Observation: Model A's localisation of the first harmful action falls from 46% (zero-shot policy's failures)
to 19% (the 78%-success round-2 policy's failures) -- exactly where attr_nofb's second round regressed
(58.2% -> 49.3%, verified-fix rate 39% -> 29%). Decision: fine-tune Model A on self-play failures labelled
by the progress oracle (no human labels, ~3 s per trajectory on CPU) and re-check on a held-out set of the
hard failures: exact 19% -> 58%, within-2 28% -> 66% after one 10-minute LoRA epoch on 1,099 rows. Rules:
(1) the attributor is refreshed per round from the previous rounds' labelled failures (held-out set never
used for training); (2) the mode label is copied from Model A's own prediction (no new mode information
is claimed); (3) the paper reports this as the mechanism that keeps the location signal alive as the
student improves -- the oracle chain compounded (58 -> 75 -> 77) because its pointer never degraded, the
Model A chain did not because its pointer did; a refreshed attributor is the fix to test (attralf_nofb
round 1 now, round 2 next).

## 45. The attribution-free baseline is strong; attribution adds a smaller but real increment on top of it (2026-09-22)

const1_r2 (round-2 failures, no attributor, always step 1, same teacher/recipe as attralf_nofb round 2) pooled
72.2%, beating the untrained student by 18.5pp with zero attribution. attralf_nofb round 2's refreshed attributor
adds a further +4.0pp on top (76.2%, p=0.0024 vs const1_r2). On the 134-game unseen split alone the ordering even
reversed (const1_r2 73.9% > attralf 68.7%, n.s.) -- a reminder that 134-game differences under ~10pp need the
774-game pool before being read at all (see #42's variance point). Decision: state the paper's claim as "verified
teacher-proposed-alternative distillation is the primary source of the gain; a learned, environment-refreshed
attributor adds a further, statistically significant increment on top of that strong baseline" rather than
attributing the full gain to attribution. Report const1 controls at every round where a headline number is claimed.

## 44. Pointer imprecision is direction-specific: early is nearly free, late is costly (2026-09-21 21:00, researcher's request)

Offset controls on oracle_nofb's 319 round-1 states (same teacher, same nofb recipe, only the supervised step
shifted by a fixed +-k from the planner oracle's exact first-harmful-action step): +1 costs verified examples
(39% -> 30%) but almost no downstream gain (pooled 63.0% -> 62.1%, n.s.); +2 collapses to the untrained student's
level on all three splits (52.2% pooled, n.s. vs student); -2 matches the exact step (53.7% vs 56.0% unseen, p=0.72)
and has the HIGHEST verified-fix rate of any offset tested (46%). Mechanism: round-1 failures' first harmful action
sits at a mean step of 2.0, so -2 mostly lands near episode start (teacher advice still reliable, easy to verify)
while +2 overshoots past the harmful decision into a state where the damage is already done (teacher alternatives
rarely lead anywhere). Decision: report imprecision tolerance as asymmetric/directional, not as a function of
|offset| alone; the practical implication for the attributor curriculum is that a pointer biased slightly EARLY is
safer than one biased slightly LATE in the verified-only regime.

## 43. The attributor curriculum is confirmed as the mechanism that makes the Model A chain compound (2026-09-21 13:34)

Prediction from #41: a pointer that tracks the improving policy should make the Model A chain compound like the
oracle chain. Result: attralf_nofb round 2 = 68.7% unseen (round 1: 56.7%, p=0.005), vs 49.3% for the same recipe
with the original pointer (p=6e-6), 6.7pp below the planner-oracle chain's round 2 (n.s.). Decisions: (a) the
attributor is refreshed every round from all previous rounds' progress-oracle labels (v2 adds the three round-2
failure sets; held-out test set fixed to the oracle round-3 failures for comparability); (b) the attribution claim
is stated as "a learned, environment-refreshed localiser that is 6-7pp behind a privileged planner oracle and
19pp ahead of the same recipe with a stale pointer" -- REVISED by the pooled read-out (774 games, 14:27): the refreshed
chain's round 2 (76.2%) is statistically identical to the planner-oracle chain's round 2 (76.5%, p=0.93) and 19.4pp above
the stale-pointer recipe (56.8%); the claim becomes "matches the privileged oracle at round 2", pending the const1
round-2 control (#42); (c) pointer-offset controls (oracle step +-2, +5; researcher's request) quantify how much imprecision
the verified-only recipe tolerates -- the hypothesis, from the fixability analysis, is "imprecision costs verified
examples, not policy quality" in the verified-only regime and "poisons" in the fallback regime.

## 42. Two controls the verified-only result was missing: the constant-step arm and contrastive verification (2026-09-21)

Trigger 1: the game-disjoint attributor test (held-out games 1500-1999) showed that "always point at step 1"
hits the novice's first harmful action 60-80% of the time on early-error failure sets and that the
Who&When-trained Model A never beats that constant baseline on ALFWorld (26-32% exact vs 34-80%), while the
refreshed attributor does by a wide margin (89% vs 34% where errors are late). So round 1's "attr_nofb =
oracle_nofb" is explained by the early-error prior plus verification, not by attribution. Decision: (a) every
localisation number is reported next to the constant-step baseline (not only random); (b) a `const1_nofb` arm
(always step 1, verified-only, same 319 failures) is run as the attribution-free control for round 1; (c) the
attribution claim rests on later rounds, where the constant step is right 26-34% of the time -- attralf_nofb
round 2 vs attr_nofb round 2 (49.3%) vs oracle_nofb round 2, with a const1 round-2 control if attralf compounds.

Trigger 2: the counterfactual check found ~20% of "verified" targets (attr_nofb 19%, oracle_nofb 24%) are states
the greedy student recovers from with its ORIGINAL action (the failure was a T=1 sample); `look` targets are
mostly neutral. Decision: `--contrastive` (also continue after the original action; skip if it wins) is the
default for new arms once `round1_cv` confirms it does not hurt; existing results stand as "verified" not
"counterfactually verified" and the paper states the fraction.

Trigger 3: attralf_nofb round 1 (58.9% pooled) vs oracle_nofb (63.0%) differ by 4pp with 96% identical steps and
107/122 identical verified examples -- one distillation round's run-to-run variance is of that order. Decision:
a training-seed replicate of attr_nofb round 1 (`round1_seed1`) is run, and differences between verified-only
arms below ~5pp pooled are not interpreted until the replicate bounds the variance.

## 10. Caught a scoring bug via an implausibly-clean 0.000, not via general suspicion

Initial attributor eval showed step accuracy at EXACTLY 0.000 for
mathchat/debate/dylan while other frameworks scored 90-100%. The
diagnostic signal was the *exact* zero across 151 examples, not "these
numbers look low" -- a real model producing genuinely wrong predictions
essentially never scores literally 0/151 by chance; that pattern almost
always means the grading path itself is broken. Root cause:
`eval_attributor.py` synthesized `gt = {"step": gold_step, ...}` using the
already-rendered coordinate string, but `whowhen_eval.score`'s registered
scorers for mathchat (`2*round+position`) and debate/dylan (round-only)
read `gt.round`/`gt.position` directly -- absent, their accept-sets are
silently empty and every prediction scores wrong regardless of content.
Fix: store the raw `ground_truth` dict in the SFT data itself
(`sft_format.py`) rather than relying on eval-time reconstruction, and
have `eval_attributor.py` use it directly. Recovered corrected numbers
from the already-generated raw predictions with a pure-CPU re-scoring
pass -- no GPU time was wasted re-running generation. General lesson: an
exact 0 (or exact 1.0) on a sub-slice is a stronger bug signal than a
merely-low number, and is cheap to grep for.

## 11. Long-running GPU jobs must run inside tmux, not just backgrounded

A harness/session restart mid-run killed both the attributor-training and
feedback-study `srun --overlap` background processes, even though their
SLURM job allocations kept running -- the processes were children of the
previous session's shell and died with it (confirmed: 0% GPU utilization,
no python in `ps aux`, right after the restart). Both had actually already
finished by the time this was caught, so no work was lost this time, but
it could easily have killed a run hours from completion. Fix going
forward: launch long GPU jobs inside `tmux` (`tmux new-session -d -s
<name> '<command>'`) run from the STABLE node the harness's own shell
lives on (not nested inside another `srun --overlap` step -- tested and
confirmed that a tmux server created *inside* an `--overlap` step dies
when that step's initiating process exits, even detached). This mirrors
the standing tmux lesson already recorded for the peer-induced-error
project; now confirmed to generalize here too.

## 5. Pivoted to attributor training on the researcher's explicit request

After Phase 1 progress (below), the researcher clarified they specifically
wanted the attributor trained on existing data now, not the oracle
feasibility track first. Switched focus accordingly -- Phase 1's code and
findings are preserved (nothing was thrown away, and the ALFWorld
oracle-attribution mechanism turns out to matter again for decision 8's
transfer-set design), but active work moved to Sec. 9.C.

## 6. Use the OFFICIAL Who&When Pro release, not the first HF mirror found

The first dataset found by search (`tmpxv7/who-when-pro`) is a community
re-upload that is 89% broken symlinks -- only `alfworld` and
`video/lvbench` resolve to real files; every multi-agent text framework
(debate/dylan/macnet/metagpt/magentic-one/mathchat/smolagents) and all
image data are dangling links to a path (`mmagent/mas/runs/...`) that was
never included in the tarball. This was caught by actually checking
`find -xtype l` counts rather than trusting the file listing, and would
otherwise have silently produced a training set missing all multi-agent
"who" supervision. Traced through the paper (arXiv:2607.09996) and its
project page (whowhenpro.github.io) to the real source:
`Leoxx/whowhen_pro` on HF (flat jsonl, no broken links) + the companion
eval harness at github.com/whowhenpro/whowhen_pro. Always verify a
downloaded dataset's actual file integrity (not just presence) before
building on it, especially for community re-uploads of nested-directory
releases.

## 7. Vendor and reuse the official eval harness rather than re-deriving 8 trajectory schemas

Who&When Pro's 8 text-modality frameworks (alfagent, smolagents, debate,
dylan, macnet, metagpt, mathchat, magentic-one) each store trajectories
and ground-truth coordinates in a different native shape (flat step
numbers vs. round+position vs. stage vs. round-only). The paper's own
repo already solves this with per-framework renderers + a per-framework
GT-to-coordinate scorer, MIT-licensed. Reimplementing this myself risks
subtle coordinate-mapping bugs (e.g. mathchat's `2*round+position` fold)
that the authors have already debugged, and -- more importantly -- using
their exact renderer + prompt means the specialized attributor trained
here and their frontier-LLM-judge baseline see byte-identical inputs, so
RQ1's comparison is fair by construction rather than fair "in spirit."
Vendored under `third_party/whowhen_pro_repo/`, pip-installed as
`whowhen_eval`.

## 8. Hold out `alfworld` entirely from attributor training, as the transfer-test set

Rather than a plain random split, the `alfworld` benchmark (311 real
traces) is withheld completely from train/val. This wasn't just
convenient -- it directly sets up RQ3 (attribution transfer to an unseen
benchmark): the trained attributor's step/mode accuracy on these 311
withheld traces, AND its downstream learning utility on this project's own
ALFWorld environment (Phase 1's oracle-attribution mechanism lives right
next to this data), can both be measured later without any additional
data engineering.

## 9. Attributor model: Qwen2.5-3B-Instruct, LoRA, deliberately smaller than the 7B policy

Sec. 9.C's own motivation for a specialized fine-tuned attributor is
amortized attribution -- a small model standing in for an expensive
frontier-LLM-judge call on every rollout. Picked 3B (vs. the 7B chosen
earlier for the policy) to keep that distinction real rather than
training two same-sized models. LoRA (r=16) rather than full fine-tune:
cheaper, and the base model's general instruction-following is a good
prior for a format-constrained (3-line) output task. max_seq_len=8192
covers ~90% of examples by actual tokenizer count (not char-length
proxy); the rest are DROPPED, not truncated -- truncating risks silently
cutting off the Response Format section or the target itself.

## 1. Start with Phase 1 (Oracle Feasibility), not "train an attributor first"

The research plan's own Sec. 25 ("Minimum Viable Experiment") and Sec. 26
(recommended development order) explicitly say not to train the
attributor first: check with oracle attribution whether attribution helps
at all before investing in a learned attributor (that's Phase 2). Followed
the plan's own ordering over the more literal reading of the initial
request ("train an attributor first" was given as an illustrative example
of "step by step," not a hard requirement) -- if oracle attribution shows
no benefit, training a specialized attributor next would be wasted effort.

## 2. Benchmark: ALFWorld first (not WebShop)

Plan Sec. 26 recommends "ALFWorld or WebShop" for Phase 1. Chose ALFWorld:
pip-installable text-only env (no Selenium/Amazon-scrape setup WebShop's
web mode needs), ships its own PDDL backend and two expert types, which
turned out to be exactly what's needed to build a genuine oracle attributor
(see decision 3). WebShop remains the natural second benchmark per the
plan's Sec. 15.B, unchanged.

## 3. Oracle attribution operationalized via PDDL-planner recoverability check, not Who&When-style injected failure

Who&When Pro's oracle construction (inject a failure after exact replay of
a successful prefix) doesn't transfer directly to ALFWorld single-agent
rollouts -- there's no pre-existing causally-labeled failure corpus for
ALFWorld. Instead: t* = earliest step after which the trajectory becomes
unrecoverable within the remaining step budget, verified by replaying the
prefix and handing off to a certifying solver.

Two experts are available in ALFWorld; validated both:
- `HANDCODED` (fixed script from the original solution): gives
  non-monotonic recoverability traces when resumed off-script -- a
  false-negative artifact of the script failing to replan, not a real
  game property. Rejected as the oracle's solver.
- `PLANNER` (TextWorld's live PDDL solver, replans from the current state
  every call): gives clean monotonic traces. **Chosen.**

Why this matters: recoverability is tied to the SAME total step budget the
policy is trained/evaluated under (not a generous fixed allowance) --
"recoverable given unlimited steps" is a much weaker and less useful
oracle than "recoverable within the actual horizon the policy operates
in." `verify_monotonicity=True` is kept as a runtime assertion (not just a
one-off check) so any residual non-monotonicity at scale is caught rather
than silently mislabeling.

**Empirical finding worth remembering:** short random-walk trajectories in
ALFWorld are very often still fully recoverable (picking up a wrong object
or wandering rarely locks the game into an unwinnable state -- ALFWorld's
action space is mostly reversible). Genuine irrecoverability mostly comes
from exhausting the step budget on a bad detour, not from a single
"destructive" action the way Who&When Pro's injected failures work. This
should be reported as a finding, not hidden: it's directly relevant to the
plan's Risk 2 (diagnostic vs. actionable attribution) and to picking task
types / rollout lengths for Experiment 0 (need long-enough or
adversarial-enough rollouts to actually produce unrecoverable failures to
attribute).

## 4. Compute: CPU hold allocation via SLURM `interactive` partition, not the login node

Installs OOM-killed on the shared login node. User requested a 24h CPU-only
node; submitted via `slurm/hold_cpu.sbatch` (job 17376356, `interactive`
partition, 8 cpus/64G, node acn50). All commands since route through
`srun --jobid=17376356 --overlap <cmd>`. Distinct from the `peer-induced-error`
project's own GPU-hold pattern (`interactive-gpu` partition) -- this
project's oracle-labeling work is CPU-bound (PDDL solving), so no GPU
needed yet. GPU will be needed once a policy LLM enters the rollout loop.
