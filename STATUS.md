# STATUS

Last updated: 2026-09-09 08:55. **Full write-up with worked examples: docs/report_attribution_and_self_correction.md (中文) / docs/report_attribution_and_self_correction_en.md (English).**
**Disk note (2026-09-15):** the home quota (200 GB) was hit; deleted regenerable artifacts -- the two merged 3B model copies
(`attributor_qwen2.5-3b-sft-merged`, `attributor_3b_noobs_merged`; re-create by merging base + adapter before evaluating the GRPO / OPD adapters that used them as base) and all Trainer `checkpoint-*` / OPD `step*` dirs (every run's `final/` adapter is kept). **Active work plan: TODO.md** ("Is that
really true?" exercises per the advisor + item 4 "more frameworks" per the
researcher) -- items 1-4 are ALL COMPLETE as of 08:50; this file holds
results and running-state detail. Headline outcomes of the night: (a)
self-correction by the erring team is content-blind in all three
frameworks but responds to authority framing (pooled p=0.0006); (b) Model
A should be trained on agent text without environment observations (same
in-distribution, +9-11pp OOD step, confound closed); (c) error position is
framework-dependent, so online attribution is feasible on embodied tasks and
infeasible on debate-style ones; (d) two grading bugs fixed, no conclusion
changed.

**Running now (2026-09-09 ~00:20), four tasks on four GPUs, all in tmux:**
- L40S (job 22443): IDLE since ~06:30 (`chain_mathchat_framing` DONE;
  `chain_framing173` DONE -- framing effect replicated across all three
  frameworks, pooled p=0.0006; see "TODO 4").
  (`chain_rerun24` DONE and spliced; `chain_dr150_framing` DONE.)
- preempt-gpu A40 (job 94232): IDLE since ~06:50 (`chain_mathchat` DONE --
  see "TODO 4"; `collect_mathchat` DONE: 593 tasks, 326 usable failures).
- PENDING: A100 #5 (88115) and preempt-gpu (94232) -> TODO 4 self-play
  collection for mathchat and magentic-one (code done, CPU-tested ~01:50).
- A100 #1 (job 41169): IDLE since ~06:35 (`chain_magentic` DONE -- see
  "TODO 4"; `sft_extractive` + eval DONE; `ablate_inputs` DONE).
- A100 #2 (job 83939): IDLE since ~08:50 (`sft_no_obs_keepids` + eval DONE:
  confound closed, ALFWorld step +9pp on identical data, p=3.4e-4 -- "TODO 3").

**Framing scale-up finished (19:10): mathchat 326 / magentic 415, all held A100s 84810/83939 idle.
ALL GPU WORK FINISHED 2026-09-10 17:25; every hold released (only 256803 pending as a spare, per
agreement with the physics session). External-reader baselines run 2026-09-12 on hold 531737 (self-released); nothing running as of 2026-09-15.** Earlier note (08:50): Every GPU hold is idle; holds expire on
their own 24h clocks (L40S 22443 ~17:40, A100s 41169/83939/84810 ~19:00-23:00,
preempt 94232 ~01:40 next day). Nothing further launches without a new decision.**
- A100 #3 (job 84810): IDLE since ~06:20 (`chain_magentic_framing` DONE;
  `collect_magentic` DONE: 593 tasks, 415 usable failures; first_line
  retrain + eval DONE).
- A100 #5 (job 88115): QUEUED -> reserved for the `--keep_ids` data-size
  control on whichever compressed retrain matches/beats full.
Comparison tooling ready: `compare_protocols.py` (same condition across
protocols), `compare_conditions.py` (cross-condition, cross-run),
`compare_attributors.py` (attributor variants). Item-3 retraining is a
one-liner (`prepare_attributor_data.py --variant`, `train_attributor.py
--data_dir/--output_dir`).

## Headline results (2026-09-08)

Both the attributor training run and the full feedback-granularity study
completed overnight (survived a harness/session restart -- the SLURM jobs
themselves kept running; only the harness's own tracking of them was
lost, see "Compute" note below). Results:

**Specialized attributor (Qwen2.5-3B LoRA) -- real, learned attribution
skill, not just format-copying:**
- In-distribution val (n=593): agent 97.6%, step 95.6%, mode 82.8%,
  all-3-correct 81.6%.
- RQ3 transfer to held-out ALFWorld (n=311, never seen in training):
  agent 100% (trivial single-agent case), step 53.4%, mode 26.0%,
  all-3 20.3%. A real, substantial, and expected transfer gap -- the
  plan's own Risk 1 anticipated this. Step localization transfers
  much better than failure-mode classification, which nearly halves.
- Training loss: 2.04 -> 0.014 (step 10 -> 890); eval_loss monotonically
  0.083 -> 0.024 across all 3 epochs, no overfitting blowup.
- **A real scoring bug was caught and fixed** during evaluation: initial
  numbers showed step accuracy at EXACTLY 0.000 for mathchat/debate/dylan
  (not "low" -- exactly zero across 151 examples), which is what made it
  suspicious rather than just "these are hard frameworks." Root cause:
  `eval_attributor.py` reconstructed a synthetic ground-truth dict using
  the already-rendered coordinate string, but `whowhen_eval.score`'s
  per-framework scorers for mathchat (`2*round+position`) and debate/dylan
  (round-only) need the RAW `round`/`position` fields -- absent, their
  accept-sets were silently empty, scoring every prediction wrong
  regardless of content. Fixed by storing the raw `ground_truth` dict in
  the SFT data (`sft_format.py`) and having `eval_attributor.py` use it
  directly. Recovered the correct numbers from already-generated
  predictions with a one-off re-scoring pass (no GPU time wasted) --
  true numbers are meaningfully better than the buggy ones (val all-3:
  0.641 buggy -> 0.816 correct). ALFWorld transfer numbers were unaffected
  by this bug (alfagent's scorer reads `gt.step` directly, coincidentally
  the same field the synthetic dict provided).

**Feedback-granularity study (n=2,328 traces x 4 conditions =
9,312, all statistically well-powered):**
- `precise_only` (exact location, no reason) is significantly WORSE than
  every condition that includes a reason -- vs `reason_only`: McNemar
  exact p=1.3e-8; vs `vague_reason`: p=0.0006; vs `precise_reason`:
  p=9.6e-9.
- `reason_only` vs `precise_reason` (does adding precise location on top
  of the reason help further?): p=0.91 -- statistically indistinguishable.
- Reading: **diagnostic content ("why") drives correction; location
  ("where") alone barely helps, and adds nothing once "why" is known.**
  Directly relevant to the main research plan's Sec. 20 T1-vs-T3 framing.
  **Scope caveat (researcher, 2026-09-08): the failed trajectories are
  from OTHER systems (Who&When Pro), not from the Qwen model that gets the
  feedback -- so this measures an external reader's correction given
  hints, NOT a team's self-correction / collaboration dynamics. The
  narrower claim stands; the collaboration framing does not. Fix = generate
  failures from the tested Qwen model itself (see
  docs/feedback_granularity_study.md, LIMITATION section). Model A is
  unaffected (cross-agent transfer is its intended setting).**
  magentic-one (GAIA) shows this pattern too but with a ceiling-effect
  caveat (see docs/feedback_granularity_study.md) -- our one-shot protocol
  gives no new tool/retrieval access, so answers needing missing
  information can't be fixed by feedback of any kind.
- **Connected Model A to this study (2026-09-08, later same day):** reran
  the same 4 conditions using Model A's own PREDICTED labels (not gold),
  on the 232 traces both in the gradable-framework set and in Model A's
  held-out val split. Result: essentially no difference from oracle labels
  (0-1.3pp gaps, McNemar p >= 0.25 on every condition, only 2-4/232 traces
  flip per condition) -- despite Model A being genuinely imperfect there
  (82.8% mode accuracy). Reassuring: the attributor doesn't need to be
  perfect to be useful as feedback content. See
  docs/feedback_granularity_study.md for the full table + caveats.

## TODO 2b: "Do we need the full trace?" -- inference-time input ablations (RUNNING, partial)

Model A (full-trace SFT) evaluated on ablated prompts, val n=593, official
scorer. `kept` = fraction of trace characters retained; `gold_vis` = fraction
of traces whose gold step is still present in the shown input. Caveat: Model A
never saw these formats in training, so drops mix "information removed" with
distribution shift -- clean answers come from the item-3 retrains.

| variant          | agent | step  | mode  | all-3 | kept | gold_vis | reading |
|------------------|-------|-------|-------|-------|------|----------|---------|
| full (reference) | 0.976 | 0.956 | 0.826 | 0.815 | 1.00 | 1.00 | reproduces the SFT eval |
| no_observations  | 0.976 | 0.949 | 0.815 | 0.803 | 0.75 | 1.00 | env/tool outputs nearly dispensable (-1.2pp) |
| no_task          | 0.978 | 0.954 | 0.806 | 0.794 | ~1.0 | 1.00 | task text nearly dispensable (-2.1pp) |
| no_agent_ids     | 0.002 | 0.946 | 0.762 | --    | ~1.0 | 1.00 | "who" is read off the label (agent trivially lost); step intact, mode -6.4pp |
| prefix_half      | 0.914 | 0.216 | 0.351 | 0.108 | 0.52 | 0.24 | errors are LATE: gold step in 2nd half for 76%; step acc ~= visibility cap |
| suffix_half      | 0.887 | 0.703 | 0.582 | 0.518 | 0.31 | 0.69 | conditional on gold visible: step ~1.0, mode ~0.84 (= full). The 2nd half is only 31% of chars. |
| shuffled         | 0.943 | 0.857 | 0.712 | 0.668 | 1.00 | 1.00 | order destroyed, all content + step labels kept: -10pp step, -11pp mode, -15pp all-3. Order matters, but less than content. |
| first_line       | 0.902 | 0.560 | 0.430 | 0.390 | 0.10 | 1.00 | 10% of chars, every step present as a one-line stub: step -40pp, mode -40pp WITHOUT retraining. The item-3 retrain on this exact format is the clean test. |
| extractive       | 0.943 | 0.855 | 0.636 | 0.612 | 0.65 | 1.00 | keep only lines with numbers/answers: step -10pp, mode -19pp. Costs more mode than no_observations (0.75 kept) -- the dropped prose lines carry the "why". |
| final_only       | 0.000 | 0.084 | 0.260 | 0.000 | 0.00 | 0.00 | lower bound (final + gold answer, no trace): mode 0.26 = the label-prior/guessing floor; step 0.08 = common-coordinate guessing. Everything above this line is trace-derived. |

**ALFWorld transfer (held-out benchmark, n=311), same variants:**

| variant          | step  | mode  | all-3 | kept | gold_vis | reading |
|------------------|-------|-------|-------|------|----------|---------|
| full (reference) | 0.534 | 0.257 | 0.203 | 1.00 | 1.00 | reproduces the SFT transfer eval |
| no_observations  | 0.605 | 0.238 | 0.193 | 0.59 | 1.00 | step +7pp: observations (41% of chars) are NOISE out of distribution |
| first_line       | 0.592 | 0.244 | 0.206 | 0.46 | 1.00 | all-3 >= full at 46% chars (first line of a step = the action; strips observations) |
| no_agent_ids     | 0.447 | 0.254 | --    | 1.01 | 1.00 | step -9pp (val: -1pp): "Agent:" headers anchor the blocks OOD |
| no_task          | 0.405 | 0.248 | 0.148 | 1.00 | 1.00 | step -13pp (val: 0): ALFWorld states the goal ONCE; chat frameworks restate it |
| extractive       | 0.486 | 0.219 | 0.138 | 0.92 | 1.00 | number-filter barely compresses ALFWorld (every line has "shelf 2"), yet hurts |
| shuffled         | 0.254 | 0.215 | 0.109 | 1.00 | 1.00 | step -28pp (val: -10pp): an embodied error is defined by sequence |
| prefix_half      | 0.531 | 0.225 | 0.199 | 0.63 | 0.82 | gold in the FIRST half for 82% (val: 24%); first half alone = full step acc |
| suffix_half      | 0.077 | 0.158 | 0.013 | 0.37 | 0.18 | mirror image |
| final_only       | 0.084 | 0.174 | 0.000 | 0.00 | 0.00 | floor |

Both passes COMPLETE (20/20). Takeaways:
(i) In-distribution, localization is robust to removing observations, task
text, and even agent names -- it is driven by the agents' own prose; mode is
the fragile head (every ablation costs mode more than step), and the lines
extractive drops (prose reasoning, not observations) are where the "why"
lives (extractive 0.65 kept, -19pp mode vs no_observations 0.75 kept, -1pp).
(ii) "Do we need the full trace?" has OPPOSITE answers by trace type. On
chat-style Who&When frameworks, compression costs accuracy (first_line
-40pp zero-shot). On ALFWorld, removing observations / first-line
compression IMPROVES step localization (+6-7pp): the model never learned
to weight ALFWorld's long room descriptions, so they act as noise OOD.
Compression is a loss in-distribution and a free denoiser out-of-distribution
-- a concrete, testable lever on the transfer gap (retrain on
no_observations-style inputs and re-measure transfer).
(iii) Error POSITION is framework-dependent: gold step in the 2nd half for
76% of val traces (debate/mathchat commit the error in the final rounds)
but in the 1st half for 82% of ALFWorld traces (a wrong early navigation
dooms the episode). Step accuracy tracks the gold-visibility cap almost
exactly in all four prefix/suffix cells, so mid-trajectory (online)
attribution is structurally feasible on embodied tasks and structurally
infeasible on debate-style tasks -- not a model limitation either way.
(iv) Task text and temporal order matter far more OOD (no_task -13pp,
shuffled -28pp on ALFWorld vs ~0 / -10pp on val): chat traces restate the
goal and expose errors within single turns; embodied traces state the goal
once and define errors by sequence.

## TODO 4: self-correction across THREE frameworks (researcher: "debate alone is not enough") -- 2/3 chains DONE

Same Qwen-7B, same 593-task pool, same grader (post #25/#26), 150 seeded
failures per framework, Model A's predicted labels, framework-native extra
round (debate: one more round; mathchat: proxy delivers the feedback;
magentic-one: orchestrator receives it and may re-delegate). Recovery %:

| condition | debate (n=118 true) | mathchat (n=150) | magentic-one (n=150) |
|---|---|---|---|
| incorrect_only (no info) | 5.1 | 7.3 | 3.3 |
| reason_only | 5.1 | 4.7 | 5.3 |
| vague_reason | 6.8 | 5.3 | 6.0 |
| precise_reason | 4.2 | 7.3 | 4.0 |
| precise_only | 5.9 | 6.7 | 4.7 |
| wrong_location (control) | 8.5 | 6.0 | 3.3 |
| placebo_reason (control) | 5.1 | 3.3 | **7.3** (top cell) |
| random_attribution (control) | 6.8 | 4.0 | 4.7 |
| expert_incorrect_only (framing) | 8.5 / pooled 10.7 | **10.7** | **7.3** |
| expert_precise_reason (framing) | 8.5 / pooled 10.0 | **10.0** | 4.7 |

ALL THREE FRAMEWORKS DONE. Content verdict is unanimous: in no framework does
any correct-content condition beat "your answer was incorrect" (all p >= 0.34),
and in no framework do the wrong-content controls do worse than the correct
ones. mathchat: incorrect_only (7.3%) ties for the best content cell.

**Pooled cross-framework paired tests -- FINAL after the scale-up (debate 291 + mathchat 326 + magentic 415 = 1,032 paired traces):**

| contrast | debate | mathchat | magentic | pooled discordant | p |
|---|---|---|---|---|---|
| expert_incorrect_only vs incorrect_only (authority, no content) | 15/4 | 21/6 | 24/12 | **60 / 22** | **3.2e-5** |
| expert_precise_reason vs precise_reason (authority + content vs content) | 13/3 | 18/8 | 16/14 | **47 / 25** | **0.013** |
| expert_precise_reason vs expert_incorrect_only (does content add to authority?) | 7/9 | 12/17 | 11/14 | 30 / 40 | 0.28 (negative) |
| precise_reason vs incorrect_only (does content add at all?) | 8/9 | 12/12 | 21/14 | 41 / 35 | 0.57 |

Per framework, authority-without-content vs neutral: debate 10.7 vs 6.5% (p=0.012, n=291),
mathchat 10.1 vs 5.5% (p=0.006, n=326), magentic 8.0 vs 5.1% (p=0.065, n=415). Same sign
and 3-5pp size in all three; significant alone in two of three; pooled p=3e-5. Diagnostic
content: zero in every framework, and if anything negative on top of authority (30/40).

magentic-one detail: every content pair vs incorrect_only p >= 0.5; controls =
correct content (placebo_reason is the best cell). Framing: expert_incorrect_only
vs incorrect_only 7.3 vs 3.3% (9/3, p=0.15) -- same direction and size as
debate (+4pp), underpowered at n=150; expert_precise_reason vs
precise_reason 4.7 vs 4.0% (6/5, p=1) -- no replication of the second pair
here. Authority-without-content again >= authority-with-content (6/2).
Model A's diagnosis profile differs by framework (debate: C.3 communication
49%; mathchat: R.4/R.3 reasoning 81%, always `assistant`; magentic: R.1 46%,
blame split coder 80 / orchestrator 68) -- the attributor reads the
topology; the recipients ignore it equally.

## TODO 1: self-correction under `debate_round` (researcher's preferred protocol) -- DONE, NULL

**Numbers restated 2026-09-09 ~04:00 under the fixed parser + tolerant
grader (DECISIONS #25/#26) on the 118 TRUE failures of the seeded 150
(32 were false failures: 20 scibench, 12 math). Pre-fix numbers are in
git history / `corrections.pre_splice.jsonl`; conclusions did not move.**

Qwen-7B 3-agent debate team; feedback injected as an extra debate round to
the SAME three agents; Model A's predicted labels as content. n=118 (gpqa 15,
math 38, mmlu_pro 17, scibench 48).

| condition       | recovery      | changed answer | correct given changed |
|-----------------|---------------|----------------|-----------------------|
| incorrect_only (no info) | 6/118 = 5.1% | 0.356 | 0.143 |
| reason_only     | 6/118 = 5.1%  | 0.373 | 0.114 |
| vague_reason    | 8/118 = 6.8%  | 0.441 | 0.135 |
| precise_reason  | 5/118 = 4.2%  | 0.398 | 0.085 |
| precise_only    | 7/118 = 5.9%  | 0.390 | 0.152 |

Every pre-registered McNemar pair p = 1 (2-3 discordant traces each). No
condition beats the no-information control. math 1-2/38 everywhere. When
the team changes its answer (36-44% of the time) it lands right 9-15% of
the time -- below the 33% chance of merely switching options on gpqa.
(compare_conditions.py --regrade --ids_file true_failure_ids.json)

**vs one_shot on the SAME 118 true failures** (`compare_protocols.py
--regrade --ids_file`): one_shot 5.1-7.6% vs debate_round 3.4-5.9%,
p = 0.73 / 1 / 1 / 0.18 / 1 -- identical within noise; debate_round raises
the answer-change rate a little (0.31-0.38 -> 0.35-0.42) without converting
any of it. The pilot's "doubled change rate" did not survive.

**vs the wrong-content controls (TODO 2a) on the SAME 118 traces**: 
wrong_location 8.5% vs precise_only 5.9% (3/0, p=0.25); placebo_reason
5.1% vs reason_only 5.1% (2/2, p=1); random_attribution 6.8% vs
precise_reason 4.2% (4/1, p=0.375). Pooled over 11 conditions x 118 traces:
recovery 4.2-8.5%, and the three highest cells are wrong_location and the
two expert-framed conditions. **Correct feedback = wrong feedback = no
feedback.**

Verdict (DECISIONS #24): on this team's own failures, self-correction is
anchoring-bound; feedback TYPE (where / why / precision) is not a lever, and
neither is the injection protocol (one_shot vs debate_round). The remaining
untested lever is the REVISION MECHANISM itself -- hide the team's previous
answer and force re-derivation, or hand the feedback to a fresh instance /
a different model (backlog items in TODO.md). The external-reader effect
("why >> where", n=2,328) stands as a statement about readers, not teams.

## TODO 2a-bis: presentation / authority framing (researcher's "expert diagnosis" question) -- DONE; framing effect REPLICATED, content effect still NULL

**Replication on the 173 fresh debate true failures (disjoint from the 150-set),
then pooled paired test on all 291 true failures:**

| pair (same traces) | n=173 fresh | pooled n=291 |
|---|---|---|
| expert_incorrect_only vs incorrect_only | 12.1% vs 7.5%, 11/3, p=0.057 | **10.7% vs 6.5%, 16/4, p=0.012** |
| expert_precise_reason vs precise_reason | 11.0% vs 7.5%, 9/3, p=0.15 | **10.0% vs 6.2%, 14/3, p=0.013** |
| expert_precise_reason vs expert_incorrect_only (does content add anything on top of authority?) | 6/8, p=0.79 | 9/11, p=0.82 |
| precise_reason vs incorrect_only (does content add anything at all?) | 8/8, p=1 | 10/11, p=1 |

Reading: the FIRST significant effect in the whole self-correction study is a
presentation effect. Telling the team that an independent expert has
verified its answer is wrong raises recovery from ~6.5% to ~10.5% (+4pp,
p~0.01) and the answer-change rate from ~0.37 to ~0.50. Attaching Model A's
diagnosis (agent / step / error type) to that message adds nothing (p=0.8),
and the diagnosis without the authority framing adds nothing over "you are
wrong" (p=1). Per benchmark the gain is on gpqa/mmlu_pro/math; scibench is
flat at ~2%. So the lever is the recipient's willingness to abandon its
answer, not the information content -- consistent with everything else in
TODO 1/2a. Honest caveat: n=291 paired, two conditions borderline
individually and significant pooled; one more independent replication
(other framework, see TODO 4) before this goes in a paper.

(Original 118-trace table below, kept for the record.)

Same 150 traces, debate_round. Content held fixed; only the framing varies.

(Restated on the 118 true failures, fixed parser + tolerant grader, 24-trace re-run spliced.)

| condition | recovery | changed answer | correct given changed | vs neutral counterpart |
|---|---|---|---|---|
| expert_incorrect_only ("an independent expert reviewer has verified your answer is WRONG", no content) | 10/118 = 8.5% | **0.475** | 0.179 | vs incorrect_only 5.1%: 5/1 discordant, p=0.219 |
| expert_precise_reason (authority lead + precise_reason content) | 10/118 = 8.5% | 0.398 | 0.191 | vs precise_reason 4.2%: 5/0, p=0.0625 |
| deferential_system_precise_reason (system prompt: "verified incorrect; re-derive from scratch") | 7/118 = 5.9% | 0.492 | 0.121 | vs precise_reason: p~0.5 |

Reading: authority framing DOES move behaviour -- the team abandons its
answer 47% of the time instead of 36% -- and the two expert-framed cells are
the highest in the whole 11-condition grid (8.5%, tied with wrong_location),
with expert_precise_reason vs precise_reason at 5/0 discordant (p=0.0625,
one trace short of significance at n=118). But note what does NOT change:
expert framing with NO diagnostic content does exactly as well as expert
framing WITH it (10 vs 10). So if there is a framing effect, it is
"an authoritative 'you are wrong' makes the team re-guess more", not "the
team uses the diagnosis". Worth one more run at larger n before claiming
anything; content-wise the null stands.

## TODO 2a: feedback-side controls under `debate_round` -- DONE

Same seeded 150 self-play failures as the main debate_round run (math 50,
scibench 68, mmlu_pro 17, gpqa 15), same 3-agent Qwen-7B team, feedback
injected as an extra round. Controls carry DELIBERATELY WRONG content:

(Restated on the 118 true failures, fixed parser + tolerant grader.)

| condition          | recovery      | changed own answer | correct given changed |
|--------------------|---------------|--------------------|------------------------|
| placebo_reason (wrong error type)          | 6/118 = 5.1% | 0.398 | 0.128 |
| random_attribution (random agent/step/mode)| 8/118 = 6.8% | 0.441 | 0.154 |
| wrong_location (consistent but wrong step) | 10/118 = 8.5% | 0.475 | 0.179 |

Each control >= its correct-content counterpart (p = 1 / 0.375 / 0.25).
Reading: WRONG feedback = RIGHT feedback = NO feedback on this team's own
failures. Location is not being used (wrong_location does not hurt -- it is
the top cell), and an explanation's diagnostic content is not what drives
the rare recoveries (placebo = real). Per benchmark, math is 1-2/38 in every
condition; the only movement is on the multiple-choice sets (gpqa/mmlu_pro),
consistent with re-guessing. Corroborates DECISIONS #22/#24: anchoring-bound,
feedback TYPE is not the lever.

## TODO 3: re-SFT on compressed inputs (clean answer to "do we need the full trace?") -- 1/3 DONE

Same recipe as the full-trace SFT (3B LoRA, 3 epochs), prompts built by the
same variant code as the ablation, evaluated on the variant's OWN prompts.

| model (train input) | kept | val agent/step/mode/all-3 | ALFWorld step/mode/all-3 | zero-shot ablation of full-SFT on same input (val / ALF all-3) |
|---|---|---|---|---|
| full-trace SFT (reference) | 1.00 | 0.976 / 0.956 / 0.826 / **0.815** | 0.534 / 0.257 / **0.203** | -- |
| first_line retrain | 0.10 | 0.973 / 0.801 / 0.531 / **0.492** | 0.125 / 0.199 / **0.051** | 0.390 / 0.206 |
| extractive retrain | 0.65 | 0.968 / 0.934 / 0.732 / **0.725** | 0.373 / 0.241 / **0.132** | 0.612 / 0.138 |
| no_observations retrain | 0.75 | 0.978 / 0.961 / 0.825 / **0.815** | **0.643** / 0.254 / **0.228** | 0.803 / 0.193 |
| no_observations, `--keep_ids` control (same 4,767 rows as full SFT) | 0.75 | 0.976 / 0.956 / 0.831 / **0.820** | **0.627** / 0.267 / **0.232** | -- |

**What the variant removes (precisely):** only `[observation] ... [/observation]` blocks, which exist solely in alfagent (ALFWorld, 311/311, 82% chars kept) and smolagents (val 327/327, 83% kept); every other framework's prompt is byte-identical between `full` and `no_observations`. So the intervention shortens smolagents (the largest training framework) at train time and strips ALFWorld's env text at test time -- both single-agent+environment traces, which is why the gain lands on ALFWorld.

**Confound CLOSED (08:50):** the data-size control reproduces the result on
exactly the full SFT's 4,767 training rows: val identical (0.820 vs 0.816,
p=0.89), ALFWorld step 0.534 -> 0.627 (46 vs 17 discordant, p=3.4e-4), all-3
0.203 -> 0.232 (20 vs 11, p=0.15); and it is statistically identical to the
5,016-row run (step 18/23, p=0.53). The +10-11pp OOD step gain is a property
of the INPUT FORMAT (no environment observations), not of the extra rows.

**no_observations reading -- the transfer-gap lever is real.** In-distribution
it is indistinguishable from the full-trace SFT (all-3 0.815 vs 0.816;
paired McNemar p = 1 / 0.63 / 0.89 / 1 on agent/step/mode/all-3). On
ALFWorld it lifts step localization 0.534 -> 0.643 (discordant 51 vs 17,
p = 4.5e-5), mode is flat (0.254 vs 0.260), all-3 0.203 -> 0.228 (17 vs 9,
p = 0.17). So: environment/tool observations carry no attribution signal
in-distribution and are pure noise out of distribution; a Model A trained
without them is the same at home and better abroad. Caveat being closed
now: it trained on 5,016 rows vs the full SFT's 4,767 (shorter prompts fit
the 8,192 cap); the `--keep_ids` control (same 4,767 ids) is RUNNING on A100
#2 (`sft_no_obs_keepids`). If it reproduces the +11pp, the result is
attributable to the input format, not the extra 5% of data.

Ladder verdict (3/3 retrains): agent-prose-preserving compression
(no_observations, 0.75) is free; prose-removing compression costs mode
in-distribution and step OOD, monotonically in the amount of prose removed
(extractive 0.65: -9pp val / -7pp ALF all-3; first_line 0.10: -32pp / -15pp).
What Model A needs is the agents' own text; everything else is optional.

extractive reading: retraining recovers half the zero-shot gap in-distribution
(0.61 -> 0.73; step nearly intact at 0.93, the residual -9pp is mode) -- so
the number-filter's cost is partly format, partly the discarded prose that
carries the "why". OOD it does NOT recover anything (0.132 vs 0.138
zero-shot; step 0.37 vs full's 0.53): training on filtered text again hurts
transfer, less catastrophically than first_line but in the same direction.
Pattern after 2/3 retrains: every compression that removes agent prose costs
mode in-distribution and costs step OOD; the compression ladder is monotone.

first_line reading: (i) in-distribution, retraining recovers a quarter of
the zero-shot gap (0.39 -> 0.49) and no more; step stays usable (0.80) but
mode collapses (0.53 vs 0.83) -- the error TYPE is written in the body of a
step, not its first line. Real information loss, not format shift (eval_loss
plateaued at 0.079 from step 400; full SFT reached 0.024). (ii) OOD it is a
disaster: predicts step "1" for 256/311 ALFWorld traces and mode R.4 for
282/311 -- trained on stubs, the model learned positional/label priors
instead of reading. The full-SFT model given the same compressed ALFWorld
input zero-shot scored 0.592 step; the compressed-trained model scores
0.125. Training-time compression at this ratio destroys transfer.

## Location precision: x / x+2 / x-2 / none (researcher, 2026-09-18) -- DONE, NULL FOR BOTH RECIPIENTS

Same `precise_only` template, only the step (and that step's agent) changes: correct step x,
two steps later, two steps earlier, or no step (`incorrect_only`). Run on account ee5239.

**Model B (the erring team, framework-native extra round; x = Model A's predicted step).**
Offsets exist only where x+-2 stay inside the trajectory (magentic errors sit at round 0 for
98/150 traces; debate has 6 steps; mathchat 1-2).

| framework | none | x | x+2 | x-2 | key paired tests |
|---|---|---|---|---|---|
| magentic-one (150) | 3.3 | 4.7 | 3.2 (n=126) | 4.0 (n=50) | x-2 vs x **0/0 discordant** on 50 shared traces; x+2 vs x 2/4 p=0.69 |
| debate (118 true failures) | 5.1 | 5.9 | 7.1 (n=84) | 8.6 (n=70) | x+2 vs x 2/0 p=0.5; x-2 vs x 3/1 p=0.63 |
| mathchat (150; mostly too short) | 7.3 | 7.3 | 2.0 (n=51) | 0.0 (n=21) | all n.s. (0/1, 0/2) |

**External reader (fresh Qwen-7B, gold step; 434 Who&When traces where both offsets exist:
debate 204, magentic-one 135, dylan 89, mathchat 6; paired, n=434):**

| condition | all | magentic-one | debate | dylan | 5-7 steps (264) | 8-11 (135) | 12+ (35) |
|---|---|---|---|---|---|---|---|
| incorrect_only (none) | 38.5 | 17.0 | 39.7 | 68.5 | 34.8 | 49.6 | 22.9 |
| x-2 | 39.9 | 19.3 | 41.7 | 67.4 | 36.7 | 50.4 | 22.9 |
| **x (precise_only)** | 39.6 | 17.8 | 41.7 | 68.5 | 36.7 | 49.6 | 22.9 |
| x+2 | 39.6 | 18.5 | 42.6 | 65.2 | 37.5 | 48.1 | 22.9 |
| reason_only (for reference) | 41.5 | 17.0 | 46.1 | 67.4 | 40.2 | 49.6 | 20.0 |

x+2 vs x **16/16 (p=1)**; x-2 vs x 11/10 (p=1); x+2 vs x-2 23/24 (p=1); every pointer vs none
p >= 0.31; no length-bucket pattern; magentic-only (the long agentic traces, n=135) all
pointers 17-19%, all p >= 0.25. (reason_only vs none on these 434: 24/11, p=0.04 -- the
small "why" effect again.)

Reading: **the exactness of a step pointer in a feedback message is irrelevant to every
recipient we have** -- the erring team ignores it (0/0 discordant at magentic; wrong =
right at debate), and a fresh external reader does exactly as well with a pointer two
steps off as with the true one (16/16). Together with the earlier precise_only < none on
the full 2,328 (-1.9pp), the picture is: a bare pointer is at best unused and at worst a
distraction, and being off by two changes nothing. Consequences for the efficiency
direction: (a) for any message-based use, the attributor's step precision is not worth
paying for -- a coarse or upper-bound index costs nothing; (b) the location can only have
value where it is USED mechanically -- as a cut point for replay or as the locus of a
training update (Attr-GRPO) -- which is where the remaining experiments go.

## ALFWorld single-agent track (researcher's pivot, 2026-09-20): step-localised, verified Attr-OPD -- STEP A RUNNING

Substrate: ReAct-over-admissible-commands policy (`alfworld_agent.py`, now with KV-cached command
scoring: 5 s/episode on an A40 vs 29 s), PDDL-planner oracle t* (`oracle_alfworld.py`; with binary
search over the prefix 20 s/trajectory, 150 trajectories in 4 min on 16 CPU cores), Who&When-style
alfagent release rows for Model A. New scripts: alfworld_collect / alfworld_expert_collect /
alfworld_oracle_label / alfworld_attribute / alfworld_sft / alfworld_step_a_report.

First diagnostic (zero-shot Qwen2.5-3B-Instruct, budget 30, T=1, 464 train games): success 12.4%;
EVERY failure used all 30 steps, and the oracle put t* at steps 24-29 in 150/150 labelled failures
(relative position 0.90) -- i.e. "the step at which the remaining budget became too short", not a
wrong decision. Greedy zero-shot 3B on the 134 unseen games: ~1.5% (repetition loops). So for an
incompetent policy ALFWorld failures are budget exhaustion and, like the debate failures, have no
decisive step; localisability needs a competent policy. Teacher check: Qwen2.5-14B-Instruct greedy
54% on the first 100 unseen games at budget 30 -- strong enough to teach.

Two oracles, two answers -- and Model A tracks the ORIGIN, not the point of no return (150 zero-shot
failures, 2026-09-20 03:05). Added a progress oracle (`progress_trace`: PDDL shortest-plan length after
every prefix, 3 s/trajectory): t_first_waste = first action that does not shorten the plan (step 0 in
111/150 -- the zero-shot policy wastes from the start), t_first_regress = first action that LENGTHENS it,
i.e. the first harmful decision (defined in 112/150, mean step 7.6). Tasks are short: initial plan
length 3.6 actions.

| Model A variant | vs point of no return t* (n=150) | vs first harmful action t_first_regress (n=112) | random step |
|---|---|---|---|
| 7B full input | exact 1.3%, within 2: 5.3% | exact 28.6%, within 2: 41.1% | exact 0.9-2.7%, within 2: 11-16% |
| 7B agent-text-only (no_obs) | exact 2.7%, within 2: 7.3% | **exact 46.4%, within 2: 55.4%** | same |

So Model A, trained on Who&When's human-marked "first decisive error", points at the first harmful
action (mean |diff| 4.0 steps for no_obs, 83% at or before it) and never at the budget-slack step
(mean |diff| 20). For a training-time intervention that is the right target: it is where a better
action existed and where budget remains. Also re-confirms the format finding on ALFWorld: the
agent-text-only attributor localises better than the full-input one (46% vs 29% exact). Decision:
the ALFWorld "oracle arm" uses t_first_regress; Model A arm uses the 7B no_obs attributor.

Zero-shot greedy baselines on the 134 unseen games at budget 50 (2026-09-20 04:30): Qwen2.5-3B 3.0%,
7B 44.8%, 14B 54.5% (14B at budget 30: 48.5%). The 3B student is the one that needs warm-start SFT; a 7B
student would be competent out of the box (kept as a later ablation).

Teacher quality at the attributed state (408 zero-shot failures, state = oracle's first harmful action;
scored by whether the proposed action shortens the planner's shortest completion, 2026-09-20 06:00):

| action at that state | makes progress | makes things worse |
|---|---|---|
| the student's original action | 0.0% | 81.1% (by construction) |
| a random admissible action | 5.4% | 15.4% |
| Qwen2.5-14B teacher, hindsight hint | 26.5% | 9.3% |
| Qwen2.5-14B teacher, no hint | 28.9% | 5.4% |

The 14B teacher's greedy choice is 5x better than random and rarely harmful, but ~70% of its choices are
neutral (look/examine -- rational under partial observability, invisible to the full-knowledge planner),
and the hindsight hint does not help at this scale (slightly worse). Consequences: (a) the teacher's top-1
alone is a weak target, so the verified variant (top-k alternatives continued by the student, keep the
ones that succeed) is the main method, not an ablation; (b) "makes progress" is a lower bound on
usefulness -- success of the continued rollout is the fair criterion; (c) the hint stays as an option
(`--no_hint` to drop it) rather than a load-bearing component.

SFT warm start result (2026-09-20 06:05): 3B student SFT'd on 549 expert trajectories (2 epochs, loss
0.66 -> 0.04) scores **89.6% greedy on the 134 unseen games** (120/134; all 14 failures are budget
exhaustion) and 82% at T=1 on train games -- above the 14B zero-shot teacher (54.5%). Two consequences,
both fixed the same way: (a) little headroom on unseen for any post-training method; (b) a teacher must
be stronger than the student. Redesign: the OPD student is a LOW-DATA warm start (3B SFT on 100
trajectories, 1 epoch; expected 50-65%), the teachers are full-data SFT models (the 3B-549 model at
89.6% now, a 7B SFT on all 1,338 trajectories being trained). Same-family expert-to-novice distillation
keeps the "where to distill" question well-posed and gives the efficiency read-out real headroom. The
strong student's T=1 failures (~18% of 1,500 games) are still collected as a "competent-policy failure"
set for the localisability analysis. 7B teacher SFT (1 epoch on all 1,338 expert trajectories, 07:45): 80.6% greedy on unseen (108/134) --
below the 3B-549 model (89.6%, 2 epochs). A 2-epoch 7B SFT scores 79.1% (106/134), no better than 1 epoch -- the 3B
full-data model (89.6%) stays the teacher for all rounds (same family: expert-to-novice distillation).

Competent-policy failures are NOT more localisable (strong 3B student, 89.6%; 158 T=1 failures on 1,330
train games, partial oracle labels, 09:35): every failure is budget exhaustion at 50 steps; a first harmful
action exists in only 63% (mean step 18.6, relative 0.37 -- the other 37% never take a distance-increasing
action, they search without finding); the point of no return sits 26 steps later (relative 0.89). Model A
(no_obs) calls 128/158 "task misunderstanding" (R.4), points very early (relative 0.17) and hits the first
harmful action at chance (exact 6.6%, within 2: 9.2% vs random 9.2%) -- versus 46% / 55% on the zero-shot
policy's failures. Reading: localisability is a property of the failure TYPE, not of policy competence.
Salient wrong decisions (taking the wrong object) are found; search failures under partial observability
(the object is somewhere not yet looked at) have no decisive step and Model A's pointer carries no
information. The low-data (42%) student's failures should be a mixture; the OPD read-out will be split by
"first harmful action exists" vs not.

Low-data student's failures ARE localisable (544 T=1 failures on 1,323 train games; first 46 oracle
labels, 10:20): a first harmful action exists in 100% and comes at step 2.0 on average (step 1 in 26/46):
the novice heads to the wrong place or grabs the wrong thing at once, then spends the remaining budget
searching. Model A (no_obs) tracks it: within 1 step 52%, within 2 steps 72% (random 4%), mean |diff| 3.7,
89% at or before it. Full labels pending; this is the failure set round 1 of Attr-OPD trains on.
Contrast with the strong student (final labels): first harmful action in 61% (mean step 18), Model A within
2 = 12.6% (random 9.5%).

### Model A fine-tuned on ALFWorld self-play labels: localisation triples on hard residual failures (06:20)

1,099 self-play failures (zero-shot, low-data round 1, oracle round 2, strong student) labelled for free by
the progress oracle (first harmful action; mode copied from Model A's own earlier call), one LoRA epoch
continued from the 7B agent-text-only attributor (10 min). Held-out test = the oracle round-3 failure set
(242 labelled failures of a 78%-success policy, first harmful action at step 13.7 on average -- the hard,
late-error regime where the original Model A had degraded):

| attributor | exact | within 1 | within 2 | mean \|diff\| |
|---|---|---|---|---|
| Model A, Who&When-trained only | 19.0% | 24.4% | 27.7% | 13.0 steps |
| **Model A + 1,099 ALFWorld self-play labels** | **58.3%** | 60.3% | 65.7% | 4.8 steps |
| random step | 2.9% | | 9.9% | |

This is the attributor curriculum the plan asked for: the environment's own progress signal supplies
labels, so the attributor tracks the policy as it improves. attralf_nofb round 1 on the shared 319 failures (07:35): 122
verified examples, unseen 76/134 = 56.7% (vs student 28/8, p=0.001; vs attr_nofb 11/13, n.s.) -- as expected
on round-1 failures, where the original attributor was already accurate. The decisive test is round 2 with
the refreshed attributor (launched 07:36 on hold 1436314, roundN_v2), where attr_nofb regressed to 49.3%.

Refreshed attributor on the regression set itself (the 528 T=1 failures of the attr_nofb round-1 policy that
attr_nofb round 2 trained on; 315 progress-oracle labels so far, 08:20):

| attributor | exact | within 1 | within 2 | mean \|diff\| | at/before |
|---|---|---|---|---|---|
| Model A, Who&When-trained only (what attr_nofb round 2 used) | 19.4% | 37.1% | 44.8% | 10.5 steps | 67% |
| **Model A + 1,099 ALFWorld self-play labels** (what attralf_nofb round 2 uses) | **77.5%** | 80.0% | 83.2% | 1.7 steps | 84% |
| random step | 1.9% | | 8.9% | | |

The original attributor pointed 6 steps too late on average (mean t_hat 11.3 vs first harmful action 5.4) --
i.e. it attributed the budget-exhausted search, not the wrong decision -- so the verified step-search rarely
found a fix (verified-fix rate 29%). The refreshed attributor puts the pointer within 1 step of the oracle's
first harmful action 80% of the time on this policy's failures. Caveat: these failures come from the same
1,500 train games the attributor's self-play labels came from (different trajectories, different policies);
a game-disjoint check on held-out games 1500-1999 (failures of the student and of the attr_nofb round-1
adapter, both attributors) is running. Per-task view of the regression (`alfworld_task_breakdown.py`, pooled
774 games): attr_nofb round 2 lost 12pp on cool and 11pp on heat tasks vs its round 1 while look_at tasks
kept improving -- the late-error, multi-stage tasks are exactly where the un-refreshed pointer drifted.

attralf_nofb round 1 extended (08:47): seen 60.7%, held-out 59.0%, unseen 56.7% -> pooled **58.9%** -- below
attr_nofb round 1 (64.3%; 51/93, p=6e-4) and below oracle_nofb (63.0%; 30/62, p=0.001), although the refreshed
attributor's chosen step equals the planner oracle's in 307/319 states (96%, mean |diff| 0.18 -- it was trained on
these very labels) and 107 of its 122 verified examples are the same (state, target) pairs as oracle_nofb's. Two
near-identical recipes differing by ~15 examples and the shuffle order (online KL updates, batch 8, ~16 optimizer
steps) differ by 4pp pooled, i.e. the run-to-run variance of one distillation round is of the same order as the
differences among the verified-only round-1 arms (58.9-64.3%). Controls launched (hold 1521595): (i) a
training-seed replicate of attr_nofb round 1 (same 319 states and targets, seed 1; `round1_seed1`), (ii) a
counterfactual check of the verified targets (`alfworld_verify_counterfactual.py`): the failed episode was a T=1
sample and the verification continues greedily, so a "verified fix" at step 0-2 may be a game the greedy student
solves anyway -- the check replays each verified state and continues greedily after the ORIGINAL action too; only
alt-wins-and-original-loses is a causal correction. Also noted: the original Model A put its pointer at step 0 in
185/319 round-1 states (58%) -- "within 2 steps of the first harmful action" is partly a floor effect when the
first harmful action is at step 1-2 -- and 27 of attr_nofb's 125 verified targets are `look`.

Counterfactual check result, attr_nofb round 1 (09:10; initial student, greedy continuation, cap 20):

| verified target (125) | causal: alt wins, original loses | both win (recoverable anyway) | neither wins | original only |
|---|---|---|---|---|
| all | **69 (55%)** | 24 (19%) | 28 (22%) | 4 |
| `go to ...` (91) | 61 (67%) | 15 (16%) | 14 (15%) | 1 |
| `look` (27) | 7 (26%) | 8 (30%) | 10 (37%) | 2 |
| step 0 (82) | 47 (57%) | 18 (22%) | 14 (17%) | 3 |

So about one verified target in five is not a correction (the T=1 failure was recoverable under greedy play from
that state with the student's own action), and `look` targets are mostly neutral; the "neither" rows are states
whose alternative won only with the adapter as it stood mid-training (verification runs with the evolving
student). Fix implemented: `--contrastive` in `alfworld_attr_opd.py distill` also continues after the student's
original action and skips the state when it wins too (only causal corrections become targets; +1 continuation
per state). Arm `attr_nofb` `round1_cv` (same 319 states and teacher targets, seed 0) queued on hold 1521595 behind
the counterfactual checks of oracle_nofb and attralf_nofb. Runner fixed: the chunk loop now ends only on a NEW
"DONE distill" marker in the shared run.log (the stale-marker bug that cut the k=3 run short).

oracle_nofb round 1, same check (09:30): 124 verified targets -> causal 65 (52%), both win 30 (24%), neither 22 (18%),
original only 7 -- the same profile as attr_nofb; the non-causal fifth is a property of the verification, not of
the pointer.

**Game-disjoint attributor test and the constant-step baseline (09:40) -- REVISES the round-1 reading.** Both
attributors on failures of held-out games 1500-1999 (never in the attributor's self-play training data; greedy
failures, progress-oracle labels), plus the strongest attribution-free baseline, "always point at step k":

| failure set (first harmful action, mean step) | original Model A exact / within-2 | refreshed Model A exact / within-2 | constant step 1 exact / best constant within-2 | random exact / within-2 |
|---|---|---|---|---|
| held-out games, student's failures (n=174, step 1.7) | 26.4% / 74.7% | **96.6% / 96.6%** | 79.9% / 92.5% | 1.7% / 6.3% |
| held-out games, attr_nofb round-1 policy's failures (n=134, step 5.0) | 32.1% / 51.5% | **88.8% / 89.6%** | 33.6% / 69.4% | 0.7% / 10.4% |
| train games, round-2 attr_nofb set (n=315-382, step 5.4) | 19.4% / 44.8% | 77.5% / 83.2% | 26.2% / 67.8% | 1.9% / 8.9% |
| train games, round-1 set (n=450, step 2.0) | (11%) / 72% | -- | 59.8% / 90.4% | 4% |

Reading: (1) the Who&When-trained Model A is at or BELOW the constant-step baseline on every ALFWorld set --
its round-1 "within 2 steps 72%" was a floor effect (it put the pointer at step 0 in 58% of round-1 states, and
the first harmful action is step 1-2 in 76% of them); on ALFWorld the original attributor carries no information
beyond the "early error" prior. (2) The refreshed attributor is far above the constant baseline where it matters
(late, spread-out errors: 89% vs 34% exact on held-out games) -- the curriculum produced a real localiser,
not memorisation of games. (3) Therefore the round-1 result "attr_nofb = oracle_nofb" does not show that Model A
localises; it shows that "fix the FIRST decision, verified" is enough when the novice's errors are all early. The
missing control is a constant-step arm: `const1_nofb` (always step 1, verified-only, same 319 failures, same
teacher) -- added to choose/runner, queued on the first preempted hold to return. The attribution-vs-prior question
is decided on later rounds, where the constant step is right 26-34% of the time and the refreshed pointer ~85%:
attralf_nofb round 2 (relaunching) vs attr_nofb round 2 (49.3%) vs oracle_nofb round 2 (running) -- and a
const1 round 2 on the same failures is the control to add if attralf round 2 compounds.

09:33 preemption wave (agd04/06/08): holds 1436314 (attralf_nofb round 2, recollection at 1,195/1,500), 1502947
(attr_nofb round 3, distill chunk 2 at k=140) and 1435620 (oracle_nofb round 2, waiting on CPU labels) requeued.
The chain scripts keep running through failed sruns after a preemption (attralf wrote an empty steps file, then
exited at the teacher stage) -- killed both, removed the empty file, truncated round 3's examples/skipped to the
chunk-1 adapter state (109/141; chunk-2 updates were never saved), and queued both relaunches on the
non-preemptible igpu hold behind the k=3 extended eval (collection resumes from the 1,195 episodes).

attralf_nofb round 2 relaunched 09:44 (igpu hold): 1,500 games, 575 failures (T=1 success 61.7%), refreshed attributor
chose 575 states (10:30). Where the two attributors point on round-2 failures (their own policies' failure sets):

| attributor (round-2 failure set) | mean step | median | step 0 | step 1 | steps 2-4 | step >= 10 |
|---|---|---|---|---|---|---|
| original Model A (attr_nofb round 2, n=528) | 11.1 | 2 | 38% | 11% | 10% | 33% |
| refreshed Model A (attralf_nofb round 2, n=575) | 8.1 | 6 | 0% | 8% | 26% | 33% |
| original Model A (attr_nofb round 3, n=632) | 13.6 | 5 | 36% | 6% | 7% | 44% |

The original attributor is bimodal -- either the very first action (never the first harmful action, which is
step 0 in 0% of labelled failures) or deep inside the budget-exhausted search -- while the refreshed one puts the
mass at steps 2-6 where the progress oracle puts the first harmful action (mean 5.4). Diagnostic labels for the
575 failures: CPU job 1560033.

10:45 GPU contention on the igpu hold: two distills (attr_nofb round 3 at 25.6 GB after 49 min, attralf round 2 at
19.2 GB) filled the 46 GB card and attralf hit an allocator OOM warning. Stopped the round-3 distill (secondary) to
protect the decisive run; its examples/skipped were re-truncated to the chunk-1 adapter state (109/141). The
oracle_nofb round-2 chain reached the choose stage (360 labelled states of 528 failures) and then died at the
teacher stage because its hold (1435620) is still requeued. Both remainders plus the three controls are now in one
dispatcher (`scripts/alfworld_dispatch_queue.sh`, one task per GPU hold, priority: oracle_nofb r2 remainder >
const1_nofb > attr_nofb r3 resume > seed replicate > contrastive arm) that fires as holds return; the igpu hold
takes one of them once the attralf distill there has finished. Rule learned: at most two 3B distills per 46 GB
GPU, and the decisive run gets the card alone when in doubt.

Queue mechanics from 10:54 (SLURM's start estimates for the five requeued/backup holds are 1.5-2.5 days out):
`scripts/alfworld_task.sh <task> <hold>` runs one task under a mkdir lock; `alfworld_igpu_queue.sh` runs the tasks
sequentially on the igpu hold after attralf round 2 finishes (order: attralf r2 extended eval > oracle_nofb r2
remainder > const1_nofb > attr_nofb r3 resume > seed replicate > contrastive arm); `alfworld_dispatch_queue.sh`
places any still-unlocked task on a preempt/backup hold the moment one restarts; `alf_igpu_renew` resubmits the
igpu hold when the current one ends (~04:00) and writes the new id to `reports/alfworld/igpu_hold.txt`.

Fixability vs pointer position (13:40; `alfworld_fixability_report.py`, progress-oracle labels on each round's own
failure set; "verified-fix rate" = share of supervised states where a top-2 teacher alternative led to success):

| arm / round | overall | pointer = oracle step | pointer earlier by 1-2 | pointer later by >2 | first harmful step 1-2 | 3-6 | >6 |
|---|---|---|---|---|---|---|---|
| attr_nofb round 1 (original A) | 39% | 47% | 47% | 22% | 41% | 37% | 16% |
| oracle_nofb round 1 | 39% | 39% | -- | -- | 39% | 45% | 16% |
| attr_nofb round 2 (original A) | 32% | 38% | 49% | 17% | 40% | 29% | 25% |
| attralf_nofb round 2 (refreshed A) | 42% | 31% | 83% (n=6) | 62% (n=63) | 59% | 42% | 26% |

Two readings. (1) Fixability is mostly a property of WHEN the failure starts: failures whose first harmful action is
at step 1-2 are fixed 40-59% of the time, those starting after step 6 only 16-26% -- every chain's later rounds
face a harder residual. (2) Agreement with the planner oracle's step is NOT what buys fixes: in attralf round 2 the
states where the refreshed attributor matches the oracle exactly are fixed 31% of the time, the states where it
points >2 steps LATER 62% -- the "first plan-lengthening action" is often a mild detour, and the salient wrong
decision the attributor flags later is the fixable one. So "exact vs oracle" is the wrong yardstick for an
attributor whose job is to pick intervention points; the verified-fix rate and the end-to-end success are the
yardsticks, and the refreshed attributor raises the fix rate from 32% to 42% on comparable round-2 failure sets.

### DECISIVE: refreshed attributor round 2 compounds -- 92/134 = 68.7% unseen (2026-09-21 13:34)

Same recipe as attr_nofb round 2 (recollect T=1 failures with the round-1 adapter, one step per failure, top-2
teacher alternatives verified by a 20-step student continuation, verified-only targets, KL at that step), but the
step comes from the refreshed attributor (Model A + 1,099 ALFWorld self-play labels) on its own round-1 adapter's
1,500-game recollection: 575 failures (T=1 success 61.7%), 575 states, **237 verified fixes (41%)**, ~29k env steps.

| adapter | unseen 134 | paired vs attralf round 2 (other-only / attralf-only, p) |
|---|---|---|
| student (100-trajectory SFT) | 41.8% | 6 / 42, p=1e-7 |
| attralf_nofb round 1 | 56.7% | 7 / 23, p=0.005 |
| attr_nofb round 2 (original Model A, same recipe) | 49.3% | 4 / 30, p=6e-6 |
| **attralf_nofb round 2 (refreshed Model A)** | **68.7%** | -- |
| oracle round 2 (planner pointer, with fallback) | 75.4% | 18 / 9, p=0.12 |
| oracle round 3 | 76.9% | 18 / 7, p=0.04 |
| teacher (3B, 549 trajectories) | 89.6% | 31 / 3, p=8e-7 |

Reading: with a pointer that tracks the improving policy, the Model A chain compounds like the oracle chain did
(56.7 -> 68.7 vs 58.2 -> 75.4 for the oracle; the original-attributor chain went 58.2 -> 49.3 on the same recipe).
The gap to the oracle chain at round 2 is 6.7pp (n.s. on 134 games); the verified-fix rate rose from 29% (original
pointer) to 41% (refreshed) on comparable round-2 failure sets, versus 54% for the planner pointer. Extended
(774-game) evaluation running; controls queued on the same GPU: `const1_r2` (always step 1 on the same 575
failures from the same start adapter -- the attribution-free control at round 2), oracle_nofb round 2 remainder
(like-for-like verified-only oracle), and the round-3 refresh (`attralf_r3`: attributor v2 trained with the
round-2 failure sets added, then round 3).

**Pooled 774-game read-out (14:27): attralf_nofb round 2 = 590/774 = 76.2%, statistically identical to the planner-oracle
chain's round 2 (592/774 = 76.5%; 62/60, p=0.93).** Seen 112/140 = 80.0%, held-out 386/500 = 77.2%, unseen 68.7%.
vs student 53.7%: 200/26, p=2e-34 (+22.5pp). vs the same recipe with the stale pointer (attr_nofb round 2, 56.8%):
177/27, p=3e-28. vs its own round 1 (58.9%): 157/23, p=1e-25. Per task type the refreshed chain fixed exactly the
multi-stage tasks the stale chain lost: clean 57.5 -> 79.5%, cool 43.8 -> 74.6%, heat 51.8 -> 85.7% (oracle round 2:
86.3 / 80.0 / 70.5). Headline for the paper: a learned attributor, refreshed each round from the environment's own
progress labels (no human labels), reaches the privileged planner oracle's downstream gain at round 2 on 774 games;
the same attributor left un-refreshed regressed. Pending controls on the same 575 failures: const1_r2 (running,
attribution-free) and oracle_nofb round 2 (verified-only oracle, the exact like-for-like).

### const1_r2: the attribution-free control at round 2 -- REVISES #43, doesn't overturn it (2026-09-22 01:52)

Same 575 failures, same starting adapter (attralf_nofb round 1), same teacher, same nofb recipe as attralf_nofb
round 2; only the supervised step is fixed at step 1 for every failure (no attributor call at all):

| adapter | unseen 134 | seen 140 | held-out 500 | pooled 774 |
|---|---|---|---|---|
| student | 41.8% | 57.1% | 56.0% | 53.7% |
| attr_nofb round 2 (stale attributor) | 49.3% | 60.7% | 57.8% | 56.8% |
| attralf_nofb round 1 (starting adapter) | 56.7% | 60.7% | 59.0% | 58.9% |
| **const1_r2 (no attribution, always step 1)** | 73.9% | 73.6% | 71.4% | **72.2%** |
| **attralf_nofb round 2 (refreshed attributor)** | 68.7% | 80.0% | 77.2% | **76.2%** |

On the 134 unseen games alone, const1_r2 was numerically HIGHER than attralf_nofb round 2 (73.9% vs 68.7%, p=0.09,
n.s.) -- a result that looked like it might overturn the attribution claim. The pooled 774-game read-out reverses
this: attralf_nofb round 2 beats const1_r2 by 4pp (65/34, p=0.0024). Reading: the verified-only OPD recipe itself
(teacher proposes, student-continuation verifies, only verified fixes become targets) carries most of the gain
even with NO attribution at all -- const1_r2 alone lifts the student 18.5pp (53.7% -> 72.2%). This is not surprising
given the pointer-offset finding above: round-2 failures still have enough early-error mass that "always step 1"
rarely overshoots into the costly +2-style region. On top of that large recipe-driven gain, a refreshed attributor
adds a further, statistically real 4pp at 774 games. Framing for the paper: attribution is not the source of the
gain, it is a smaller but real multiplier on top of a strong attribution-free baseline -- report both numbers, not
just attralf-vs-student.

### t* (point-of-no-return oracle) as the pointer -- the weakest of the three exact-position methods (2026-09-22, researcher's request)

Same recipe as oracle_nofb/attr_nofb (verified-only, teacher = 3B-549, cap 20 continuation steps), but the
supervised step is `t*` (`find_critical_step`: binary search for the last prefix from which the handcoded-expert
planner can still win within budget -- "the point of no return", not "the first harmful action"). Coverage: 233 of
the 319 shared round-1 failures have a t* label (86 hit the 900s binary-search timeout on the first pass; a 1800s
retry is running in the background, non-blocking), so `tstar_nofb` and its offset variants use only those 233.

| pointer | verified examples | verified-fix rate | unseen 134 | seen 140 | held-out 500 | pooled 774 |
|---|---|---|---|---|---|---|
| t_first_regress (oracle_nofb) | 124/319 | 39% | 56.0% | 67.9% | 63.6% | 63.0% |
| Model A's own step (attr_nofb) | 125/319 | 39% | 58.2% | 65.0% | 65.8% | 64.3% |
| **t\* (tstar_nofb)** | 20/233 | **8.6%** | 53.0% | 63.6% | 59.6% | **59.2%** |

t* is significantly worse than both at the pooled level (vs oracle_nofb 92/62, p=0.019; vs attr_nofb 99/59,
p=0.0018) though not on 134 unseen games alone (p=0.57/0.23) -- another case where the 134-game split understates a
real difference the 774-game pool resolves. Mechanism: t* sits deep in the trajectory by construction (mean
relative position 0.90 in the zero-shot diagnostic), where the teacher's proposed alternative rarely has enough
remaining budget to complete the task -- the verified-fix rate (8.6%) is the lowest of any exact-position arm
tried. Reading: "the point of no return" is a well-defined causal label but a poor training target under a fixed
budget; "the first harmful action" (either oracle-labelled or Model A's own prediction) is both more frequent and
more fixable. tstar_nofb_p1/_p2 (researcher's request) running.

tstar_nofb_p1 and _p2 (t*+1, t*+2): **0/233 verified examples for both** -- by construction t* is the boundary of
recoverability (the last prefix from which the expert can still win within budget), so t*+1 is already past that
boundary: no teacher-proposed alternative can complete the task within the remaining budget from there, and
neither can t*+2. Zero gradient steps means the resulting adapter is byte-identical to the starting checkpoint, so
evaluation was skipped and the plain student's numbers are reused (41.8%/57.1%/56.0%, pooled 53.7%) rather than
re-running ~70 minutes of GPU time for a result known in advance. This is a clean, almost tautological confirmation
of the pointer-offset asymmetry above: t* has no "safe forward margin" at all, unlike t_first_regress (mean step
2.0, so +1/+2 still land inside a fixable region on many failures).

### Pointer-offset controls (researcher's request): how much imprecision does verified-only distillation tolerate? (2026-09-21)

Same 319 round-1 failures, same teacher, same nofb recipe as oracle_nofb; only the supervised step is shifted from
the planner oracle's exact first-harmful-action step by a fixed offset (`--offset`, choose step clamped to [0, n-1]):

| offset | verified examples (of 319) | unseen 134 | seen 140 | held-out 500 | pooled 774 | vs exact step (b/a, p) |
|---|---|---|---|---|---|---|
| 0 (oracle_nofb, exact) | 124 (39%) | 56.0% | 67.9% | 63.6% | 63.0% | -- |
| **+1** | 97 (30%) | 53.0% | 67.9% | 63.0% | **62.1%** | 64/71, p=0.61 (n.s.) |
| **+2** | 84 (26%) | **41.8%** | 57.1% | 53.6% | **52.2%** | 66/78, p=0.36 (n.s. vs student -- no effect) |
| **-2** | 116 (46%) | **53.7%** | 65.0% | 61.0% | **60.5%** | 14/17, p=0.72 unseen (n.s. -- matches exact step); pooled 97/45 p=1.5e-5 vs student |

**Result: the imprecision tolerance is direction-specific, not symmetric in |offset|.** +1 costs verified examples
(39% -> 30% of states) but almost no downstream gain (62.1% vs 63.0% pooled, n.s.). +2 collapses to student level on
all three splits (52.2% pooled, statistically indistinguishable from the untrained student, 66/78 p=0.36) even
though its verified-fix rate declined only gradually (39 -> 30 -> 26%). -2 does the opposite: its verified-fix rate
is the HIGHEST of any offset (46%, vs 39% exact) and its unseen success (53.7%) matches the exact step (p=0.72) and
+1 (p=1), significantly better than +2 (p=0.014). Reading: round-1's failures have their first harmful action very
early (mean step 2.0), so "-2" mostly lands at or near the start of the episode, where the teacher's advice is
still reliable and easy to verify; "+2" overshoots PAST the harmful decision into a state the damage has already
been done in, where teacher-proposed alternatives rarely lead anywhere. The failure mode is "landing after the true
error, not merely far from it" -- a pointer that errs early is nearly free in the verified-only regime, a pointer
that errs late pays close to the full cost. Extended (774-game) eval of -2 done: pooled 60.5% (unseen 53.7%, seen 65.0%, held-out 61.0%) -- 2.5pp below the
exact step (n.s., 67/87 p=0.13) and 8.3pp ABOVE +2 (52.2%), confirming the asymmetry holds at the pooled level, not
just on 134 unseen games.

### Offsetting Model A's own predicted step (not the planner oracle's) -- much less tolerant of +1 than t_first_regress (2026-09-22)

Same idea, but the base pointer is Model A's own t_hat (the `attr_nofb` arm) instead of the planner oracle's
t_first_regress. Same 319 shared failures, same teacher/recipe:

| pointer | verified examples | unseen | seen | held-out | pooled 774 | vs exact (b/a, p) |
|---|---|---|---|---|---|---|
| Model A exact (attr_nofb) | 125/319 (39%) | 58.2% | 65.0% | 65.8% | 64.3% | -- |
| **Model A +1** | 107/319 (34%) | 50.0% | 57.1% | 57.8% | **56.3%** | 41/103, p=2.4e-7 |

+1 on Model A's OWN step costs 8.0pp pooled (vs only 0.9pp for +1 on the planner oracle's t_first_regress) and is no
longer significantly above the untrained student (p=0.11). Reading: t_first_regress is, by construction, exactly
the first plan-lengthening action; Model A's prediction already carries some slack relative to that true position
(it is a LEARNED approximation), so adding +1 on top more often pushes past the true harmful action than +1 on the
oracle's exact label does. The tighter the base pointer already is to ground truth, the more forward margin it has
before the offset asymmetry (#44) bites. `attr_nofb_p2` (Model A +2) running next.

**Operational note:** three concurrency bugs surfaced while running these offset controls back-to-back: (1) editing
`alfworld_task.sh`/`run_alfworld_opd_arm.sh` while the igpu queue's in-flight invocation was still reading them
corrupted that invocation's execution (classic "never edit a running script", re-confirmed); (2) a `--ids_from`
override left unset for a new arm silently fell back to `progress_${TAG}_train_b50.jsonl`, which had grown from 319
to 454 valid entries over the session from later, unrelated CPU labelling runs -- always pin new round-1 arms to
`$A/attr_nofb/steps.jsonl` (or equivalent), never the raw progress file, once other jobs may have appended to it;
(3) `alfworld_task.sh` cases that call `run_alfworld_opd_arm.sh` without `|| exit 1` let a crashed choose/teacher/
distill stage fall through into evaluating a nonexistent adapter -- fixed with an explicit adapter-exists guard.

### Verification breadth ablation: k=3 candidates vs k=2 (attr_nofb round 1 recipe; 2026-09-21 08:40)

Same 319 failures, same Model A steps, same 319 teacher calls; the student continues the teacher's top-3
alternatives instead of top-2 and every state with a verified success becomes a target:

| verify_k | verified examples (of 319 states) | env steps | unseen success | vs k=2 (b/a, p) | vs student |
|---|---|---|---|---|---|
| 2 (attr_nofb round 1) | 125 (39%) | 14.3k | 78/134 = 58.2% | -- | 27/5, p=1e-4 |
| 3 | 158 (50%) | 21.2k | 79/134 = 59.0% | 7/6, p=1.0 | 29/6, p=1e-4 |

A third candidate finds a verified fix in 33 more states (+26%) for +48% environment steps, and the extra
targets buy nothing on unseen (the partial adapter after 250/319 states scored 82/134 = 61.2%, 4/7 vs the
full one, p=0.55 -- 134-game noise of about +-4pp). The third-ranked alternative fixes states the top-2 could
not, but those are states whose fix does not transfer; k=2 stays the default. Extended (09:44): k=3 seen 67.1%,
held-out 67.6%, unseen 59.0% -> pooled 66.0% vs k=2's 64.3% (57/44, p=0.23) -- a +1.7pp trend inside the noise band.

### Pooled 774-game read-out of the round-1 arms (unseen 134 + seen 140 + held-out train 500; 2026-09-21 03:20)

| arm | pooled success | vs student (arm-only / student-only, p) |
|---|---|---|
| student (100-trajectory SFT) | 53.7% | -- |
| **attr_nofb** (Model A step, verified-only targets, 125 examples) | **64.3%** | 130/48, p=6e-10 |
| oracle (first harmful action, with fallback) | 59.7% | 130/84, p=0.002 |
| attr (Model A step, with fallback) | 54.0% | 117/115, p=0.95 (null) |
| random step (with fallback) | 39.5% | 69/179, p=2e-12 (harm) |
| last step (with fallback) | 32.4% | 72/237, p=1e-21 (harm) |
| oracle_nofb (first harmful action, verified-only, 124 examples) | 63.0% | 123/51, p=5e-8 |
| uniform (every step, equal teacher calls, with fallback) | 5.9% | 17/387, p=2e-92 (collapse) |

attr_nofb vs oracle_nofb pooled: 77/67, p=0.45 -- with verified-only targets, Model A's step and the planner
oracle's first harmful action are equivalent.

attr_nofb beats oracle-with-fallback pooled (125/89, p=0.017). Iterated: oracle round 2 (with fallback)
76.5% pooled (see below); attr_nofb round 2 and oracle round 3 running.

### attr_nofb: Model A's step + verified-only targets = the oracle-with-fallback arm (2026-09-21 00:05)

Same 319 failures, same 319 teacher calls, Model A's step, but distil ONLY where a top-2 alternative was
verified to succeed (125 examples; 194 states skipped): unseen **78/134 = 58.2%** -- vs the student 27/5
(p=1e-4), vs attr-with-fallback 20/6 (p=0.009), vs random 39/7 (p=2e-6), and exactly equal to the
oracle-with-fallback arm (18/18). Reading: with the fallback removed, Model A's localisation is good enough
-- the 7B attributor trained on Who&When Pro, never on ALFWorld, gives the same downstream gain as the
planner oracle did in round 1. Verification, not the teacher's opinion, is the source of the signal; the
teacher's role is to propose 2 candidates at the attributed step. And the sampled policy is repaired too: the attr_nofb adapter's T=1 success on the
round-2 collection games is 66% (first 400 games) vs 59% for the untrained student and 43% for the
fallback attr adapter -- verified-only targets sharpen the argmax without flattening the distribution.
**oracle_nofb: 75/134 = 56.0%** (124 verified examples; vs student p=0.001; vs attr_nofb 12/15, n.s.; vs
oracle-with-fallback 10/13, n.s.) -- with verified-only targets Model A's step and the planner's first
harmful action are indistinguishable. attr_nofb extended: seen 91/140 = 65.0%, held-out 329/500 = 65.8%, unseen 58.2% -> **pooled 64.3% vs the
student's 53.7% (+10.6pp, 774 games)** -- the strongest single-round result, ahead of oracle-with-fallback
(59.7% pooled) and obtained from Model A's step with 125 verified corrections.
last adapter extended: seen 32.1%, confirming its harm. attr_nofb round 2 collecting on hold 1521595.

attr round 2 WITH fallback (751 states, 225 verified; 04:30): 67/134 = 50.0% on unseen -- +2.2pp over its
round 1 (47.8%), far below oracle round 2 (75.4%) and below attr_nofb round 1 (58.2%): iterating a
fallback-poisoned recipe barely moves. attr_nofb round 2 (verified-only) is the informative iteration. (attr round 2 extended: seen 55.0%, held-out
56.4% -- null, like its round 1.) To separate "iteration hurts" from "Model A's pointers degrade on the
better policy", oracle_nofb round 2 (verified-only, oracle pointers) launched 05:55 on hold 1435620.

attr_nofb round 2 (verified-only, from the round-1 attr_nofb adapter; 528 failures of the 65%-at-T=1
policy, 155 verified examples; 05:10): unseen 66/134 = **49.3%**, DOWN from round 1 (58.2%; 6/18 vs r1,
p=0.023) though still above the student (41.8%, 27/17, p=0.17). The verified-fix rate also fell, 39% ->
29% of attributed states, i.e. Model A's pointer found a fixable state less often on the better policy's
failures. Diagnostic (progress-oracle labels on 137 of these round-2 failures, 06:20): the original Model A
hits the first harmful action exact 19.7% / within-2 44.5% here vs 11% / 72% on the round-1 set -- the
pointer degraded as the policy's failures moved later (first harmful action at step 5.7 vs 2.0). Cause
identified: attributor drift, not the iteration itself; fix = refresh the attributor (DECISIONS #41).
attr_nofb round-2 extended: seen 60.7%, held-out 57.8%, unseen 49.3% -> pooled 56.8% (round 1: 64.3%;
student 53.7%) -- the regression holds on all splits. Unlike the oracle chain (58 -> 75), the second
verified-only round on Model A's pointers did not compound. Candidates: the round-1 policy's residual
failures are less localisable for Model A (competent-policy effect), or unlucky 134-game noise -- the
extended eval of this adapter is queued; round 3 from it is running anyway.

### Attr-OPD round 2 (iterated) -- oracle arm reaches 75.4% (2026-09-20 23:48)

Recollect T=1 failures with the round-1 oracle adapter (1,281 games, 551 failures), progress-oracle labels,
one verified teacher call per failure (338 states, 183 verified), distil from the round-1 adapter: unseen
success **101/134 = 75.4%** (student 41.8%, round 1 58.2%, teacher 89.6%). Paired: vs student 50/5
(p=2e-10), vs round 1 30/7 (p=0.0002), vs teacher 2/21 (p=7e-5). Cumulative cost: 657 teacher calls and
~28k env steps to move a 41.8% student to 75.4% -- most of the way to a teacher that needed 6,450
supervised pairs. Extended eval of the round-2 adapter (01:00): seen 79.3% (student 57.1%), held-out 76.0% (56.0%), unseen
75.4% (41.8%); **pooled over 774 games 76.5% vs 53.7% (+22.8pp)** -- robust across all three splits, unlike
round 1 (+6pp pooled). Round 3 (round2 -> round3, same recipe): 329 failures of the 78%-at-T=1 policy, 253 states, 98 verified;
unseen **103/134 = 76.9%** (round 2: 75.4%; 8/6, p=0.79) -- the chain plateaus ~13pp below the teacher
(89.6%; 2/19 vs teacher, p=0.0002). Verified-fix rate per round: 41% -> 54% -> 39%; the remaining failures are the
less fixable ones. Cumulative: 910 teacher calls, 3 rounds, 41.8% -> 76.9% on unseen. Extended: round 3 seen 75.0%, held-out
73.0% -> pooled 74.0% vs round 2's 76.5% (r3-only 56 / r2-only 75, p=0.12): the oracle chain saturates at
round 2 -- the third round's 253 states yielded only 98 verified fixes and the fallback targets on the rest
cost a little.

### Attr-OPD round 1 on ALFWorld -- LOCATION CARRIES THE SIGNAL (complete 2026-09-20 19:45)

Same student (3B, 100-trajectory SFT, 41.8% on the 134 unseen games), same teacher (3B full-data SFT,
89.6%), same 319 failures, same budget (one teacher call per failure, top-2 alternatives verified by a
20-step student continuation, KL at the chosen step); only WHERE the step is differs:

| arm | supervised step | unseen success | vs student (b/a, p) | verified fix rate | teacher calls |
|---|---|---|---|---|---|
| student (no OPD) | -- | 41.8% (56/134) | -- | -- | 0 |
| **oracle** | first harmful action (progress oracle) | **58.2%** | 33/11, p=0.0013 | 41% | 319 |
| **attr** | Model A's step (7B agent-text-only) | **47.8%** | 25/17, p=0.28 | 36% | 319 |
| random | uniformly random step | 34.3% | 17/27, p=0.17 | 10% | 319 |
| last | final step | 32.8% | 17/29, p=0.10 | 0% | 319 |
| uniform | every step, 20 failures (equal teacher calls) | **8.2%** (11/134) | 1/46, p=2e-13 | 4% | 1,000 |

Pairwise: attr beats random 28/10 (p=0.005) and last 31/11 (p=0.003); oracle beats random 37/5 (p=4e-7)
and attr 24/10 (p=0.024). Uniform OPD at the SAME teacher budget collapses the policy (41.8% -> 8.2%):
1,000 calls buy 20 trajectories x 50 states, 96% of them unverifiable (no one-action fix exists there), so
the student is fitted to unverified teacher guesses at neutral states of a handful of games. Two readings,
both favourable to localisation: the budget is spent where nothing can be fixed, and 20 trajectories is
too few to generalise -- spreading the same budget over 319 failures at one informative step each avoids
both. Reading: with the teacher and budget fixed, distilling at a random or final
step HURTS (the teacher's unverified target at a state that cannot be fixed pulls the policy off), at
Model A's step it helps, and at the oracle's first harmful action it helps a lot. The verified-fix rate
(share of attributed states where a one-action change lets the student finish) orders the arms
identically (41 / 36 / 10 / 0%), so it is a usable proxy for attribution quality without training. This is
the positive counterpart of the debate result: on localisable failures the location is the signal; Model A
captures roughly a third of the oracle's gain, in line with its 72%-within-2-steps localisation. Next:
attr_w1 (Model A's step +-1, 2.4 calls/failure -- does a small window recover the oracle's gain?), round 2
(recollect with the round-1 adapter and repeat) for oracle and attr.

Why unverified fallbacks hurt -- a concrete state (attr arm, step 5 of "find two creditcard and put them in
armchair"): after placing creditcard 1 the student stands at sofa 2 (keychain, pillow) and takes the keychain
(the first harmful action; the second card is elsewhere). The teacher, hint included, puts p=1.000 on the
SAME action -- the full-data 3B makes the identical mistake here. With top-2 verification the alternatives
tried are near-zero-probability neutral moves; when none succeeds the fallback target is the teacher's
distribution with the student's action zeroed out, i.e. an essentially arbitrary residual. Training toward
it is what dragged random/last/uniform below the untrained student. Rule for later variants: when nothing
verifies, skip (no gradient) rather than distil the residual; and teacher-agrees-with-student states are
exactly where the environment check, not the teacher, carries the information.

attr_w1 (Model A's step +-1, 769 states, 22:17): **16.4%** (22/134) -- far below attr (47.8%) and the student.
Across the six round-1 arms the count of UNVERIFIED targets predicts the outcome almost perfectly
(corr -0.91): oracle 189 -> 58.2%, attr 203 -> 47.8%, random 288 -> 34.3%, last 319 -> 32.8%, attr_w1 597
-> 16.4%, uniform 960 -> 8.2%. Widening the window added 2.4x more unverifiable states than verified ones.
So the fallback target is the poison, not a side effect: `attr_nofb` (verified targets only) is running.

Caveat found 21:55 -- the round-1 adapters are better GREEDY but worse when SAMPLED. Recollecting T=1
rollouts on the same train games for round 2: attr adapter 43.4% vs the untrained student 61.4% on the same
249 games (adapter-only wins 26 / student-only 71, p=5e-6; mean episode length 27 -> 36 steps), oracle
adapter 57.0% vs 59.1% on 1,281 games (n.s.). So the KL updates sharpened the argmax at the supervised
states but flattened the sampled policy elsewhere (longer, wandering rollouts at T=1). The greedy unseen
gains are the standard ALFWorld metric and stand, but the mechanism needs (a) the extended greedy evals to
confirm they are not a 134-game artefact and (b) a fix in later rounds: no unverified fallback (those
targets are near-arbitrary residual distributions and the likeliest cause of the flattening), and/or a
KL-to-previous-policy regulariser at the supervised state. attr_nofb / oracle_nofb variants queued next.

Extended evaluation (valid_seen 140 + 500 held-out train games, indices 1500-1999, touched by no SFT/OPD
step; greedy, budget 50; reports/alfworld/ext/): student 57.1% seen / 56.0% held-out. **Oracle round-1
adapter: 63.6% seen (+6.4pp, 27/18, p=0.23), 59.0% held-out (+3.0pp, 70/55, p=0.21); pooled over all 774
games 59.7% vs 53.7% (+6.0pp, 130/84, p=0.002).** So the effect is real but its size is ~+6pp pooled; the
+16.4pp on the 134 unseen games is the upper tail (largest gain on new layouts). **attr round-1 adapter (with fallback): 52.1% on seen (73/140, -5.0pp), 56.2% on held-out (281/500,
+0.2pp)** -- pooled 54.0% vs 53.7%, null: its +6pp on unseen was noise. attr_nofb's extended eval is queued
after random/last/uniform. random / last / uniform extended evals queued; oracle round-2 adapter's extended
eval running (35/43 on seen so far).

Attr-OPD round 1 launch note (11:44 on 354 oracle-labelled failures of the low-data student; teacher =
3B full-data SFT with hindsight hint; verify top-2 alternatives with a 20-step student continuation; KL over
the admissible set at the chosen step; LoRA from the low-data adapter, lr 1e-5). Arms: attr (Model A step,
319 failures), oracle (first harmful action, 319), uniform (every step, 20 failures = 1,000 teacher calls,
equal budget), then random and last. Interim at 12:16: a verified success exists for 41% (attr) / 39%
(oracle) of failures vs 16% of uniform's states -- the attributed step is where a one-action change fixes
the episode far more often than an average step; ~22 s per example (verification dominates).

Round-1 first results (14:05): **attr** arm 64/134 = 47.8% on unseen vs the low-data student's 56/134 =
41.8% (+6.0pp; discordant 25/17, p=0.28); **oracle** arm (first harmful action from the progress oracle) 78/134 = **58.2%** (+16.4pp; discordant
29/7 among the first read, p<0.001); **last** arm (final step) 44/134 = 32.8% (-9.0pp; no alternative can be verified with zero budget left,
so it distils unverified teacher targets and HURTS); random and last were restarted at 14:05 restricted to
exactly the attr arm's 319 failures (the auto-started random had drawn 454 as labels accrued); random and
uniform (equal teacher calls, chunked rerun after a host-RAM OOM at 688/1,000) pending. Verified-fix rates
during distillation rank the arms the same way: oracle 41% > attr 36% > random 12% > last 0%. Cost of the attr arm: 319 teacher calls (one per failure), 14,575 env steps for
verification, 1.9 h on an A40.

Low-data student result (06:35): 3B SFT on 100 trajectories, 1 epoch -> **41.8% greedy on unseen**
(56/134): the OPD student, with ~48pp of headroom to the full-data 3B (89.6%). Its T=1 failures on 1,500
train games are being collected (hold 1436314), then Model A + both oracles run on them; the 7B teacher
SFT (all 1,338 trajectories) is training on hold 1435620; the strong student's collection continues on
hold 1446769.

Pipeline changes taken: (1) game order deterministically shuffled so every slice mixes the six task
types (the sorted order grouped them: the first 442 games were 2 types); (2) budget 30 -> 50 for
everything downstream (the handcoded expert itself fails 69% of games at 30; 50 is the ReAct
standard); (3) warm-start SFT of the student on expert trajectories (PDDL-planner expert at budget
50 on 1,053 held-back train games + handcoded expert wins) before any failure collection or OPD --
its failures are the ones that can carry a decisive step. Running: planner-expert collection (CPU
job 1439539) -> SFT chain on hold 1435620 (SFT 2 epochs -> greedy eval on unseen at budget 50 ->
T=1 failure collection on 1,500 train games at budget 50); Model A (7B full / no_obs) on the 408
zero-shot failures as a diagnostic (hold 1436314, after the 14B teacher eval).

## Attr-GRPO (Method II) on the debate team -- CLOSED 2026-09-20 07:00: hard masks hurt, soft masks neutral, location irrelevant, uniform = base

Read-out on the 238 held-out test tasks (primary metric mean@4 at T=1, paired Wilcoxon on
per-task accuracy vs the same base; greedy with exact McNemar):

| arm | gradient on failed rollouts | mean@4 | vs base | greedy | vs base (b/a, p) | no-answer share (sampled) |
|---|---|---|---|---|---|---|
| base (untrained team) | -- | 0.299 | -- | 0.248 | -- | 0.152 |
| uniform (standard GRPO) | all 6 turns | 0.301 | +0.002, p=0.72 | 0.261 | 18/15, p=0.73 | 0.133 |
| random | 1 random turn | 0.274 | -0.025, p=0.078 | 0.218 | 16/23, p=0.34 | 0.232 |
| last | final-round turns | 0.238 | -0.061, p=0.0004 | 0.206 | 19/29, p=0.19 | 0.271 |
| attr (Model A's turn) | 1 blamed turn | 0.268 | -0.032, p=0.032 | 0.189 | 12/26, p=0.034; vs uniform 10/27, p=0.008 | 0.370 |

Final table, all seven arms (mean@4 at T=1 on the 238 test tasks; paired Wilcoxon vs base):

| arm | failed-rollout gradient | mean@4 | vs base | vs uniform |
|---|---|---|---|---|
| base (untrained team) | -- | 0.299 | -- | -- |
| uniform (standard GRPO) | all 6 turns | 0.301 | +0.2pp, p=0.72 | -- |
| attr | Model A's turn (hard) | 0.268 | -3.2pp, p=0.032 | -3.4pp, p=0.048 |
| random | random turn (hard) | 0.274 | -2.5pp, p=0.078 | -2.7pp, p=0.064 |
| last | final round (hard) | 0.238 | -6.1pp, p=0.0004 | -6.3pp, p=0.0002 |
| attr_committed | Model A's turn, committed-wrong only (hard) | 0.247 | -5.3pp, p=0.0003 | -5.5pp |
| attr_soft | floor 0.2 + 6:1 on Model A's turn, uniform budget | 0.297 | -0.2pp, p=0.96 | -0.4pp |
| random_soft | floor 0.2 + 6:1 on a random turn, uniform budget | 0.280 | -1.9pp, p=0.17 | -2.1pp, p=0.14 |

attr_soft - random_soft = +1.7pp (p=0.19): no significant location effect even with the floor. Verdict for
the debate setting: every hard mask is worse than uniform (length drift -> no-answer teams), soft masks
are at best neutral, and Model A's location is never distinguishable from a random turn -- consistent
with 98% of these failures having no decisive step (see ROOT CAUSE below). Method II is closed here and
moves to ALFWorld, where failures are localisable (section above).

Follow-up arms detail (2026-09-20): **attr_soft** (uniform floor eps 0.2 + 6:1 emphasis on Model A's turn, same
total budget as uniform, committed-wrong failures only) final mean@4 **0.297** (greedy 0.235) -- the floor
removes the damage of the hard mask (0.268 -> 0.297) but there is no gain over uniform (0.301) or base
(0.299): localisation at equal budget is neutral. **attr_committed** (hard mask only on committed-wrong
failures) final mean@4 0.247 (-5.3pp vs base, p=0.0003; greedy 0.197): restricting the hard mask to
"real" wrong answers did not stop the length drift (no-answer share 15% -> 39% at T=1). **random_soft**
greedy 0.231 (sampled pending). Both hard-mask follow-ups drifted long exactly like attr.

(The attr arm's sampled final was first lost to a host-RAM OOM -- 67 GB RSS on e9 vs the 64 GB
hold limit -- and recomputed from the saved step-48 adapter with `attr_grpo.py --eval_only`.)

Pairwise on mean@4 (Wilcoxon over the 238 tasks): attr - uniform -0.034 (p=0.048);
attr - random -0.006 (p=0.74); random - uniform -0.027 (p=0.064); last - uniform -0.063
(p=0.0002); last - random -0.036 (p=0.024). So: WHERE the single turn is does not matter
(attr = random), sparsity itself costs ~3pp, and concentrating on the final round costs 6pp.

Answer-rate vs precision. The attr team answers less often (answer rate per task
-0.217 vs base, p=3e-19; no-answer share 15% -> 37% at T=1) but is NOT measurably more often right when it does -- the pooled precision (0.425 vs 0.353)
is a selection artifact of which tasks get answered: per-task P(correct | answered) vs base +0.016 (p=0.639, n=200) for attr,
-0.007 (p=0.47) for random, -0.006 (p=0.88) for uniform; attr - random
+0.019 (p=0.340). So the whole attr effect is fewer committed answers (length drift past the 512-token cap),
with no change in answer quality where an answer is given.

What this says. (1) Standard GRPO at this budget (48 steps, ~260 informative groups, ~2,000
team rollouts trained) does not move the team on the primary metric (+0.2pp). It shifts the
greedy failure mix -- fewer no-answer teams (48% -> 34%), more committed wrong answers -- which
is why the 60-task greedy subset looked like +12pp mid-run and the full test set shows +1.3pp
(n.s.). (2) Every localised mask is worse than uniform, and worse than the untrained base;
`last` significantly so (-6.1pp mean@4, p=0.0004). (3) The damage is a length/termination
effect, not a reasoning effect: the masked arms' no-answer share rises (15% -> 23-27% at T=1,
48% -> 53-63% greedy) while their precision given an answer does not fall (attr greedy
P(correct|answered) 0.51 vs base 0.48 vs uniform 0.39). Mechanism: a failed rollout's negative
gradient on all six turns is uniform GRPO's only length regulariser; a hard mask drops 5/6 of
it, so the positive gradient from long successful rollouts wins and turns run past the 512-token
cap (attr_committed: 52% -> 74% of turns at the cap over 20 steps). (4) attr vs random: mean@4 0.268 vs 0.274 (paired
-0.006, p=0.74) -- Model A's location is no better than a random turn for this purpose; the
masked arms differ from uniform by sparsity, not by where the gradient lands. Follow-ups running: attr_committed (hard mask
only on committed-wrong failures), attr_soft / random_soft (uniform floor + 6:1 emphasis at the
uniform budget) -- see below.

ROOT CAUSE (2026-09-20 00:45): the failures have no decisive step. Classifying every self-play
debate team by per-agent correctness per round (data/selfplay/debate{3b,}_trajectories.jsonl):

| team | failures | round-0 all three wrong | round-0 one right, two wrong | round-0 majority RIGHT, lost in round 1 |
|---|---|---|---|---|
| 3B (593 teams) | 393 | 348 (88.5%) | 36 (9.2%) | 9 (2.3%) |
| 7B (862 teams) | 568 | 523 (92.1%) | 42 (7.4%) | 3 (0.5%) |

Only the last column is a localisable failure (a correct majority that one agent talked out
of the answer). 98-99% of failures are capability failures: nobody had the answer at any step,
so "the step where it went wrong" does not exist. Model A was trained on Who&When Pro, where
every failure has one annotated decisive step; applied to these traces it must still name a
step, and what it names is position (agent_1 49%, final round) -- i.e. noise, which is exactly
why attr = random everywhere (message feedback, location precision, Attr-GRPO). Debate itself
helps the other way: 10% of teams start majority-wrong and are FIXED by round 1. Consequence:
location-based supervision needs a localisability check first; for diffuse failures the only
useful supervision is capability (teacher/OPD) or the "why" content. DECISIONS #36.

Original launch note (2026-09-18 22:14 CDT):

Researcher's choice ("可以做方案 1 ... ok sounds good"): instead of replaying from the
attributed step, train the team with RL and let the attribution decide WHERE the gradient
lands. Standard RLVR on the self-play debate team (Qwen2.5-3B-Instruct + LoRA r32, three
debaters sharing one adapter, 2 rounds, majority vote): G=8 team rollouts per task at
T=1, reward = final answer correct (0/1), group-normalised advantage, REINFORCE + k3 KL
to the frozen reference (beta 0.04). The one manipulated variable is the per-turn weight
on FAILED rollouts (successful rollouts always use all six turns):

| arm | gradient on failed rollouts lands on | question it answers |
|---|---|---|
| uniform | all 6 turns (= standard GRPO) | baseline |
| attr | the turn Model A blames (agent, round) -- 7B agent-text-only attributor | does localising credit help? |
| random | one uniformly random turn | does WHERE matter, or just sparsity? |
| last | the three final-round turns | is pointing earlier than the answer step worth anything? |

Dynamic sampling (DAPO-style): zero-variance groups carry no gradient and are dropped;
8 tasks are sampled per step, resampled once if fewer than 4 informative groups (so
8-16 tasks/step). Pre-check on 60 train tasks: 43% of groups are informative at G=8.
Train/test split = the OPD task split (355 train / 238 test tasks: math, gpqa, mmlu_pro,
scibench). 48 steps per arm, same seed, same config; greedy eval on a fixed 60-task test
subset every 12 steps; final eval on all 238 test tasks both greedy and mean@4 at T=1,
per-task records saved for paired tests. The base policy's full-test greedy + mean@4 is
computed once (attr arm, `--full_eval_at_start`) as the shared reference.

Engineering notes: (1) the 2-step smoke test (A40) ran clean -- Model A's prediction was
parseable on every failed rollout, KL moved 0 -> 8e-4 after one update; (2) rollouts, not
gradient steps, dominate cost (A100-40: 853 s rollouts vs 99 s training per step), so the
rollout code batches all tasks of a step together, length-sorted; (3) generate() on this
stack materialises the attention scores, so memory is quadratic in sequence length -- both
first timing runs OOMed at 96/192 sequences x ~1,700 tokens; fixed with a per-call budget
n_seq x (prompt+new)^2 <= 1.2e8 (40 GB) / 3e8 (H100); (4) greedy decoding is NOT batch-
invariant here (the same 8 tasks scored 2/8, 4/8, 3/8 under three chunkings), so the
sampled mean@4 on 238 tasks is the primary metric and greedy is secondary.

Base reference (untrained 3B debate team, all 238 test tasks, same code path as the arms):
greedy 0.248; mean@4 at T=1 0.299 (sampling + majority vote beats greedy for this team).
Subset-60 greedy at step 0: 0.317 (A100 chunking) / 0.250 (A40 chunking) -- the batch jitter.

Mid-run diagnostic (step 24 of 48, 60-task greedy subset): uniform 0.317 -> 0.417, random
0.317 -> 0.250, last 0.250 -> 0.250, attr 0.317 -> 0.200. The greedy metric is dominated by
NO-ANSWER teams (no parseable majority `\boxed{}` within 512 tokens -- greedy repetition
loops): base 48% of greedy teams (scibench 67%) vs 15% of T=1 sampled teams. Uniform's gain
is mostly fewer no-answers (30 -> 15 of 60); attr's loss is more of them (29 -> 37). At the
training temperature ~22% of failures are no-answer, and attributing a truncated trace is
meaningless -- a confound for the attr arm, which pushes down one blamed (often round-0)
turn and never the unfinished final turn. Follow-up arm added to attr_grpo.py:
`attr_committed` (attr mask only when the team committed a wrong answer; no-answer failures
get the uniform mask); step records now log fail_noanswer / fail_wrong / mean_out_tokens /
share_maxlen. The final mean@4 read-out is unaffected by greedy loops.

Truncation is pervasive, not just in no-answer teams: the first attr_committed step logs
share_maxlen 0.515 -- 52% of all debater turns hit the 512-token cap (mean 430 tokens), and
18% of failures in informative groups had no answer (6 of 34). So Model A attributes on
traces whose turns are half cut off. A clean re-run needs a 1,024-token cap (2x rollout
cost -> ~32 steps per 24 h hold); decide after the 48-step read-out.

Follow-up arms launched 2026-09-19 10:15 (holds 1388664 attr_soft, 1388665 random_soft; the
attr_committed arm has been running since 05:22 on L40S hold 1371660). Why: the masked arms
drift long -- attr_committed's turns at the 512 cap went 52% -> 64% -> 74% over 20 steps and
no-answer failures rose from 18% to 44% of failures -- because failed rollouts lose 5/6 of
the negative gradient that, in uniform GRPO, keeps outputs inside the budget. `attr_soft` /
`random_soft`: for committed-wrong rollouts every turn keeps a floor weight (eps 0.2) and the
blamed (or a random) turn gets +1, rescaled so the token-weighted budget equals the uniform
mask's (emphasis 6:1, same total push); no-answer failures and successes stay uniform. This
tests the mechanism directly: if attr_soft >= uniform > attr, localisation needs a uniform
floor; random_soft says whether the emphasised turn's location matters.

What the mask looks like in practice (early diagnostics): in the H100 timing step, Model A
blamed a final-round turn in 20 of 26 failed 3B rollouts (agents 8/7/5) and a first-round
turn in 6, so the attr mask is mostly a one-turn subset of the last mask -- the attr-vs-last
contrast asks whether that single turn beats the whole final round, attr-vs-random whether
the choice of turn matters at all. For reference, on the 150 attributed 7B debate failures
(selfplay_chain_debate_round_150/attributions.jsonl) Model A split rounds 73/77 and agents
73/45/32 (a first-agent lean in a symmetric team), modes C.3 74 / R.2 50 / R.4 24.

Where: uniform A100-40 (1333436, agb02), random A100-40 (1333437, agb04), last A40
(1348013, agd06), attr H100 (1333435, e9); all ee5239. Outputs reports/attr_grpo_3b/<arm>/
(-> /scratch.global). Read-out: `python scripts/compare_attr_grpo.py`.

## External-reader study: the missing baselines (researcher, 2026-09-10; run 2026-09-12) -- DONE, REVISES THE "WHY >> WHERE" CLAIM

Same 2,328 traces, same Qwen-7B reader, same output constraint; three new conditions
(`task_only` = task text only; `no_feedback` = task + trajectory, not told it failed;
`incorrect_only` = task + trajectory + "your team's answer was incorrect"). Current grader.

| condition | recovery | mathchat | magentic-one | debate | dylan |
|---|---|---|---|---|---|
| task_only (fresh solve, answer-only) | 14.3 | 17.6 | 11.2 | 12.8 | 15.1 |
| no_feedback (trajectory, no label) | 28.5 | 27.2 | 29.3 | 22.0 | 50.7 |
| **incorrect_only (trajectory + "wrong")** | **42.4** | 48.9 | 41.4 | 26.7 | 53.9 |
| precise_only (+ exact step, no reason) | 40.5 | 43.0 | 41.7 | 28.1 | 54.6 |
| vague_reason | 41.8 | 46.5 | 42.2 | 28.1 | 51.3 |
| reason_only | 43.6 | 49.7 | 42.5 | 30.1 | 52.0 |
| precise_reason | 43.5 | 49.2 | 41.8 | 31.5 | 53.3 |

Paired McNemar (n=2,328): no_feedback vs task_only 547/217 (p=1e-33); incorrect_only
vs no_feedback 351/27 (**p=5e-73**); reason_only vs incorrect_only 78/50 (**p=0.017**,
+1.2pp); precise_reason vs incorrect_only 84/58 (p=0.036, +1.1pp); vague_reason vs
incorrect_only 53/67 (n.s.); **precise_only vs incorrect_only 34/79 (p=3e-5, -1.9pp)**;
reason_only vs precise_only 111/38 (p=2e-9) as before.

**What changes:** the earlier headline "diagnostic content drives external
correction; location alone barely helps" was measured against the wrong zero. With
the baselines in place the decomposition is: seeing the trajectory +14pp, being told
it is WRONG +14pp, adding the error TYPE +1pp (small, significant), adding the exact
STEP without a reason **-2pp** (a bare pointer misdirects the reader). The "why >>
where" gap (3.1pp) is therefore two-thirds "where hurts" and one-third "why helps".
**What this unifies:** for external readers as for the erring teams, the active
ingredient is the outcome signal ("this is wrong" -- and, for teams, how
authoritatively it is said); Model A's diagnosis is worth ~1pp to a reader and ~0 to
the team. Location-only feedback is the one condition that is worse than saying
nothing beyond "wrong" -- consistent with wrong_location = precise_only in the
self-play controls: the pointer is not being used as information, only as a
distraction. docs/feedback_granularity_study.md and the report Sec. 5 carry the
corrected framing; DECISIONS #31.

## TODO 7: OPD for MODEL A -- the third training paradigm (researcher, 2026-09-10) -- DONE, NULL

GKD-style on-policy distillation. Student = best 3B attributor (agent-text-only
SFT, merged) + fresh LoRA; teacher = 7B attributor (same format, frozen; ALFWorld
step 0.913). Student samples its 3-line answer on 5,018 in-distribution train
prompts (T=1), loss = per-token full-vocab KL(teacher||student) at the sampled
positions, no gold labels. Half an epoch (2,509 samples, 314 optimizer steps,
~2h/arm). Two arms: uniform token weights; mode-weighted (x3 on the Error Mode
line). Evaluated on the variant's own prompts, paired McNemar vs the student init.

| 3B attributor | val agent/step/mode/all-3 | ALFWorld step/mode/all-3 |
|---|---|---|
| init: SFT (agent-text-only, keep_ids) | 0.976 / 0.956 / 0.831 / 0.820 | 0.627 / 0.267 / 0.232 |
| + OPD uniform | 0.975 / 0.958 / 0.821 / 0.809 (all-3 9/15, p=0.31) | 0.643 / 0.257 / 0.225 (step 11/6 p=0.33; all-3 3/5 p=0.73) |
| + OPD mode-weighted | 0.976 / 0.956 / 0.820 / 0.804 (all-3 5/14, p=0.064) | 0.621 / 0.267 / 0.222 (all n.s.) |
| for reference: GRPO hard-mined ckpt100 | 0.978 / 0.956 / 0.811 / 0.799 | 0.550 / 0.270 / 0.215 |
| teacher: 7B SFT | 0.985 / 0.961 / 0.852 / 0.838 | 0.913 / 0.399 / 0.389 |

Reading: OPD from the 7B moved the 3B nowhere on either split -- in-distribution
within noise (mode-weighted slightly negative, p=0.06), and on ALFWorld exactly
at its init while the teacher sits 29pp higher on step. Diagnosis is in the
training log: mean per-token KL between student and teacher on the training
prompts is 0.007-0.015 nats (they already agree there; both fit the training
set), so on-policy sampling in-distribution exposes almost no disagreement to
distill, and whatever makes the 7B better OOD is not expressed on these prompts.
Attributor training verdict across three paradigms: SFT learns it; GRPO with a
sparse 0/1 reward does not improve it and can erode mode; OPD from a stronger
same-format teacher, on in-distribution prompts, is a no-op. The only thing that
moved OOD attribution was input format (drop observations) and base-model scale.
Untested and the obvious next lever if OPD is revisited: distill on prompts where
the two DISAGREE -- i.e. OOD-style or perturbed prompts (unlabeled ALFWorld would
work but breaks the held-out protocol; synthetic embodied traces would not).

## TODO 6: Attribution-guided OPD (Research Plan Sec. 10) -- 7B self-teacher DONE (null/negative); 14B-teacher and 3B running

Setup: student = Qwen2.5-7B debate team (3 debaters sharing weights). 181
failures on the 355 TRAIN tasks; Model A (3B SFT) attributes each (agent i*,
round r*, mode). Privileged teacher = the SAME 7B given the exact turn prompt +
hindsight (team wrong; this turn decisive; error type; original response).
Student LoRA distilled toward the teacher's turn at the same prefix WITHOUT
hindsight (token-level KL, teacher = adapter-off pass). Eval: greedy team
accuracy on the 238 held-out TEST tasks.

**Teacher quality (the plan's hidden assumption):** the 7B with hindsight
answers correctly on only 128/1,448 generations (attr turn 6.1%, random turn
8.3%, uniform 9.4%); it changes its answer ~55% of the time, mostly to another
wrong one. A 14B teacher on the same prompts: 197/1,448 (13.6%; attr 14.4%,
random 15.5%). Knowing where/what the error is does not let the same model
fix it -- the OPD-side twin of the self-correction null.

| arm (7B student, 7B self-teacher) | test acc (n=238) | vs base (arm-only/base-only, p) |
|---|---|---|
| base (no training) | 112/238 = **47.1%** | -- |
| Attr-OPD (w=1 at attributed turn) | 108 = 45.4% | 14/18, p=0.60 |
| random-turn OPD (mislocated) | 101 = 42.4% | 11/22, p=0.08 |
| uniform OPD (all turns, 1/T) | 98 = 41.2% | 17/31, p=0.06 |

attr vs random 20/13 (p=0.30); attr vs uniform 27/17 (p=0.17).

| arm (3B student, 3B self-teacher, 7.8% correct) | test acc (n=238) | vs base |
|---|---|---|
| base | 69/238 = **29.0%** | -- |
| Attr-OPD | 73 = 30.7% | 22/18, p=0.64 |
| random-turn OPD | 81 = 34.0% | 30/18, p=0.11 |
| uniform OPD | 70 = 29.4% | 23/22, p=1 |

3B: nothing significant; the best cell is the MISLOCATED one (random 34.0%, attr
vs random 24/32 p=0.35), so no evidence that attribution location matters
for a 3B student either. Reading: with a
teacher that is right <10% of the time, distillation can only hurt, and it
hurts LEAST where the attribution puts it (attr > random > uniform, the
plan's predicted ordering, but as damage limitation rather than gain).
| arm (7B student, **14B teacher**, CE, 13.6% correct) | test acc (n=238) | vs base 47.1% |
|---|---|---|
| Attr-OPD | 94 = 39.5% | 12/30, **p=0.008** (worse) |
| random-turn OPD | 101 = 42.4% | 15/26, p=0.12 |
| uniform OPD | 97 = 40.8% | 14/29, p=0.03 (worse) |
| Attr-OPD, verified-only (26 rows) | 110 = 46.2% | 19/21, p=0.88 |
| uniform OPD, verified-only (143 rows) | 112 = 47.1% | 19/19, p=1 |

14B teacher, hard CE imitation: every unverified arm HURTS (attr the most, -7.6pp,
p=0.008 -- concentrating a still-mostly-wrong teacher on one turn concentrates the
damage); verifying the teacher's answer removes the harm but, with 26/143 usable
rows, buys nothing. Together with the self-teacher runs this closes the by-product:
Attr-OPD's sign is set by teacher correctness at the attributed step (8.8% / 13.6%
-> negative; filtered -> zero; a reliable teacher is untested here). First hold for
this run (84810) hit its 24h TIMEOUT mid-training; rerun on 172957.

## TODO 5: attributor scale test, Qwen2.5-7B (researcher, 17:00) -- DONE (2x2 complete, 2026-09-10 00:20)

Same LoRA recipe (r=16, 3 epochs, 8,192 cap), IDENTICAL 4,767 training rows in
all four runs, each evaluated on its own prompt format, paired McNemar.

| model (train input) | val agent/step/mode/all-3 | ALFWorld step/mode/all-3 |
|---|---|---|
| 3B full-trace | 0.976 / 0.956 / 0.828 / 0.816 | 0.534 / 0.260 / 0.203 |
| 3B no_observations | 0.976 / 0.956 / 0.831 / 0.820 | 0.627 / 0.267 / 0.232 |
| 7B full-trace | 0.975 / 0.948 / 0.831 / 0.818 | 0.865 / 0.341 / 0.328 |
| **7B no_observations** | 0.985 / 0.961 / 0.852 / **0.838** | **0.913 / 0.399 / 0.389** |

Paired tests (ALFWorld, n=311):
- **Scale, same format** (7B full vs 3B full): step 106/3 (p=7e-28), mode 37/12
  (p=5e-4), all-3 46/7 (p=4e-8). +12.5pp all-3.
- **Format, at 7B** (7B no_obs vs 7B full): step 21/6 (p=0.006), mode 23/5
  (p=9e-4), all-3 25/6 (**p=9e-4**). +6.1pp all-3 -- LARGER than at 3B (+2.9pp,
  p=0.15), and now it moves mode too (at 3B mode was flat).
- The two effects are additive: 20.3 -> 23.2 (format) -> 32.8 (scale) -> 38.9 (both).
Paired tests (val, n=593): scale does nothing (7B full vs 3B full: all-3 32/33,
p=1); format at 7B +2.0pp (27/15, p=0.088), borderline.

Reading: in-distribution, the attribution task is saturated at 3B (~82%) and
neither scale nor format matters. Out of distribution, both matter and stack:
a 7B trained on agent text only localizes the decisive step in 91% of
never-seen embodied traces (3B full-trace: 53%) and gets mode to 40% -- the
first time any run leaves the 0.26 guessing floor on ALFWorld. Removing
environment observations is not a small-model crutch; it helps MORE when the
model is stronger. Canonical Model A going forward: **7B, agent-text-only
input** (reports/attributor_7b_no_observations/final).

## TODO 3B: LLM-judge baseline (untuned Qwen2.5-7B, identical prompt) -- DONE

val n=593, same prompt, official scorer, paired McNemar vs SFT-3B:

| model      | agent | step  | mode  | all-3 |
|------------|-------|-------|-------|-------|
| SFT-3B (Model A) | 0.976 | 0.956 | 0.828 | 0.816 |
| judge-7B (untuned) | 0.906 | 0.626 | 0.160 | 0.096 |
| untuned-3B | 0.776 | 0.395 | 0.169 | 0.049 |

ALFWorld (added 2026-09-16): untuned 7B judge step 0.421 / mode 0.199 / all-3 0.125 (untuned 3B: 0.084 / 0.186 / 0.029; SFT-3B: 0.534 / 0.260 / 0.203).
Scale alone buys step localization (0.395 -> 0.626) but NOTHING on mode
(0.169 -> 0.160; 17-way taxonomy is not zero-shot-learnable from the
prompt). SFT-3B beats judge-7B on all-3 438 vs 11 discordant traces
(p=4.7e-114). Per framework, judge-7B only approaches SFT on mathchat step
(0.84 vs 1.00). 32B judge deferred (does not fit an L40S in bf16 next to
the running chain; low expected information given 3B->7B mode is flat).

## GRPO (RL) training of the attributor: DONE -- verdict (2026-09-08/09)

**Three arms, one consistent answer: at this scale (200-250 GRPO steps,
~800-1,000 prompts, LoRA r=16, beta=0.04), a sparse 0/1 verifiable reward
does not teach the 17-way error-mode classification and can erode it;
SFT's dense supervision does.** (1) SFT-start on all prompts: no gradient
-- 85-95% of groups had identical rewards because SFT had memorized them.
(2) RL from the base model: learned in-distribution step localization
(0.40 -> 0.58) and nothing else; on ALFWorld transfer statistically
identical to the untuned base. (3) SFT-start on 714 hard-mined prompts:
plenty of signal, but held-out mode fell 0.828 -> 0.811 (0-2 fixed,
11-12 broken, p <= 0.013) via a mining selection effect (the hard set
over-represents R.3-labeled mathchat, so RL learned "mathchat -> R.3");
stopped early at ~step 115. Agent and step were never hurt. Untested fixes
if revisited: mode-balanced mining, mode-weighted reward, larger scale.
Details per arm below; DECISIONS.md 12-13, 17, 19-20.

### Arms (chronological detail)

Researcher asked for RL in addition to SFT. Full design rationale in
DECISIONS.md entry 12. Verifiable reward = official `whowhen_eval.score`
on each sampled completion (3 separate 0/1 rewards: agent/step/mode, so
per-axis curves are logged), shared parser with eval
(`src/error_attr/attribution/parse.py`). trl 1.12 `GRPOTrainer`, LoRA
r=16, 8 generations/prompt, 4 prompts/update, 64-token completions,
prompts filtered to <=4096 tokens (3,639).

Three arms:
1. **SFT-start, all prompts -- STOPPED after 2 log points.** The SFT model
   has memorized these training prompts (agent/step reward 1.0, mode 0.91
   on train), so `frac_reward_zero_std` was 0.85 -> 0.95: 85-95% of
   groups had identical rewards -> zero advantage -> zero gradient
   (grad_norm fell to 2.7e-05). Not a bug -- a real limitation of RL on
   already-fit prompts. Log: `reports/attributor_grpo_sftstart_baseline.log`.
2. **Base-model start (no SFT) -- RUNNING (v2)** on the L40S (job 22443,
   tmux `grpo_base`, log `reports/attributor_grpo_basestart_train_v2.log`,
   output `reports/attributor_qwen2.5-3b-grpo-basestart`). Tests RL vs SFT
   as alternative paradigms; v1's first log showed zero_std=0.5 (half the
   groups informative -- real signal, unlike arm 1) but mode reward exactly
   0.0 and 99% clipping at 64 tokens. Cause was parser strictness (base
   model writes `A.3: Premature Termination`; scorer wanted bare `A.3`),
   fixed in the shared parser to mirror the official harness's lenient
   code extraction -- SFT eval numbers verified unchanged -- plus a
   160-token completion cap. See DECISIONS.md entry 13. 200 steps first.
   **Mid-run read (checkpoint-50 = 200 prompts, first 150 val rows =
   macnet/metagpt/smolagents):** agent 0.887, step 0.573, mode 0.160,
   all-3 0.073 -- vs SFT 0.987/0.980/0.833/0.833 on the same rows, and vs
   the UN-tuned base on the same rows: 0.873/0.440/0.153/0.053. So 50 RL
   steps moved step localization +13pp (0.44 -> 0.57), agent/mode ~flat,
   all-3 +2pp. Real but small; mode (the hard axis) untouched so far; SFT
   far ahead in-distribution. The arm's real test remains the ALFWorld
   transfer set at step 200. Paired vs untuned base on those rows: step
   21-vs-1 (p=1.1e-5), agent/mode n.s.
   **Training trend, steps 90-125:** step reward 0.54-0.72 (no collapse),
   agent 0.82-1.00 (one 0.70 point), lengths/clipping stable, KL spiked
   to 0.042 at step 110 then 0.01-0.03 (beta=0.04 holding). Mode drifted
   DOWN: 0.14-0.23 through step 105 -> 0.06-0.09 at steps 115-125. Not
   intervening (cosine LR already 3.5e-6, 75 steps left); instead
   evaluating checkpoint-100 (the reward peak) on the same 150 val rows as
   checkpoint-50, as a 50->100 held-out check and a fallback if the final
   checkpoint is worse. **Result: flat.** ckpt-100 agent 0.887 / step
   0.600 / mode 0.140 / all-3 0.067 vs ckpt-50 0.887 / 0.573 / 0.160 /
   0.073. The 50->100 training-reward rise did NOT become held-out
   accuracy; RL-from-base plateaus far below SFT in-distribution and does
   not learn mode at this scale. Only the step-200 ALFWorld transfer eval
   remains open for this arm.
   **Training FINISHED:** 200 steps in 18,240 s (5.07 h), 0.22 epochs
   (800 of 3,639 prompts seen), final adapter at
   `reports/attributor_qwen2.5-3b-grpo-basestart/final`. Training reward
   at the end: mode had recovered to ~0.26 (step 150) after the 0.06-0.09
   dip. Full eval (val 593 + ALFWorld 311, 160 tokens) running on the
   L40S (tmux `eval_grpo_base`) -> `reports/eval_{val,transfer_alfworld}_grpo_basestart_final.*`.
   **FINAL RESULT (arm 2, 200 steps):**
   | | agent | step | mode | all-3 |
   |---|---|---|---|---|
   | val: untuned | 0.776 | 0.395 | 0.169 | 0.049 |
   | val: GRPO-base | 0.789 | 0.575 | 0.170 | 0.083 |
   | val: SFT | 0.976 | 0.956 | 0.828 | 0.816 |
   | transfer: untuned | 1.000 | 0.084 | 0.186 | 0.029 |
   | transfer: GRPO-base | 1.000 | 0.064 | 0.206 | 0.019 |
   | transfer: SFT | 1.000 | 0.534 | 0.260 | 0.203 |
   Paired McNemar: on val, GRPO-base > untuned on step (131 vs 24,
   p=5e-19) and all-3 (31 vs 11, p=0.003), NOT mode (20 vs 19, p=1);
   GRPO-base < SFT on every axis (p <= 1e-27). On ALFWorld transfer,
   GRPO-base is indistinguishable from untuned on every axis (step p=0.24,
   mode p=0.45, all-3 p=0.25); SFT beats both decisively. **Conclusion:
   RL-from-base learns in-distribution step localization only (smolagents
   0.51->0.63, mathchat 0.15->0.63, magentic-one 0.12->0.42), never mode,
   and none of it transfers. The "RL generalizes better than SFT"
   hypothesis is not supported at this scale (200 steps, 0.22 epochs).**
   Full tables: `scripts/compare_attributors.py --split {val,transfer_alfworld} ...`.
3. **SFT-start on hard-mined prompts -- MINING DONE; first GRPO run
   PREEMPTED at step 40, RELAUNCHING on a non-preemptible A100.** Mining
   (`scripts/mine_grpo_hard_prompts.py`, 8 samples/prompt from the SFT
   model at t=1.0) kept **714/3639 (19.6%)** prompts with reward variance
   -> `data/attributor_grpo/train_hard.jsonl` (+ per-prompt stats in
   `mining_stats.jsonl`). First GRPO run (A40, `preempt-gpu` job 26096)
   confirmed the mining worked -- frac_reward_zero_std 0.1-0.3 vs arm 1's
   0.85-0.95, mode reward 0.56 -> 0.64 by step 25, grad_norm ~0.3-0.5 --
   then was preempted at step 40 before its first checkpoint (log kept:
   `reports/attributor_grpo_hard_train_preempted_at_step40.log`; see
   DECISIONS.md 17). **Relaunched** on a fresh non-preemptible `a100-4`
   allocation (job 67517, node aga29, 24h): same args plus
   `--save_steps 25`; tmux `grpo_hard`, log
   `reports/attributor_grpo_hard_train.log`, output
   `reports/attributor_qwen2.5-3b-grpo-hard`. ~2.2 min/step on the A40;
   expect similar or faster on the A100 -> ~9h.
   **Held-out checkpoint reads (full val, 593):**
   - ckpt-25 (100 hard prompts of RL, KL~0.0004): agent 0.978 / step 0.956
     / mode 0.818 / all-3 0.808 vs SFT 0.976/0.956/0.828/0.816. Mode: 0
     fixed, 6 broken (p=0.031) -- ~1pp the wrong way, spread across
     mathchat/magentic-one/smolagents. Early perturbation, but the null
     ("RL on hard prompts does not improve mode") is live.
   - ckpt-50 (KL 0.012): agent 0.976 / step 0.954 / mode **0.809** / all-3
     0.799. Mode: 0 fixed, 11 broken (p=0.001); all-3: 0 fixed, 10 broken
     (p=0.002). Monotone: 0.828 -> 0.818 -> 0.809. Per-framework: mathchat
     0.74 -> 0.68, macnet 0.73 -> 0.65, dylan down, nothing improved.
     **Emerging conclusion across arms 2+3: the 17-way mode classification
     is learned by SFT's dense supervision, not by GRPO's sparse 0/1 reward
     at these scales; RL on the SFT policy only degrades it (never fixes a
     held-out mode error, only breaks correct ones).** Mechanism (the 11
     broken rows): not distribution collapse (overall predicted-mode counts
     barely move) but drift WITHIN the reasoning family -- 5/11 are mathchat
     rows where gold R.1 (hallucination) / R.4 (task misunderstanding)
     became R.3 (numerical error); the rest drift R.1/R.4/PL.1/C.1 ->
     R.2/R.3/A.1/C.2. RL pushes toward "locally plausible" error types and
     away from ones requiring a grounding/framing judgment -- consistent
     with learning the mathchat prior (mathchat = 49% of the hard set).
     Decision point:
     checkpoint-100 -- if mode is still falling, stop the run (saves ~6h of
     A100) and run the ALFWorld transfer eval on ckpt-100 for the record.
   - ckpt-100: agent 0.978 / step 0.956 / mode **0.811** / all-3 0.799.
     Mode 2 fixed / 12 broken vs SFT (p=0.013); all-3 1/11 (p=0.006).
     Plateaued at the degraded level (0.809 -> 0.811), no recovery.
     **ARM 3 STOPPED at step ~115 (early stop; ~6h of A100 saved).**
     Log: `reports/attributor_grpo_hard_train.log`; checkpoints 50/75/100
     kept. ckpt-100 ALFWorld transfer eval running for the record.
   - ckpt-25 on ALFWorld transfer (311): agent 1.000 / step 0.534 / mode
     0.267 / all-3 0.203 -- identical to SFT (step 2-vs-2, mode 2-vs-0
     p=0.5, all-3 0-vs-0). Slight RL on hard prompts leaves transfer
     untouched; a useful stability control.
   **Composition of the hard set (mining_stats.jsonl):** mathchat 352
   (47% yield), magentic-one 160 (30%), smolagents 139 (6% -- 61% of the
   train data but nearly all memorized), macnet 36, metagpt 20 (95%
   yield), debate 7. The SFT model's sampled mode accuracy on its own
   train prompts is the weak spot (metagpt 0.43, macnet 0.72, mathchat
   0.76, magentic-one 0.85, smolagents 0.98); kept groups average 0.83
   reward -> the RL signal here is mostly the mode axis on mathchat/
   magentic-one/macnet/metagpt. Eval question for arm 3: does mode improve
   on those frameworks without regressing smolagents (val) -- and does any
   of it transfer to ALFWorld.
   Yield started at ~38% on the first (harder) prompts and settled at ~9%
   cumulative -> ~330 hard prompts expected (vs 5-15% informative groups
   GRPO was seeing online). Plan for the hard-variant run: ~3 epochs over
   the small set (`--max_steps 250`), `--temperature 1.2`.
   GRPO on the mined set launches on the A40 when mining finishes:
   `python scripts/train_attributor_grpo.py --data data/attributor_grpo/train_hard.jsonl --output_dir reports/attributor_qwen2.5-3b-grpo-hard`.

Evaluation plan for arms 2 and 3: `scripts/eval_attributor.py --model
<output>/final --split val --max_new_tokens 160` and `--split
transfer_alfworld`, compared to SFT (val all-3 81.6%, transfer all-3
20.3%). The question that matters most is whether RL narrows the transfer
gap. **Reference point for arm 2 -- DONE:** the UN-tuned Qwen2.5-3B-Instruct evaluated the same way (lenient parser, 160 tokens): val agent 0.776 / step 0.395 / mode 0.169 / all-3 0.049; ALFWorld transfer agent 1.000 / step 0.084 / mode 0.186 / all-3 0.029 (`reports/eval_{val,transfer_alfworld}_base_untuned.*`). On transfer, SFT's mode accuracy (0.260) is barely above the untuned base (0.186): SFT's error-type knowledge does not transfer to ALFWorld; its transfer gain is almost entirely step localization (0.084 -> 0.534). Paired test of GRPO-from-base checkpoint-50 vs the untuned base on the same 150 val rows: step 21 vs 1 (McNemar p=1.1e-5) -- a real localization gain from 50 RL steps; agent (2 vs 0) and mode (3 vs 2) not significant. Per-framework (val), the untuned base already gets `agent` for single-agent frameworks (trivial fixed label) but is near zero on `mode` everywhere (0.11-0.40); SFT's gains are step + mode.

Event-driven waits (background, fire once): mining done -> launch arm 3;
arm-2 checkpoint-50 -> mid-run eval on ~100 val prompts; arm-2 finished ->
full eval; base-untuned eval finished -> record. Reward-trend monitors are
1h-capped and noisy at 4 prompts/step (see DECISIONS.md 13) -- judge trends
over >= 5 logs or via checkpoint evals, not single logs.

## Self-play track (Qwen-generated failures): BUILT, NOT YET RUN (2026-09-08 evening)

Response to the researcher's objection that the feedback study's failed
trajectories came from other systems, not the tested Qwen model (see the
LIMITATION section in docs/feedback_granularity_study.md). Both fixes --
route A (natural Qwen failures, labeled by Model A) and route B (Who&When-
style error injection into Qwen's own successful runs) -- need Qwen-driven
trajectories in release schema. Built:
- `data/selfplay/task_pool.jsonl`: 862 unique (query, gold answer) tasks
  extracted from the 4 gradable frameworks' rows (math 192, scibench 225,
  simpleqa 210, mmlu_pro 117, gpqa 59, gaia 59) -- same tasks as the
  external-transcript study, so results are directly comparable.
- `src/error_attr/selfplay/debate_runner.py`: N Qwen agents, round 0
  independent, later rounds see peers' previous outputs, majority-vote final
  answer; emits the official `debate` schema. Verified that a hand-built
  debate-schema trajectory renders through `whowhen_eval`'s renderer and
  the attributor prompt builder unchanged. Chosen over mathchat/magentic-one
  because it needs no tool execution -- the only multi-agent protocol
  reproducible with pure Qwen.
- `scripts/run_selfplay_debate.py`: runs the pool with Qwen2.5-7B-Instruct
  as the debaters (the SAME model that later receives feedback), writes
  `data/selfplay/debate_trajectories.jsonl` (+ release-row format) with
  `final_correct` per task. Failures -> route A input; successes -> route B.
- RUNNING on a 3rd GPU (A100-40GB, job 41169, node aga34, tmux
  `selfplay_debate`, log `reports/selfplay_debate.log`): started at
  8.4 s/task on GAIA, slowed to ~18 s/task on GPQA's longer prompts ->
  ~4h for 862 tasks. At 120/862: gaia 6/59 (10%), gpqa 16/59 (27%) --
  i.e. ~43 natural GPQA failures already, plenty for route A. GAIA's
  near-zero is expected for a tool-less debate (no retrieval). Pilot of
  the full chain (40 GPQA failures x 5 conditions) launched on the A40
  (tmux `chain_pilot`, log `reports/selfplay_chain_pilot_gpqa.log`) to
  validate every stage end-to-end while collection continues. **Pilot
  done, pipeline validated** (40/40 attributions parsed; outputs archived
  at reports/selfplay_chain_pilot_gpqa/). n=40 results are unpowered (all
  p >= 0.25): incorrect_only 17.5%, reason_only 12.5%, vague 15%,
  precise_reason 15%, precise_only 20% -- no condition beats the no-info
  control. Anchoring: the corrector changed its own wrong answer only
  30-42% of the time in every condition; correct-given-changed 42-47%.
  Found + fixed a feedback-consistency artifact (DECISIONS.md 15). The
  chain now also reports change / correct-given-changed rates.
- **`--protocol debate_round` added to the chain** (off by default):
  injects the coordinator feedback as an EXTRA debate round -- each agent
  sees its own and peers' last outputs plus the feedback, revises, then
  majority vote. This is the faithful test of the original collaboration-
  dynamics question (vs. one_shot's single-model answer), at ~3x the
  generations. Piloted on 8 GPQA failures (tmux `chain_pilot_dr`, log
  `reports/selfplay_chain_pilot_debate_round.log`) -- **validated end-to-
  end** (8/8 attributions parsed, all 5 conditions ran; outputs archived at
  reports/selfplay_chain_pilot_debate_round/). n=8 rates are meaningless,
  but answer-change rates were 37-75% vs one_shot's 30-42%: a revision
  round may loosen anchoring more than a single corrected answer.
  **Researcher's decision (2026-09-08 night): `debate_round` is the more
  reasonable protocol -- run it on 150 trajectories first.** Plan: when
  collection finishes, `--protocol debate_round --sample 150 --seed 0`
  over the reasoning-benchmark failures (math/gpqa/mmlu_pro/scibench),
  5 conditions incl. the `incorrect_only` control, on the idle L40S,
  `--out_dir reports/selfplay_chain_debate_round_150`. ~2,250 agent
  generations (150 x 5 conditions x 3 agents) -> est. 3-5h. `one_shot` on
  the full pool is deprioritized (cheap to add later if wanted).
  **LAUNCHED 22:40** on the L40S (tmux `chain_dr150`, log
  `reports/selfplay_chain_debate_round_150.log`) -- the four reasoning
  benchmarks were already fully collected (final failure pool: scibench
  190, math 123, gpqa 43, mmlu_pro 43 = 399), so the 150-sample is drawn
  from the complete pool without waiting for SimpleQA (excluded anyway).
  Sample: scibench 68, math 50, mmlu_pro 17, gpqa 15. Model A parsed
  150/150. At correction 100/750: 0 correct in every condition. Jobs run
  shortest-prompt-first, so the first 100 are plausibly scibench (exact-
  numeric grading; the original debate solved only 16%) -- the per-
  benchmark table at the end will tell. corrections.jsonl is 0 bytes
  mid-run on both nodes: Python write buffering (rows land on close), not
  data loss; flushes added for future runs (not the two in flight).
  **0/100 resolved as benign:** reproducing the seeded sample on CPU, the
  20 shortest-prompt traces (= the first ~100 jobs) are scibench 12 / math
  6 / mmlu_pro 2 -- exact-numeric-graded problems. Independent witness:
  the `one_shot` full-pool chain at 920/1805 shows 3-6% recovery in EVERY
  condition (incorrect_only 11/193, reason_only 6/182, vague 6/175,
  precise_reason 10/181, precise_only 12/189) -- no condition beats the
  no-info control. On its own failures this Qwen-7B team barely self-
  corrects, whatever it is told. Per-benchmark tables at the end will show
  whether the MC benchmarks (gpqa/mmlu_pro) differ.
- **Self-play collection COMPLETE (862/862, no errors; ~5.5h on the
  A100).** Debate-team accuracy: gaia 6/59 (10%), gpqa 16/59 (27%), math
  69/192 (36%), mmlu_pro 74/117 (63%), scibench 35/225 (16%),
  simpleqa 8/210 (4%). Reasoning-benchmark failure pool = 399 (scibench
  190, math 123, gpqa 43, mmlu_pro 43) -- scibench-heavy, so the chain's
  summary includes a per-benchmark x condition table with the MC chance-
  level re-guess reference (gpqa 1/3, mmlu_pro 1/9). Files:
  `data/selfplay/debate_trajectories.jsonl` (+ release-row format).
  Successful trajectories (463) are route B's input if pursued.
- **`one_shot` on the FULL pool -- DONE (361 usable failures):** recovery
  3.9-5.5% in EVERY condition; no condition beats the no-info control
  (all p >= 0.5); the external-reader "why >> where" effect is absent (sign
  reversed, n.s.). Answer-change rate 28-32% everywhere; correct-given-
  changed 13-18%; gpqa 12-21% vs 33% chance re-guess. Anchoring, not
  information, is the bottleneck for self-correction under one_shot.
  Route B's gold-vs-predicted question is moot here (label-free control =
  everything). Full table in docs/feedback_granularity_study.md.
- Arm-3 ckpt-100 on ALFWorld transfer: step 0.550 / mode 0.270 / all-3
  0.215 vs SFT 0.534/0.260/0.203 -- nominally above, none significant
  (p=0.23/0.51/0.22). The in-distribution mode damage did not carry to
  transfer. A100#2 (job 67517) released; A100#1 kept idle for follow-ups. Full-run plan: one_shot on all
  of them + debate_round on a seeded random 150-subsample (`--sample 150`),
  separate `--out_dir`s. Mining at 3000/3639 with yield back to 13.5%
  (~480 hard prompts expected). Those failures are information-access
  failures, not attributable reasoning errors, so the self-correction chain
  (`scripts/run_selfplay_feedback_chain.py`) defaults to the self-contained
  reasoning benchmarks (math, gpqa, mmlu_pro, scibench) via `--benchmarks`;
  gaia/simpleqa are reported separately if at all. The chain adds an
  `incorrect_only` control to the 4 conditions. Route choice (A / B / both)
  still with the researcher; collection serves either.

## Compute lesson learned (2026-09-08)

A harness/session restart mid-run killed the two `srun --overlap`
background processes (attributor training, feedback study) even though
their SLURM job allocations kept running -- the processes were children
of the previous session's own shell and died with it. Both had, in fact,
already finished completely by the time this was noticed (checkpoints/
result files were complete) so no work was lost, but future long-running
GPU jobs should be launched inside `tmux` (confirmed available on the
compute nodes) so they survive a harness restart, not just backgrounded
via the Bash tool -- this is exactly the standing lesson already recorded
for the peer-induced-error project, now confirmed to matter here too.

Project started from `Research Plan_ Failure Attribution as Transferable
Supervision for Agentic Post-Training.md`. Work began with the plan's own
recommended order (Sec. 26, Phase 1 Oracle Feasibility on ALFWorld) but the
researcher then asked directly for Sec. 9.C: train the specialized
attributor on existing causally-labeled data (Who&When Pro). Both tracks
are now live -- see the sections below. The Phase-1 ALFWorld oracle
work from 2026-08-28 is unchanged and still described further down. A
third track (feedback-granularity/collaboration-dynamics study) was added
2026-09-07 at the researcher's request.

## Feedback granularity study (multi-agent collaboration dynamics): DATA READY, GENERATION NOT YET RUN

Full design in `docs/feedback_granularity_study.md`. Researcher's own
framing: 4 conditions -- `reason_only`, `vague_reason`, `precise_reason`
(= Sec. 20's T3), `precise_only` (= Sec. 20's T1) -- crossing location
specificity (none/vague/precise) with whether an error-type explanation is
given. Studies whether feedback TYPE changes a multi-agent team's
correction ability and (eventually) collaboration dynamics like
overcorrection/blame attribution.

Built and validated on real data:
- `src/error_attr/attribution/feedback_conditions.py`: builds all 4
  condition strings from a trace's gold (agent, step, mode). Location
  coarsening for `vague_reason` is framework-agnostic (buckets the gold
  step's ordinal position in the renderer's own step_index into quarters,
  not each framework's native round/position math) -- validated on all
  2,328 rows in the gradable subset, zero bad buckets after fixing an
  edge case (quarter math produced `lo > hi` when a trajectory had fewer
  than 4 steps). Mode descriptions were originally the FULL taxonomy.yaml
  entry (multi-sentence, written for an annotator, includes "Distinguished
  from R.2..." disambiguation asides) -- wrong register for natural
  coordinator feedback; fixed to name + first sentence only, re-read
  directly from taxonomy.yaml rather than reverse-parsing the judge-prompt
  block.
- Scoped to the 4 easily-gradable multi-agent frameworks (debate/dylan:
  multiple-choice `\boxed{}`; mathchat: numeric `\boxed{}`; magentic-one:
  `FINAL ANSWER:` short string) = 2,328 traces. macnet/metagpt (code
  generation) deferred -- grading needs executing candidate code against
  benchmark test cases, no harness for that yet.
- Protocol is one-shot corrective continuation (render full failed
  trajectory + feedback message + ask for corrected final answer), NOT a
  full multi-turn re-run of the original framework's live orchestration --
  we only have whowhen_eval's read-only renderers, not the frameworks'
  actual orchestration engines. This tests correction ABILITY under each
  feedback type, not live multi-round collaboration dynamics (that's the
  "Plan B" sketched in conversation, if this first pass shows an effect).
- `scripts/prepare_feedback_study_data.py`: builds
  `data/feedback_study/prompts.jsonl`, 9,312 rows (2,328 traces x 4
  conditions). Ran cleanly, zero skipped.
- `scripts/run_feedback_study.py`: generation + grading. Parser is
  intentionally simple exact-match-after-normalization (not an LLM-judge)
  -- applied identically across all 4 conditions, so parsing noise is
  symmetric and shouldn't bias the cross-condition comparison, though it
  will undercount magentic-one's true recovery rate somewhat (a
  factually-correct-but-verbose answer like "6 crocodiles" scores wrong
  against gold "6"). Sanity-checked against synthetic examples before
  running for real.

**Not yet run:** actual generation. Needs a GPU; currently sharing the one
L40S with attributor training, so requested a second GPU (A40 via
preempt-gpu, since interactive-gpu caps this account at 1 concurrent job)
to run in parallel rather than wait ~3.5h for the attributor to finish.

## Attributor training on Who&When Pro: TRAINING IN PROGRESS

**Dataset resolution (important, cost real time):** the first HF mirror
found (`tmpxv7/who-when-pro`) is 89% broken symlinks (9,845/11,072 JSON
entries point to a path structure -- `mmagent/mas/runs/...` -- never
included in the tarball; only `alfworld` and `video/lvbench` were real
files). The correct, complete, OFFICIAL release is
`Leoxx/whowhen_pro` (linked from the paper's own project page,
https://whowhenpro.github.io/, arXiv:2607.09996) -- flat `data/text.jsonl`
etc., no broken links. Downloaded `data/text.jsonl` (6,257 real traces,
23 framework/benchmark pairs, 2,073 with real non-null multi-agent `agent`
labels) + `data/video.jsonl` + `taxonomy.yaml` to
`data/whowhen_pro_official/`. The broken mirror's extracted copy was
deleted. image.jsonl/image_gui.jsonl (6.7GB/11GB, base64-embedded images)
were not downloaded -- text modality is the priority track.

**Reused the official eval harness rather than reverse-engineering 8
trajectory schemas.** The paper's own repo,
https://github.com/whowhenpro/whowhen_pro (MIT), ships per-framework
renderers + the exact LLM-judge prompt + a per-framework GT-to-coordinate
scorer. Vendored into `third_party/whowhen_pro_repo/`, pip-installed as
`whowhen_eval`. `src/error_attr/attribution/sft_format.py` builds each SFT
example by: rendering the trajectory with the official per-framework
renderer (byte-identical to what their frontier-LLM-judge baseline sees --
matters for a fair RQ1 comparison later), building the exact `all_at_once`
judge prompt as the SFT input, and resolving the gold
(agent, step, mode) by brute-force-scanning the renderer's own
`step_index` against their official `score()` function (reuses their
per-framework round/position/ordinal conventions instead of re-deriving
8 different coordinate math schemes). Validated: **100% of 6,257 text rows
convert cleanly**, zero unattributable, in 1.3s.

`scripts/prepare_attributor_data.py` builds the split: `alfworld`
benchmark held out ENTIRELY (never seen in training) as a natural
transfer-test set for RQ3 -- it sits right next to this project's own
ALFWorld oracle-attribution work (Phase 1 below), so the trained
attributor's accuracy AND downstream learning utility on ALFWorld can both
be checked later. Everything else: 90/10 train/val, stratified per
framework. Result: `data/attributor_sft/{train,val,transfer_alfworld}.jsonl`
= 5,353 / 593 / 311 examples.

`scripts/train_attributor.py`: LoRA SFT, Qwen2.5-3B-Instruct (deliberately
smaller than the 7B policy model -- Sec. 9.C's "amortized attribution"
motivation wants a small specialized model, not another big LLM). Input =
the official judge prompt; target = 3-line `Agent Name / Step Number /
Error Mode` (Reason dropped -- no gold reason text exists to supervise
against, and `score.py` never grades it). Prompt tokens masked to -100 in
labels. max_seq_len=8192 (covers ~90% of examples by real Qwen tokenizer
count; the rest are DROPPED rather than truncated, since truncating risks
silently cutting the Response Format section or the target itself off the
end -- 53/593 val examples dropped under this rule, consistent with the
~90th-percentile cutoff). Dataset + collator logic sanity-checked (label
masking correctly isolates the target region, padding/attention-mask
consistent) before spending GPU time on it.

**Running now** (job 18507969, L40S node agd04, launched
2026-09-07 ~19:35). Two false starts before this, both fixed: (1) this
future transformers version dropped `TrainingArguments(warmup_ratio=...)`
in favor of `warmup_steps` only; (2) torch had been installed as the
CPU-only wheel back when this env was first set up (before GPU access
existed) and never upgraded -- `torch.cuda.is_available()` was silently
False even with a GPU allocated, so the first real launch hit CUDA OOM
warnings that turned out to be a red herring for a deeper "not actually
on GPU-appropriate settings" issue... actually the OOM was real once CUDA
worked (batch_size=2 with 8192-token sequences was too much per-step
memory on a single L40S; fixed by dropping to batch_size=1 /
gradient_accumulation_steps=16, same effective batch size, plus
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`). Steady progress since:
~15s/step, no OOM, ETA ~3.5-4h for all 894 steps (3 epochs).

## Phase 1 (ALFWorld oracle feasibility): AS OF 2026-08-28, UNCHANGED

## Infra: DONE

- Dedicated conda env `error_attr` (python 3.11), separate from
  `peer_error` / other envs on this account.
- ALFWorld 0.4.2 (text-only, no ai2thor/vision extras -- those pulled in
  `visdom`, which failed to build under this Python's setuptools and isn't
  needed for the text-based training/eval this project uses) + textworld +
  torch (cpu wheel for now) + transformers/accelerate + usual scientific
  stack, installed in `error_attr`.
- Game data downloaded to `data/alfworld/` (2.2G; train=3553 games after
  filtering to non-movable/non-sliced trajs, valid_seen/valid_unseen/valid_train
  splits also present).
- `configs/base_config.yaml`: minimal ALFWorld config (task_types 1-6, no
  domain randomization, `dqn`-shaped training block just to satisfy the
  env class's config reads -- not actually training via alfworld's own RL
  code, we only use it for env + expert access).
- **Compute note (2026-08-28):** installs were OOM-killed on the shared
  login node. Work ran on a 24h `interactive` partition CPU hold
  allocation via `srun --jobid=<id> --overlap <cmd>`.
- **Compute note (2026-09-07, new session):** the previous session's SLURM
  jobs had long since expired. This session's shell is now itself running
  directly on a compute node (job 18506193, `interactive` partition,
  16cpu/64G, no srun wrapping needed) -- looks harness-provisioned rather
  than something to manage manually. Submitted a fresh GPU hold
  (`slurm/hold_gpu_l40s.sbatch`, `interactive-gpu` partition, 2x L40S,
  24h) for the attributor training run -- queued as of this writing.

## Environment substrate: VERIFIED

`scripts/smoke_test_alfworld.py` -- loads the env, runs a random-policy
episode end to end (admissible commands, reward, done all behave as
expected). 3553 train games load correctly.

## Oracle attribution mechanism: VALIDATED

`src/error_attr/attribution/oracle_alfworld.py` implements the oracle
critical-step attributor described in the plan's Sec. 14/25: for a failed
trajectory, replay the action prefix of length p in a fresh instance of
the same game, then hand off to a certifying expert and check whether it
can still win within the REMAINING steps of the same total step budget the
trajectory was collected under. The largest recoverable p, +1, is t*.

Key finding from validation (`scripts/test_oracle_attribution.py`):
ALFWorld ships two expert types. `HANDCODED` (a fixed script derived from
the game's original solution) gives **non-monotonic** recoverability
traces when resumed from off-script states -- a false-negative artifact of
the script not replanning properly, not a real property of the game.
`PLANNER` (TextWorld's live PDDL solver, replans fresh from the current
state every call) gives a clean, monotonic trace and a well-defined t*.
Switched the oracle to use `PLANNER`. First real validation run: 30-step
random-policy trajectory, monotonic trace `[True]*28 + [False]*3`,
t*=27 (`'go to cabinet 8'` was the action that exceeded the remaining
budget for the PDDL-shortest completion). This is a genuine
causally-verified label, not a heuristic guess.

**Cost note:** PDDL replanning per prefix-check is noticeably slower than
scripted-expert checks (single-trajectory oracle labeling with T~30 took
>2 min wall time on CPU, i.e. multiple seconds per prefix check x ~30
prefixes). This is O(T) env resets + PDDL solves per trajectory and will
be the throughput bottleneck for building an oracle-labeled dataset at
scale -- needs a batch/parallel labeling script before Phase 1's actual
Oracle Attr-OPD run, and should probably shard across the CPU hold
allocation's 8 cores.

## Not yet started

- Actually running `train_attributor.py` (blocked on GPU allocation
  starting; code + data are ready).
- Evaluating the trained attributor against the official `score.py`
  metrics on val + the alfworld transfer set, and against the oracle
  ALFWorld labels from Phase 1 (the RQ3 connection this split was designed
  for).
- ALFWorld on-policy rollout loop / Oracle Attr-OPD training (Sec. 10) --
  code from 2026-08-28 (`src/error_attr/rollout.py`,
  `src/error_attr/policy/alfworld_agent.py`) exists but untested against a
  real policy forward pass; still needs a GPU to actually run.
- Everything past Phase 1/attributor-training (Phase 3 transfer proper,
  Phase 4 RL, Phase 5 multi-agent) -- not started, intentionally.

## Next concrete step

Once the L40S GPU allocation starts: run `scripts/train_attributor.py`,
then evaluate. See DECISIONS.md for the dataset-source and model-size
decisions behind this.
