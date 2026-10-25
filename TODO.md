# TODO -- "Is that really true?" exercises (advisor's framing, 2026-09-09)

Challenge common assumptions in multi-agent error attribution with
targeted experiments. Status legend: [ ] not started, [~] running, [x] done.
Details/results land in STATUS.md and docs/feedback_granularity_study.md.

## 1. Self-correction under the researcher's preferred protocol (`debate_round`)
- [x] `debate_round`, 150 failures, 5 conditions: NULL. Recovery 1.3-3.3%,
      no condition beats the no-info control (all 7 McNemar pairs p >= 0.5).
      Table in STATUS.md "TODO 1".
- [x] Paired vs `one_shot` on the same 150 traces: identical within noise
      in every condition (p >= 0.22); debate_round only nudges the
      answer-change rate (0.26-0.31 -> 0.31-0.35), none of it correct.
- Conclusion (DECISIONS #24): feedback type and injection protocol are
  both non-levers on this team's own failures; the untested lever is the
  revision mechanism (see backlog: hide previous answer / fresh corrector).

## 2. Immediately runnable, no retraining (advisor's two example questions)

### 2a. "Do we really need the exact step to fix an error?" -- feedback-side controls
Added to the chain (`--conditions`), RUNNING under `debate_round` on the SAME
150 traces (seed 0) on the new A100 (job 83939, tmux `chain_dr150_ctrl`,
reports/selfplay_chain_debate_round_150_controls):
DONE (450/450; table in STATUS.md "TODO 2a"): wrong_location 4.0%,
random_attribution 2.7%, placebo_reason 2.0%; pairwise n.s.; same band as
the main run's live tallies (2-5%). Wrong = right = no feedback.
- [x] `wrong_location`  -- does NOT hurt vs anything -> location is not used.
- [x] `placebo_reason`  -- ~= real reasons -> diagnostic content is not what
      drives the (rare) recoveries either.
- [x] `random_attribution` -- 2.7% (= a random attributor; item 3A anchor).
- [x] Paired cross-run test vs the main run: wrong_location 4.0% vs
      precise_only 2.7% (p=0.5); placebo 2.0% vs reason_only 2.0% (p=1);
      random 2.7% vs precise_reason 1.3% (p=0.5). Correct = wrong = none.
- [ ] Location-precision gradient: none / round-only / +-k window / exact,
      crossed with reason present/absent (curve, not two endpoints).

### 2b. "Do we really need the full trace for attribution?" -- input ablations on Model A
`scripts/ablate_attributor_inputs.py`, inference-time, val (593) +
ALFWorld transfer (311), official scorer. [~] RUNNING on A100 #1 (job
41169, tmux `ablate_inputs`, log reports/ablations_run.log, outputs
reports/ablations/) -- all 10 variants dry-run verified first:
[x] DONE (20/20 variant x split; both tables + takeaways in STATUS.md "TODO 2b").
Headline: compression hurts in-distribution but HELPS on ALFWorld transfer
(no_observations step +7pp) -- observations are OOD noise; error position is
framework-dependent (gold in 2nd half 76% val vs 1st half 82% ALFWorld) ->
online attribution feasible on embodied, infeasible on debate-style tasks.
- [x] full (reference) 0.815           - [x] no_observations 0.803 (-1pp; observations dispensable)
- [x] no_agent_ids: agent trivially 0, step 0.946, mode 0.762
- [x] no_task 0.794 (-2pp)             - [x] prefix_half 0.108 / suffix_half 0.518 -- STRUCTURAL: gold step
      in the 2nd half for 76% of traces; step acc == visibility cap; given visibility, suffix (31% chars) = full
- [x] shuffled 0.668 (order matters, less than content)
- [x] extractive 0.612 (65% chars; -19pp mode: prose reasoning lines carry the "why")
- [x] first_line 0.390 (10% chars, zero-shot)   - [x] lower bound final_only: mode 0.26 = guessing floor
Caveat: Model A was trained on full traces, so inference-time ablations
mix in distribution shift -> retrains (item 3) give the clean answer.
Picks for item 3 from this: extractive (biggest mode hit at moderate
compression -- retrain tells info-loss from format-shift), no_observations
(cheap, near-lossless -> expected to match full), first_line (extreme point, already running).

## 3. Needs retraining (clean answers)
- [x] first_line retrain DONE: val all-3 0.492 (zero-shot 0.390, full 0.815;
      mode collapses to 0.53), ALFWorld all-3 0.051 (degenerate: step "1"
      for 82% of traces). 10% compression = real information loss + destroyed
      transfer. Table in STATUS.md "TODO 3". keep_ids control NOT needed
      (it lost, with MORE data).
- [x] extractive retrain DONE: val all-3 0.725 (zero-shot 0.612, full 0.815;
      step 0.93 intact, mode 0.73), ALFWorld all-3 0.132 (= zero-shot 0.138,
      full 0.203). Half the in-distribution gap is format, half is the lost
      prose; OOD nothing is recovered.
- [x] no_observations retrain DONE: val all-3 0.815 (= full 0.816, p=1);
      ALFWorld step 0.534 -> 0.643 (p=4.5e-5), all-3 0.203 -> 0.228 (p=0.17).
      Observations are noise OOD, signal-free in-distribution. THE transfer lever.
- [x] `--keep_ids` data-size control DONE: val 0.820 (= full), ALFWorld step
      0.627 (vs 0.534, p=3.4e-4), = the 5,016-row run (p=0.53). Confound
      closed: the OOD gain is the input format. Item 3 COMPLETE (3/3 + control).
- [x] Re-SFT Model A on compressed/ablated input variants (~3.5h each) -- all three + control done:
      does training-time compression reach full-trace accuracy? Report
      accuracy vs compression ratio. Datasets BUILT (compression ladder):
      data/attributor_sft_first_line (~0.10), _extractive, _no_observations.
      first_line retrain RUNNING on a 4th GPU (A100 job 84810, tmux
      `sft_first_line`, log reports/attributor_first_line_train.log, output
      reports/attributor_first_line) -- independent of 2b's pick (extreme
      point of the ladder); 2b's results pick the rest. Evaluate with
      `eval_attributor.py --data_dir data/attributor_sft_first_line` (its own
      ablated prompts) on val + transfer, then `compare_attributors.py` vs SFT.
      **Confound to control:** compressed prompts never exceed 8,192 tokens,
      so first_line trains on all 5,353 rows vs the full-trace SFT's 4,767
      (586 long rows were dropped). If first_line matches/beats full, rerun
      it restricted to the same 4,767 ids (`--keep_ids` to add) before
      attributing the result to compression robustness.
- [ ] 3A. Accuracy -> utility curve: deliberately degraded attributors
      (random step / random mode / shuffled) as feedback sources, on the
      same 150 traces (partly covered by 2a's random_attribution).
- [x] 3B. LLM-judge baseline: untuned Qwen-7B zero-shot, identical prompt,
      val: all-3 9.6% (step 0.626, mode 0.160) vs SFT-3B 81.6% / untuned-3B
      4.9%. Scale buys step, not mode. 32B deferred (no L40S headroom, low
      expected information). Table in STATUS.md "TODO 3B".
- [~] extractive retrain RUNNING on A100 #1 (job 41169, freed by 2b; tmux
      `sft_extractive`, log reports/attributor_extractive_train.log, output
      reports/attributor_extractive). no_observations retrain RUNNING on
      A100 #2 (job 83939, freed by 2a; tmux `sft_no_observations`, log
      reports/attributor_no_observations_train.log) -- 2b says this variant
      should match or BEAT full on ALFWorld transfer; the key transfer-gap
      test. A100 #5 (job 88115, queued) reserved for the `--keep_ids` control.
- Prep done for item 3: `scripts/prepare_attributor_data.py --variant X
  --out_dir data/attributor_sft_X` + `scripts/train_attributor.py --data_dir
  data/attributor_sft_X --output_dir reports/attributor_X` (variants shared
  with the ablation via `src/error_attr/attribution/ablation.py`).

## 4. Self-correction beyond debate (researcher, 2026-09-09: "只有 debate 不够")
Two more Qwen-7B self-play frameworks, emitting the OFFICIAL Who&When Pro
schemas so Model A / renderers / the chain are reused unchanged:
- mathchat  (AutoGen 2-agent: assistant <-> user_proxy that EXECUTES Python;
  sequential reasoner + tool proxy)          src/error_attr/selfplay/mathchat_runner.py
- magentic-one (hierarchical: orchestrator routes each round to coder /
  computerterminal; orchestrator alone answers)  src/error_attr/selfplay/magentic_runner.py
- shared sandbox: src/error_attr/selfplay/pyexec.py (fresh interpreter, 30s timeout)
- collection: scripts/run_selfplay_collect.py --framework {mathchat,magentic-one}
  (lockstep batches, per-batch flush, --resume => safe on preempt-gpu)
- chain: scripts/run_selfplay_feedback_chain.py --framework X; protocol
  `debate_round` now means "framework-native extra round" (proxy delivers the
  feedback / orchestrator receives it and may re-delegate; runner.revise_many)
- [x] code written, CPU-tested (sandbox, proxy rule, orchestrator parser,
      renderer round-trip through the official mathchat/magentic renderers)
- [x] collect mathchat trajectories (593 tasks; 2h on a preempt A40). Re-graded
      (DECISIONS #26): math 93/192, scibench 92/225, mmlu_pro 67/117, gpqa 13/59
      (debate on the same tasks: 90 / 84 / 74 / 16) -> 326 usable failures.
- [x] collect magentic-one trajectories (593 tasks; 2h on an A100). Re-graded:
      math 52/192, scibench 57/225, mmlu_pro 45/117, gpqa 12/59 (lower than
      debate/mathchat -- the orchestrator loop fails more) -> 415 usable failures.
- [x] chain on mathchat DONE (+ expert_* add-on): content null (incorrect_only
      7.3% ties for best; all p >= 0.34); expert_incorrect_only 10.7 vs 7.3
      (p=0.23), expert_precise 10.0 vs 7.3 (p=0.42) -- same +3pp direction.
- [x] cross-framework table + POOLED framing test (591 paired traces):
      authority-no-content 33/10 p=0.0006; authority+content vs content 29/13
      p=0.02; content on top of authority 16/23 p=0.34; content alone 21/21 p=1.
      STATUS.md "TODO 4"; DECISIONS #28.
- [x] chain on magentic-one DONE (+ expert_* add-on): all content conditions
      3.3-7.3%, null (placebo_reason is the top cell); expert_incorrect_only
      7.3 vs 3.3% (p=0.15, same +4pp as debate, underpowered); expert_precise
      4.7 vs 4.0% n.s. Cross-framework table in STATUS.md "TODO 4".
- [x] cross-framework table (STATUS.md "TODO 4"): content null in all three;
      authority framing +3-4pp in all three, pooled p=0.0006.
- Not built: dylan (near-duplicate of debate with roles + agent dropping);
  code-task frameworks (macnet/metagpt) would need unit-test grading.

## 2a-bis. Framing conditions (researcher's "expert diagnosis" question) -- DONE; REPLICATED
- [x] 150-set (118 true failures): expert-framed cells top of the grid (8.5%), p=0.06-0.22.
- [x] Replication on the 173 fresh debate true failures + pooled n=291:
      expert_incorrect_only vs incorrect_only 10.7% vs 6.5% (16/4, p=0.012);
      expert_precise_reason vs precise_reason 10.0% vs 6.2% (14/3, p=0.013);
      content on top of authority 9/11 p=0.82; content alone 10/11 p=1.
      => authority framing is the first real lever (+4pp); diagnosis content is not.
- [x] Cross-framework replication + scale-up to ALL fresh true failures
      (mathchat 326, magentic 415; total 1,032 paired): authority-no-content vs
      neutral 60/22 p=3e-5 (debate 15/4, mathchat 21/6, magentic 24/12);
      authority+content vs content 47/25 p=0.013; content on top of authority
      30/40 (negative); content alone 41/35 p=0.57. FINAL.
- Table + reading in STATUS.md "TODO 2a-bis"; DECISIONS #28.
Same 150 traces, debate_round, content = precise_reason, presentation varied:
`expert_incorrect_only` (authority lead, no content), `expert_precise_reason`
(authority lead + content), `deferential_system_precise_reason` (system prompt
tells agents to re-derive given verified reviewer feedback). Motivation: the
original REVISE_SYSTEM says "keep your answer unless a PEER is more convincing"
-- the null may be a presentation effect. reports/selfplay_chain_debate_round_150_framing/.

## 5. Attributor scale test: 7B (researcher, 2026-09-09 17:00) -- DONE
Same LoRA recipe on Qwen2.5-7B-Instruct, paired: full-trace vs no_observations
(`--keep_ids`, identical 4,767 rows). Eval val + ALFWorld, compare_attributors
vs each other and vs the 3B pair (3B: full 81.6/20.3, no_obs 82.0/23.2; ALF step
53.4 -> 62.7). A100 172492 (`sft7b_full`) / 172493 (`sft7b_noobs`), ~7-8h each.
- [x] 7B full: val 0.818, ALFWorld 0.865/0.341/0.328   - [x] 7B no_observations: val 0.838, ALFWorld 0.913/0.399/0.389
- [x] 2x2 paired McNemar DONE: scale +12.5pp ALF all-3 (p=4e-8); format at 7B +6.1pp (p=9e-4),
      larger than at 3B and now moves mode too; additive. val: nothing. STATUS "TODO 5", DECISIONS #29.

## 6. Attribution-guided OPD (Research Plan Sec. 10) on 3B and 7B students -- RUNNING
Student = debate team (3 debaters sharing weights). Rollout -> Model A attributes
(agent i*, round r*, mode) -> privileged teacher = same base model given the exact
turn prompt + hindsight (team wrong; this turn decisive; error type; original
response) -> student LoRA distilled toward the teacher's turn at the SAME prefix
without hindsight (token-level KL, adapter-off teacher). Arms: attr (w=1 at the
attributed turn) / random turn (control: does WHERE matter?) / uniform (all turns,
1/T; the plan's baseline) / base. Tasks split 60/40 (355 train / 238 test,
seeded); OPD data = failures on train tasks; eval = greedy team accuracy on test.
Code: scripts/opd_{attribute_all,build_dataset,train,eval}.py, opd_pipeline.sh.
- [x] 7B student, 7B self-teacher: base 47.1%, attr 45.4% (p=0.60), random 42.4% (p=0.08),
      uniform 41.2% (p=0.06). Teacher correct only 8.8% -> distillation is damage; attr is the
      least damaging (plan's ordering, wrong sign). STATUS "TODO 6".
- [x] 7B student from 14B teacher (13.6% correct), CE: attr 39.5% (p=0.008 WORSE), random 42.4%,
      uniform 40.8% (p=0.03 worse); verified-only attr 46.2% / uniform 47.1% (= base). Verification
      removes the harm, too few rows to help. Item 6 CLOSED (by-product): sign = teacher correctness.
- [x] 3B student (3B team 33.7% on the pool, self-teacher 7.8% correct): base 29.0%, attr 30.7%
      (p=0.64), random 34.0% (p=0.11), uniform 29.4% (p=1). Null; mislocated arm is the best cell.
- [ ] paired comparison of the 4 arms per size on the 238 test tasks; teacher_correct rate;
      Verified-Attr-OPD variant (--verified_only) if unverified is flat.
- Not this round: ALFWorld + PDDL oracle as attributor (same code path, needs the env loop).

## 7. OPD for MODEL A (researcher, 2026-09-10 12:00: "训练 model A ... 还没用过 opd") -- DONE, NULL
Third training paradigm for the attributor after SFT (#1) and GRPO (#12-20).
GKD-style on-policy distillation: student = 3B attributor (agent-text-only SFT
init, merged) + fresh LoRA; teacher = 7B attributor (same format, frozen);
student SAMPLES its 3-line answer on train prompts, loss = per-token full-vocab
KL(teacher || student) at the sampled positions. No gold labels. Arms: uniform
token weights vs mode-weighted (x3 on the Error Mode line). Baselines: student
init (3B: val 82.0 / ALF 23.2) and teacher ceiling (7B: 83.8 / 38.9). Question:
does distilling ONLY on in-distribution prompts transfer the 7B's OOD ability to
the 3B (ALFWorld never seen)? Script: scripts/opd_attributor.py. Fresh A100 holds
256803/256804 requested (old holds expire ~17:50).
- [x] smoke test OK (vocab-padding fix; ~3 s/prompt, 1 epoch ~4h; train-prompt token-KL only 0.018 -- teacher and student already agree in-distribution)
- [x] uniform arm (0.5 epoch): val 0.809, ALF 0.643/0.257/0.225 -- = init (all p>=0.31)
- [x] mode-weighted arm (0.5 epoch): val 0.804 (p=0.064, slightly worse), ALF 0.621/0.267/0.222 -- = init
- [x] paired vs init / GRPO / teacher: OPD is a no-op; teacher 29pp higher on ALF step. STATUS "TODO 7", DECISIONS #30.
- [x] external-reader controls DONE (2026-09-12): task_only 14.3 -> no_feedback 28.5 -> incorrect_only 42.4;
      reason adds +1.2pp (p=0.017); precise location alone is -1.9pp (p=3e-5). "why >> where" was mostly
      "where hurts". STATUS section "External-reader study: the missing baselines"; DECISIONS #31.
(The policy-side Attr-OPD of item 6 was a misreading of the request; kept as a by-product.)

## 8. Location-precision (x / x+2 / x-2 / none) and Attr-GRPO (researcher, 2026-09-18) -- DONE, BOTH NULL; METHOD II CLOSED ON DEBATE (2026-09-20)
Queued as one GPU run (scripts/run_offset_experiment.sh, launched by the tmux waiter
`offset_exp` on whichever of holds 1251001 a100 / 1251002 preempt starts first):
- [x] [A-C] Model B: plus2 / minus2 on magentic / debate / mathchat: all n.s.; magentic x-2 vs x
      0/0 discordant; the team does not consume the location at any offset.
- [x] [D] external reader (434): none 38.5 / x-2 39.9 / x 39.6 / x+2 39.6; x+2 vs x 16/16 p=1;
      no length or framework pattern. Pointer exactness is irrelevant to readers too.
      STATUS "Location precision"; DECISIONS #33.
- [x] [E] Attr-GRPO pre-check, 3B team, G=8, T=1.0, 60 train tasks: **26/60 = 43% informative**
      (mixed) groups; 27 all-wrong, 7 all-correct; by benchmark mixed = mmlu_pro 9/18, scibench 8/22,
      math 6/16. Dynamic sampling needs ~9-10 sampled tasks per step for >=4 informative groups.
      7B probe skipped (7h at the probe's sequential pace; 3B answers the feasibility question).
- [x] [F] attr_grpo.py 2-step smoke test (A40, attr arm, G=4, 4 tasks/step): ran end to end,
      Model A parsed on every failed rollout (attr_fallback 0), mask share 0.375 / 0.583, KL 0 ->
      8e-4 after one update, ~290 s/step at the old per-task pace. Then two 1-step timing runs
      with the batched rollout code: A100-40 uniform = 952 s/step (853 s rollouts for 16 tasks x
      G=8, 99 s training); first attempts OOMed in generate() because the attention scores are
      materialised (n_seq x L^2) -> added a per-call attention budget (`--attn_budget`).
- [x] LAUNCHED 2026-09-18 22:14 CDT (account ee5239), same config for all four arms:
      G=8, 8 tasks/step, >=4 informative groups (max 2 resamples = <=16 tasks/step), 48 steps,
      lr 2e-5, beta 0.04, LoRA r32, T=1.0, 512 new tokens; greedy eval on a 60-task subset every 12
      steps; final: greedy + mean@4 (T=1) on all 238 test tasks, per-task records saved.
      uniform -> A100-40 hold 1333436 (agb02); random -> A100-40 1333437 (agb04);
      last -> A40 1348013 (agd06); attr -> H100 1333435 (e9), plus `--full_eval_at_start`
      (the base policy's full-test greedy + mean@4, shared reference for all arms).
      Outputs: reports/attr_grpo_3b/<arm>/ (symlink to /scratch.global). Compare with
      scripts/compare_attr_grpo.py. ETA ~14 min/step on A100 -> 12-14 h + evals.
- [x] Read out (2026-09-19 11:00, STATUS "Attr-GRPO ... READ-OUT"): mean@4 base 0.299, uniform 0.301
      (n.s.), random 0.274 (p=0.078), last 0.238 (p=0.0004); attr 0.268 (p=0.032; vs uniform p=0.048; vs
      random -0.006 p=0.74 -> location irrelevant). attr's sampled final was recomputed with
      `--eval_only` after a host-RAM OOM killed the original. Masks hurt through length drift / no-answer teams, not
      through worse answers (precision given an answer unchanged or higher).
- [ ] Follow-ups (all three were preempted once on preempt-gpu L40S nodes; each resumes from its
      last 12-step checkpoint with `--resume_from`): attr_committed from step 24 and random_soft
      from step 12 running since 16:00 on agc08 (holds 1414867/1414868, preemptible again);
      attr_soft (preempted at step 17) resuming from step 12 on whichever of holds 1415812
      (interactive-gpu, non-preemptible) / 1415813 starts first. Same 48-step protocol. Read out with
      scripts/compare_attr_grpo.py when done (~06:00-10:00 on 2026-09-20).
- [x] Soft arms done: attr_soft 0.297, random_soft 0.280 (diff +1.7pp, p=0.19); neither beats uniform
      (0.301) or base (0.299). Method II closed on debate as a negative result with mechanism (STATUS
      final table; DECISIONS #38). All Attr-GRPO holds released.
- Deferred: rollback / cut-point curve (judged too heavy); Attr-Distill with a strong
  single-call teacher (needs API access or a 2-4 GPU 32B/72B teacher -- researcher's call).

## 9. Long-horizon single-agent harness + step-localised Attr-OPD (researcher, 2026-09-20) -- ROUND 1 DONE, ITERATION RUNNING
Direction: ALFWorld first (substrate exists: src/error_attr/policy/alfworld_agent.py,
src/error_attr/attribution/oracle_alfworld.py), WebShop second. See DECISIONS #37.
- [x] A0. Zero-shot 3B diagnostic (budget 30, 464 games): success 12%; 100% of failures are budget
      exhaustion, oracle t* at 0.90 of the trajectory in 150/150 -> no decisive step for an
      incompetent policy (STATUS "ALFWorld single-agent track"). 14B teacher greedy 54% on unseen.
- [ ] A1. Warm-start: PDDL-planner expert at budget 50 (CPU job 1439539, shuffled game order) ->
      SFT 3B (2 epochs) -> greedy eval on unseen (budget 50) -> T=1 failure collection on 1,500
      train games (hold 1435620, scripts/launch_alfworld_sft_chain.sh).
- [x] A2 (zero-shot part). Two oracles on 150 zero-shot failures: point of no return t* (0.90 of the
      trajectory) vs first harmful action t_first_regress (mean step 7.6). Model A no_obs hits the first
      harmful action 46% exact / 55% within 2 (random 1% / 16%); full-input 29% / 41%; neither tracks t*.
      -> oracle arm = t_first_regress; Model A arm = 7B no_obs. STATUS "Two oracles, two answers".
- [ ] A2 (SFT student). Same two oracles + Model A on the SFT student's budget-50 failures; slack at the
      attributed step (budget - step - plan length) = room for an intervention.
- [x] B0. scripts/alfworld_attr_opd.py written and smoke-tested on GPU (choose / teacher / distill with
      verified alternatives; 0.5B stand-in: KL update + adapter save OK). Drivers: prepare_alfworld_opd_data.sh
      (Model A no_obs + full, progress oracle on all SFT failures, t* on 300; auto-starts after the SFT chain),
      run_alfworld_opd_arm.sh <hold> <arm> [max_failures] [verify_k] (choose -> 14B teacher targets ->
      verified distill from the SFT adapter -> greedy eval on the 134 unseen games).
- [x] A1 result: 3B SFT (549 traj) = 89.6% greedy on unseen -> too strong to be the OPD student and
      stronger than the 14B zero-shot teacher. Redesign: low-data student (SFT on 100 traj) + full-data
      teachers (3B-549 now, 7B-1338 training). Chains: launch_alfworld_student_chain.sh (SFT -> eval ->
      T=1 failures -> Model A + oracles), launch_alfworld_teacher7b.sh.
- [~] B1. Round 1 (TAG=student3b_low100, teacher = 3B-549 adapter; 319 shared failures; verify_k 2, cap 20):
      oracle 58.2% (+16.4pp, p=0.001), attr 47.8% (+6.0pp, p=0.28; vs random p=0.005), random 34.3%,
      last 32.8%, uniform (equal teacher calls) 8.2%. DONE -- STATUS "Attr-OPD round 1"; DECISIONS #39.
      attr_w1 (Model A step +-1, 769 states) running on the igpu hold (~22:00).
- [x] B1b. Variants on the same 319 failures: attr_w1 (Model A step +-1, 769 states) 16.4% -- widening the
      window HURTS (more fallback poison); attr_nofb (verified targets only) 58.2% unseen / 64.3% pooled = the
      oracle-with-fallback arm and the best single round; oracle_nofb 56.0% / 63.0% pooled (= attr_nofb, p=0.45);
      verify_k 3 (158 vs 125 verified examples, +48% env steps) 59.0% = k=2 (7/6, p=1). DECISIONS #40.
- [x] B2/B3 iterated rounds: oracle (with fallback) 58.2 -> 75.4 -> 76.9% on unseen (pooled 774: round 2 =
      76.5% vs student 53.7%); attr with fallback 47.8 -> 50.0% (null pooled); attr_nofb 58.2 -> 49.3% (round 2
      REGRESSED; verified-fix rate 39% -> 29%); attr_nofb round 3 running anyway (hold 1502947).
- [~] B4. Close the attr-oracle gap on later rounds (attributor curriculum, DECISIONS #41) -- ROUND 2 COMPOUNDS: attralf_nofb
      round 2 = 92/134 = 68.7% unseen (vs its round 1 56.7% p=0.005; vs attr_nofb round 2 49.3% p=6e-6; vs oracle round 2
      75.4% n.s.); 237/575 verified fixes (41% vs 29% with the original pointer). Queued: const1_r2 control, oracle_nofb r2
      remainder, attributor v2 + round 3 (`alfworld_task2.sh`), pointer-offset controls +-2/+5 (`--offset`).
      Earlier: Model A + 1,099
      ALFWorld self-play labels (progress-oracle first harmful action) -> exact 58% vs 19% on the oracle round-3
      failures; on the attr_nofb round-2 regression set exact 77.5% / within-2 83.2% vs 19.4% / 44.8% (315 labels).
      attralf_nofb round 1 = 56.7% (= attr_nofb, as expected on round-1 failures). RUNNING: attralf_nofb round 2
      (hold 1436314, roundN_v2) = the decisive test vs attr_nofb round 2's 49.3%; oracle_nofb round 2 (hold
      1435620) separates "iteration" from "pointer drift"; attr_nofb round 3 (hold 1502947); game-disjoint
      attributor check on held-out games 1500-1999 (CPU labels 1547834/1547835, both attributors already run).
      Next if round 2 compounds: refresh the attributor every round (build_attributor_alf_data.py ->
      finetune_attributor_alf.py -> launch_alfworld_roundN_v2.sh with ATTRIBUTOR=<new adapter>).
- [~] B5. Extended (774-game) evals (scripts/alfworld_eval_extended.sh; table: scripts/alfworld_pooled_report.py,
      per task type: scripts/alfworld_task_breakdown.py): done for student, oracle r1/r2/r3, attr r1/r2, random,
      last, uniform, attr_nofb r1/r2, oracle_nofb r1, attralf_nofb r1 (seen); running attralf_nofb r1 (held-out),
      then attr_nofb k=3; queued for every new round as it finishes.
      Original spec: recollect failures with the round-1 adapter -> re-attribute -> distill again;
      oracle started 16:25 on hold 1436314 (scripts/launch_alfworld_round2.sh), attr after uniform frees hold A.
      Original B1 spec: arms attr / oracle / random / uniform (equal teacher calls) / last, verify_k 2, cap 20,
      verify_k 2, continuation cap 20 steps; read-out = unseen success vs teacher calls vs env steps,
      paired McNemar vs the SFT student and between arms. Launch when PREPARE_DONE appears in
      reports/alfworld/prepare_opd.log (holds: 1436314 idle, 1446769 after the chain, 1435620 requeued).
- [ ] B. Verified Attr-OPD: replay prefix to t_hat, privileged teacher scores admissible commands,
      top-k alternatives continued by the student, environment-verified successes become the
      distillation target (KL over the admissible set at t_hat; optional uniform floor elsewhere).
      Arms: Model-A step / oracle step / random step / uniform OPD (equal teacher budget) / none.
- [ ] C. Eval on the 134 unseen games: success rate vs teacher calls vs env steps.

## Other assumptions worth challenging (backlog, not yet scheduled)
- [ ] "First decisive error is the one to fix": point feedback at a later /
      secondary error or the final step; compare recovery.
- [ ] "17 fine-grained modes are needed": 6 coarse categories vs 17 as
      feedback content.
- [ ] "Self-correction fails for lack of information": hide the team's own
      previous answer (force re-derivation); fresh instance of the same
      model as corrector (identity anchoring vs context anchoring).
- [ ] "Injected-error benchmarks reflect natural failures": Model A's mode
      distribution on injected vs natural self-play failures (C.3 was 74/150
      on self-play -- real or bias?); spot-check by hand.
- [ ] "Attribution needs the completed failure": prefix truncation as an
      online-monitoring feasibility test (overlaps 2b prefix_k).

## 10. Budget-constrained selection under attribution ambiguity (researcher's pivot + external review, 2026-09-22) -- IN PROGRESS

Reframed research question (supersedes the "adaptive window length" framing in Section 9 of
`docs/alfworld_attr_opd_summary.md`): given that attribution is ambiguous (multiple candidate repair
locations per failure, quantified in Section 7 of the summary doc) and the student keeps changing across
rounds, how should a fixed verification/training budget be allocated across candidates to maximise policy
improvement? Novelty check done: conformal-set attribution (arxiv 2605.06788) and LESS-style gradient data
selection (arxiv 2402.04333) are related-work / borrowed sub-components, not the main mechanism -- see
DECISIONS for the exact boundary. Steps, in order:

- [ ] 10.1 Reference-gradient calibration set: pick ~20-30 of attr_nofb round-1's 125 verified examples as
      a held-out calibration subset (excluded from all downstream scoring/selection experiments to avoid
      leakage); compute their averaged low-rank-projected gradient as the "useful update" reference direction.
- [~] 10.2 Implement three cheap (no-rollout) candidate scores, computable from things the pipeline already
      produces: (a) attributor confidence (Model A's top-1 vs top-2 margin / entropy over its own step
      distribution) -- BLOCKED: `alfworld_attribute.py` uses `model.generate(do_sample=False)` and only saves
      `raw`/`pred`/`t_hat`, no log-probs; needs a small code extension before this signal exists. (b) teacher
      admissible-command distribution entropy at the candidate state -- DONE, recomputed with a saved/checkable
      script (`teacher_entropy_score.py` -> `reports/alfworld/teacher_entropy_score_out.jsonl`) on the same
      957-candidate pool: **Pearson r=-0.0186, Spearman rho=-0.0183** -- no usable signal. Mean entropy for
      verified (outcome=1) vs skipped (outcome=0): 1.1033 vs 1.1328 (lower entropy weakly associated with success,
      directionally sensible but negligible in magnitude). Lowest-entropy top-K (50/100/200/300) precision:
      24.0%/29.0%/35.0%/35.3% vs 33.3% base rate -- non-monotonic, consistent with noise. **Note: this replaces an
      earlier reported r=0.052 from a one-off computation done before a context-compaction event, whose script was
      not saved to disk** -- the sign and magnitude differ (+0.052 vs -0.0186), and the discrepancy could not be
      reconciled since the original script no longer exists; both computations agree on the substantive conclusion
      (no usable signal, |r| an order of magnitude below anything actionable), but the earlier exact figure should
      not be cited going forward -- use this one, which has a saved, reproducible script. (c) LESS-style low-rank gradient cosine
      similarity to the 10.1 reference direction -- DONE: `scripts/alfworld_gradient_score.py` (feature-hashing
      sketch, not a dense projection matrix -- the first dense-matrix version was computationally infeasible and
      was rewritten) run on the full 957-candidate pool (calib_n=25, 932 scored). **Result: Pearson r=0.018,
      Spearman rho=-0.0015 against real verification outcomes -- no usable signal.** Mean grad_sim for verified
      (outcome=1) vs unverified (outcome=0): 0.168 vs 0.162 (negligible). Top-K precision by grad_sim
      (K=50/100/200/300): 28.0%/39.0%/36.5%/32.7% vs 31.5% base rate -- non-monotonic, consistent with noise.
- [x] 10.3 Score all existing round-1 candidates we already have full verification ground truth for (the
      attr_nofb / attr_nofb_p1 / attr_nofb_p2 offset pool, 957 states incl. skipped, minus the 10.1 calibration
      subset) with all three scores; correlate each against the real (already-known) verification
      success/failure per state. **Result: all three cheap (zero-rollout) signals tested this session -- teacher
      entropy (r=0.052), retrieval-based verb-typicality (10.10, within-2 9.3%, ~random), and LESS-style gradient
      similarity (r=0.018) -- failed to predict real verification outcomes.** A consistent negative result across
      three structurally different signal designs, not one weak signal -- real risk to the "budget-constrained
      selection via cheap signals" premise of Section 10. Before 10.4/10.5, decide: (a) try a 4th signal class
      (attributor confidence, still blocked on 10.2a) before concluding cheap signals don't work here, or (b)
      accept the negative result and reframe around a *partial/cheap-but-not-free* signal (e.g. a short 3-5 step
      rollout instead of the full 20-step verification) rather than a truly zero-rollout static feature.
- [ ] 10.4 Candidate-pool construction (simple, ours, not conformal prediction): temperature-sampled Model A
      step predictions (multiple generations per failure -> empirical set of distinct candidate steps),
      quantifying per-failure attribution uncertainty by how much the samples disagree.
- [ ] 10.5 Core comparison at matched budget (the revised go/no-go criterion): from the 10.4 pool, compare
      (a) score-based selection (best signal from 10.3) picks the top-B candidates, (b) random selection of B
      candidates from the *same* pool, (c) status-quo single-pointer selection, all at identical total
      teacher-call / verification-rollout budget B. Train + evaluate (unseen/seen/heldout/pooled) each.
      Success criterion: does (a) reliably beat (b)? If not, diagnose signal/pool quality before concluding
      the direction fails.
- [ ] 10.6 Location-vs-action-count budget allocation (Section 7's 51.4% union-coverage motivates this):
      same total verification budget, compare "4 candidate actions at 1 location" vs "2 candidate actions each
      at 2 locations" (from the 10.4 pool) vs the same allocation at depth-matched random locations
      (`random_matched` arm, already implemented).
- [ ] 10.7 Evolving-student check: repeat 10.1-10.5 on the round-2 failure pool (already collected from
      earlier rounds); test whether (a) the round-1-calibrated reference gradient direction / cheap-signal
      ranking still predicts round-2 verification outcomes, or needs recalibrating (the same staleness failure
      mode already demonstrated for the raw attributor in Section 4 -- attr_nofb regressing at round 2 -- but
      now at the *scoring/selection* level); (b) the winning selection strategy from round 1 still wins at
      round 2.
- [ ] 10.8 Write up: update `docs/alfworld_attr_opd_summary.md` with the full pipeline and results; state the
      contribution boundary explicitly (candidate-pool generation and gradient-scoring machinery are borrowed/
      adapted from prior work; the novel parts are the budget-matched selection comparison, the repairability-
      vs-training-utility separation already demonstrated this session, and the evolving-student recalibration
      failure mode).
- [x] 10.9 Retention/forgetting check (researcher's idea, 2026-09-22): does training on failed trajectories
      break tasks the pre-distillation student already solved? Zero new compute -- joined the existing 774-game
      pooled eval (baseline `student3b_low100` vs each round-1 arm, matched by `game_file`) restricted to the
      416 games baseline already won. Result, % of those 416 retained after training: `uniform` (every step of
      every failed trajectory supervised) 7.0% -- near-total collapse; `random` (one uniformly random step, no
      localisation) 57.0%; `attr`/`oracle` (fallback-allowed) 72.4%/79.8%; `attr_teacheronly`/`oracle_teacheronly`
      (localised, no verification) 64.3%/72.9%; `attr_nofb`/`oracle_nofb` (localised + verified-only) 88.5%/87.7%
      -- best retention of any real arm. Conclusion: verification protects against catastrophic forgetting, not
      just improves the fix rate -- a new, independent angle on top of Section 8b's "imprecise + unverified is
      worse than either alone" finding. Added to the write-up (10.8).
- [x] 10.10 Oracle-free localization signal from successful trajectories (researcher's idea, 2026-09-22): instead
      of the privileged PDDL-planner oracle (ALFWorld-specific, unavailable in most real domains), retrieve
      same-task-type trajectories where the baseline student already succeeded (feasibility checked: only 6
      ALFWorld task types, each with 37-105 baseline-won reference trajectories -- plenty of density) and use
      them to localize / propose a repair candidate for a failed trajectory.
      **Cheapest first cut result: NEGATIVE.** Prototype (zero GPU compute): phase = count of "milestone" verbs
      (take/move/clean/heat/cool/use/slice) completed so far, task-agnostic proxy for subgoal progress; built
      P(verb | task_type, phase) from 1,195 baseline-won trajectories (177 populated cells); predicted critical
      step = argmin over the failed trajectory's own steps of P(actual verb | task_type, phase(t)); correlated
      against real t_first_regress on 450 failed training-pool trajectories. Result: exact match 0.9%, within
      +-2 only 9.3%, mean |offset| 21.8 / median 20 steps -- indistinguishable from random, far worse than any
      real pointer. Diagnosis: this operationalization only compares the *verb*, not its target (most real
      ALFWorld errors are "right verb, wrong object/receptacle", which is invisible to a verb-only signal), and
      several (task_type, phase) cells are data-sparse, so rare verbs late in a trajectory get spuriously
      flagged as "least typical", biasing predictions late. A verb+target-role signal might fix this but needs
      extra task metadata (which object/receptacle is the actual goal target) and is no longer a free prototype
      -- paused here pending a decision on whether that extra investment is worth it before pursuing further.

## 11. Self-distillation ablation: is the teacher the source of the forgetting in Section 8c? (2026-09-22) -- CLOSED

**Correction (2026-09-24):** the text below originally called `attr_nofb`'s teacher "the (large, 14B) teacher". That
was wrong: every canonical round-1/round-2 arm uses the 3B SFT teacher (`--teacher Qwen/Qwen2.5-3B-Instruct
--teacher_adapter reports/alfworld/student3b_sft/final`, 549 expert trajectories, 89.6% unseen), confirmed from the
run logs (only 434-tensor = 3B models are loaded). So the comparison here is "stronger same-size SFT teacher vs the
student itself", not "14B vs 3B". The verdict below is unchanged; only the motivation's wording was wrong.

Motivated directly by Section 8c: `attr_nofb`/`oracle_nofb` still lose 11-12% of the pre-distillation student's
previously-solved games, and the hypothesis is that pulling the (small, 3B) student toward the (large, 14B)
teacher's candidate ranking is itself a source of distributional shift, independent of localisation/verification
quality. Test: rerun the *exact* `attr_nofb` recipe (same 319 states from `attr_nofb/steps.jsonl`, same verify_k=2,
same exclude_a_minus, no fallback) but with candidates proposed and weighted by the **student's own current
checkpoint** instead of the external teacher -- reusing `cmd_teacher` unchanged with `--teacher
Qwen/Qwen2.5-3B-Instruct --teacher_adapter reports/alfworld/student3b_low100/final` (the same starting adapter
`cmd_distill` itself starts from) in place of the 14B teacher. Zero new code; only the candidate-source
config changes, isolating "who proposes/weights the candidates" as the sole variable against `attr_nofb`.

- [x] 11.1 `self_nofb` arm run to completion (`reports/alfworld/opd/student3b_low100/self_nofb/`): 87/319 verified
      (27.3%, vs `attr_nofb`'s 125/319=39.2%, confirming 11.3's prediction that the 3B student's own top-2 ranking
      wins the verification rollout much less often than the 14B teacher's). **Result: pooled 466/774=60.2%,
      retention of the 416 baseline-won games 379/416=91.1%.**
- [x] 11.2/11.3 resolved together: raw retention (91.1%) looks better than `attr_nofb`'s 88.5%/`oracle_nofb`'s
      87.7%, but 11.3's predicted confound is exactly what explains it -- self_nofb trained on far fewer verified
      examples (87 vs 125/124), so it moved the policy less in EVERY direction, not more safely. Controlling for
      training volume via newly-fixed : newly-broken ratio (per-arm, same 416-game baseline reference): self_nofb
      **2.35**, attr_nofb **2.71**, oracle_nofb **2.41** -- self_nofb is not better, it is marginally *worse* than
      attr_nofb and about the same as oracle_nofb once volume is controlled for. **Verdict: this experiment does
      NOT support the hypothesis that the external teacher is a distinct source of the Sec. 8c forgetting.** The
      forgetting looks like a roughly constant side effect of verified training itself (any candidate source),
      scaling with how much training happens, not with where the candidates come from. Closes this thread --
      write up in 10.8 as a tested-and-rejected hypothesis, not left open.

## 12. Moving error frontier + planner-free refresh of Model A (2026-09-23/24) -- IN PROGRESS

Findings so far (all from existing data, CPU only):
- [x] 12.1 Frontier moves within the same game: on games failing in both rounds (184-221 per chain), t_first_regress
      moves later in 71-78% of games after one round (median +2-3 steps; relative position 0.02 -> 0.08-0.10). Not
      only a composition effect. Games whose round-1 state was actually verified+trained shift later LESS often
      (62-65%) than untrained games (73-82%) -> the shift looks like global generalization, not per-game repair
      (trained n=34-41 only, and selected, so directional evidence).
- [x] 12.2 Pointer vs planner agreement (within +-2): round 1 stale Model A 65.1% (constant "step 1" 84.3%);
      round 2 stale 43.5% (constant "round median" 67.8%); round 2 planner-refreshed 69.2%, exact 60.2% (best
      constant 58.9%, exact 10.7%). Visualised in artifact "Attribution Drift".
- [x] 12.3 E1 (memorisation check on the planner-refreshed Model A): refresh data (v1, 1,099 rows) covers 770 train
      games. On attralf round-2 failures from games NEVER in the refresh data (n=149): exact 52.3% vs best constant
      15.4% (content signal real), but within +-2 63.1% = best constant 63.1% (tie). The within-2 advantage came from
      seen games. No sign of copying labels from near-identical oracle-chain round-2 trajectories.
- [x] 12.4 Leakage audit vs test splits: OPD training pools (all rounds/chains) 0 overlap with unseen/seen/heldout;
      student SFT (100 expert traj) 0. **Teacher SFT (549 expert traj, `expert_all_train.jsonl`) contains 45 of the 500
      heldout games; Model A's v1 refresh data contains 22 heldout-game rows (source `zeroshot`, collected with
      `--n_games 2500`).** unseen and seen are clean everywhere. Paths are indirect (teacher/attributor only act on
      train-pool states during OPD), but the report's "heldout never touched by SFT or OPD" is false and must be
      corrected; cheapest check = rescore heldout/pooled without those games (CPU).
- [x] 12.5 **My error:** the corrected-looking `const1_nofb` rerun of 2026-09-23 used the 14B zero-shot default teacher
      (run_alfworld_opd_arm.sh default; 579-tensor model in the log) instead of the canonical 3B SFT teacher, so its
      44.8% unseen is NOT comparable to attr_nofb/oracle_nofb. Moved aside to `const1_nofb_14bteacher` (+ its ext eval
      files); rerun queued with the correct teacher.

Running (hold 1964218, `scripts/alfworld_queue_vl.sh`, log `reports/alfworld/queue_vl.log`), in order:
- [~] 12.6 Planner-free refresh: label = earliest step, among steps tried by planner-free round-1 arms (attr_nofb,
      attr_nofb_p1/p2, const1_nofb_14bteacher's verify data counts too since the label only needs "a replacement
      won", self_nofb), at which a replacement action was verified to win -> 187 labelled failures of 319
      (`data/attributor_sft_verify`, `scripts/build_attributor_verify_data.py`). Fine-tune from the original Model A,
      5.9 epochs (= the planner refresh's ~1.1k samples). Evaluate (a) agreement on attr_nofb's 528 round-2 failures
      vs planner labels, alongside the planner-refreshed v1 on the same failures; (b) downstream `attrvl_nofb` round 2
      from attr_nofb/round1 (only the pointer differs from attr_nofb round 2 = 56.8% pooled).
      **(a) done, NEGATIVE:** on the 382 planner-labelled round-2 failures: verify-refreshed exact 15.2% / within-2
      47.9% vs stale 19.1% / 43.5%, planner-refreshed v1 78.3% / 83.2%, constant step 1 26.2% / 52.4%, best constant
      (3) 12.8% / 67.8%. It collapsed early: 89.5% of its predictions at steps 0-2 (round-2 planner labels: 39.5%;
      median 3). Diagnosis: verify labels are 86% at steps 0-2 and earlier than the planner label 63% of the time
      ("earliest repairable" != "where it went wrong": an alternative action before the real error, with a good
      continuation, can also win), and they are round-1 data. **Confound in the reference:** v1's 1,099 rows are not
      round-1-only -- only `low100_r1` (450, label median 1) is; `oracle_r2` (440, median 6) is a sibling chain's
      ROUND-2 trajectories, plus `zeroshot` (112, median 6) and `strong` (97, median 16). So VL-vs-v1 mixes label
      source, data round (frontier lag) and size. Controls added (queue2, run first): PL450 = v1's 450 round-1 rows
      only (2.45 epochs); PL187 = planner labels on the same 186 failures as VL (5.9 epochs). If PL450 also
      collapses early, the problem is frontier lag of previous-round data, not verify labels.
      **Controls done (2026-09-24):** PL450 (round-1 planner labels only) exact 78.0% / within-2 82.2% (new games
      n=198: 73.7% / 80.3% vs best constant 11.1% / 73.2%) = v1 -> frontier lag is NOT the problem; v1's round-2 rows
      added nothing. PL187 (planner labels on the SAME 186 failures, same epochs) 72.8% / 77.0% (new games n=319:
      71.5% / 76.2% vs 12.9% / 68.7%) vs VL 15.2% / 47.9% -> **the verify label definition is the cause**, not data
      size (186 -> 450 adds ~5 points). "Earliest repairable step" is not "where it went wrong"; with only early steps
      tried it degenerates into a position prior. Proposed planner-free replacement (not started): first action that
      increases the teacher's greedy steps-to-success from the next state (teacher completion length as a proxy for
      the planner's remaining plan length); check agreement vs planner on a few dozen failures BEFORE fine-tuning.
      **(b) done, POSITIVE downstream:** `attrvl_nofb` round 2 (from attr_nofb/round1, same 528 failures/teacher as
      the stale branch): unseen 70.9%, seen 76.4%, heldout 73.8%, **pooled 73.8%** vs stale branch 56.8% (162/31,
      p=1.3e-22) and vs its start 64.3% (114/41, p=3.8e-9); vs oracle_nofb R2 76.0% (35/52, p=0.086) and attralf_nofb R2
      76.2% (34/53, p=0.053) -- different starts. Unchanged with the 67 contaminated heldout games removed (73.6%).
      Verified 231/528 (44%) vs stale 155/528 (29%). Its chosen steps: 91% at steps 0-2 (stale: 51%, and 33% >= 10).
      So planner agreement (15% exact) does not predict downstream utility; the gain may just be "stop pointing late".
      Missing same-start controls: `attrpl_nofb` (12.9, queued) and a constant-early-step branch from attr_nofb/round1.
- [x] 12.7 const1_nofb rerun with the 3B SFT teacher: verified 114/319, unseen 51.5%, **pooled 55.2%** = student 53.7%
      (94/83, p=0.45), far below attr_nofb round1 64.3% (49/120, p=4.7e-8), although step 1 matches the round-1
      planner label exactly 62% of the time. Another case of planner agreement != training utility.
- [ ] 12.8 Finish round-2 oracle-pointer offsets: -1 (resume distill), +1, +2 (-2 done: unseen 85/134 vs exact 98/134).
- [-] 12.9 CANCELLED by the researcher (2026-09-24 16:xx; never started). Was: `attrpl_nofb`: round 2 from attr_nofb/round1 with the
      planner-refreshed v1 pointer, same 528 failures and teacher -> the clean stale / planner-refresh / verify-refresh
      three-way from ONE checkpoint that Section 4 of the report flagged as missing. Note: v1 on this pool already
      scores exact 78.3% / within-2 83.2% (new games n=122: 72.1% / 80.3%, best constant 9.8% / 72.1%).

## Closed (for the record)
- [x] Location vs reason for EXTERNAL readers: reason is the active ingredient (n=2,328).
- [x] Attributor accuracy vs usefulness: Model A's predictions ~= gold labels (n=232).
- [x] Self-correction under `one_shot`: null; anchoring dominates (n=361).
- [x] GRPO on the attributor (3 arms): sparse reward does not learn/improve mode at this scale.
