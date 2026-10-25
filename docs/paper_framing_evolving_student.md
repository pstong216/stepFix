# Paper framing: attribution must evolve with the student

Working title options
- *Chasing a Moving Error Frontier: Oracle-Free Step-Level Distillation for Self-Improving Agents*
- *Attribution Goes Stale: Why Error Localization Must Co-Evolve with the Learner*

## One-sentence thesis
When an agent improves by learning from its own failures, the place where it fails keeps moving; an error
attributor that does not move with it quickly becomes worse than a constant guess and can make the agent regress,
while an attributor refreshed from the agent's *own* data -- verification outcomes or successful trajectories --
keeps improvement going without any privileged oracle. What must be tracked is where the learner currently makes
*correctable* mistakes, not agreement with a reference label.

## Setting in one paragraph
ALFWorld; a weak 3B student (SFT on 100 expert trajectories, 53.7% on 774 test games); each round: collect the
student's failures, point at one step, let a teacher propose two alternatives, keep an alternative only if a greedy
continuation from that state wins (verification), distil it into the student, evaluate greedily. A PDDL planner is
used only as a ruler (never as a training input in the oracle-free variants).

## Named phenomena (the packaging)
| name | statement | key numbers |
|---|---|---|
| **Moving error frontier** | each round of learning pushes the student's first harmful action later | same game failing in both rounds: first harmful action later in 71-78% of games; median step 1 -> 5; relative position 0.02 -> 0.10 |
| **Attribution staleness** | a frozen attributor falls behind the frontier and its supervision becomes harmful | pointer within-2 agreement 65.1% (R1) -> 43.5% (R2), below a constant guess (67.8%); training with it: 64.3% -> 56.8%, 106 solved tasks broken vs 48 fixed; 33% of its pointers at step >= 10 |
| **Oracle-free refresh** | refreshing the attributor from the student's own data restores improvement | refresh from verification outcomes: R2 73.8% (vs 56.8% stale, 162/31, p=1e-22); progress estimator trained only on own wins (M5): R1 65.5%, the best round-1 pointer |
| **Agreement is not utility** | matching the oracle's error step does not predict training value | "always step 1": 62% exact agreement, no gain (55.2%); Model A: 15% exact, 64.3%; M5: 7.5% exact, 65.5%; verify-refreshed: 15% exact, 73.8% at R2 |
| **Luck-dominated failures** | as the student improves, more of its failures are sampling bad luck rather than a real error | original action also wins under greedy continuation: 22-30% (R1) -> 43-46% (R2) of verified states |
| **Generalization-driven repair** | improvement spreads beyond the trained states | untrained games' frontier moves later more often than trained ones (73-82% vs 62-65%) |
| **Verification as the active ingredient** | outcome verification, not localization precision, makes step-level distillation work and limits forgetting | no verification: 53.4% / 47.0%; unverified fallback: 54.0% / 59.7%; verified-only: 64.3% / 63.0%, retaining 88% of solved tasks; every-step imitation retains 7% |

## Contributions (as they would appear in the paper)
1. A controlled study of step-level verified distillation for a self-improving agent, identifying verification as
   the active ingredient for both gains and retention.
2. Evidence that the error distribution is non-stationary across self-improvement rounds, and that a frozen
   attributor goes stale to the point of causing regression.
3. Two oracle-free ways to keep attribution current: refresh from the loop's own verification outcomes, and a
   progress estimator learned from the agent's own successes.
4. The observation that oracle agreement is a misleading metric for attributors used as training signal, with a
   proposed alternative: downstream learner improvement per unit of verification budget.

## Figure plan
1. **Frontier shift**: distribution of the first harmful step, round 1 vs round 2 (paired games) -- the
   "Attribution Drift" page already has the absolute-step panels.
2. **Staleness**: left, pointer agreement R1 / R2-stale / R2-refreshed; right, downstream score of the same three.
3. **Agreement vs utility** (scatter, round 1, same start): x = exact agreement with the planner, y = pooled
   score; points: constant step 1 (62%, 55.2%), Model A (15%, 64.3%), M5 (7.5%, 65.5%), planner pointer (100%, 63.0%).
4. **Luck-dominated failures**: share of verified states whose original action also wins, R1 vs R2.
5. **Retention**: solved-task retention per recipe (verified-only vs fallback vs no verification vs every step).

## Honest limits (must be stated)
- Single training seed; one distillation round varies by ~4 points on its own (offline vs online KL on identical
  targets differed by 3.6). Differences under ~5 points are not interpretable yet.
- The strongest oracle-free pointers (verify-refreshed, M5) mostly point at the first action (M5: 75% at step 0).
  Constant step-0 control (same start, same recipe): 59.3% pooled vs M5 65.5% (84/36, p=1.4e-5) and Model A 64.3%
  (82/43, p=6e-4) -- the pointer matters beyond "train the opening". Caveat: 87% of M5 pointers are search moves
  (first visit to a new location), so part of its gain may be teacher location-guessing (see pointer_action_types).
- M5 has only one round; the co-evolution claim for it needs round 2.
- One environment, one deliberately weak student whose demonstrations never show search behaviour.
- The teacher's SFT data overlaps 45 held-out games (unseen/seen are clean; excluding them changes nothing).

## Experiments required before submission (priority order)
1. **Constant step-0 control** (round 1, same start as M5/attr_nofb) -- decides whether the attributor matters.
2. **Seeds**: 2 extra training seeds for student-facing arms: attr_nofb, M5, VL R2, stale R2, no-verification.
3. **M5 round 2** (refresh the estimator with round-1 wins) -- the cleanest test of "oracle-free co-evolution".
4. **Generality**: a medium-strength student (e.g. 300 demonstrations) or a second environment.
5. **Related work**: step-level process supervision with rollouts (IPR, Math-Shepherd), trajectory calibration and
   reflection (STeCa, Agent-R), step-level DPO (Step-DPO) -- position against verification-and-refresh.

## What goes to the appendix
Multi-agent inference-time feedback (null; more discussion rounds raise answer changes, not correctness), GRPO with
attribution masks, cheap signals and uncertainty (none predicts verification), relaxed success (adds noise),
untrained teacher (-5 points), success-data variants (continuation SFT +2.6, n.s.).
