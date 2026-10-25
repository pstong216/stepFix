# Research Plan: Failure Attribution as Transferable Supervision for Agentic Post-Training

## 1. Working Title

**From Diagnosis to Learning: Failure Attribution as Transferable Supervision for Agentic Post-Training**

Alternative title:

**Learning from Attributed Failures: Transferable Credit Assignment for Agentic Policies**

---

## 2. Central Research Question

> **Can failure attribution serve as transferable supervision for post-training agentic policies?**

Existing failure-attribution research primarily asks:

\[
\text{Where did the agent fail?}
\]

or, in multi-agent systems,

\[
\text{Who failed, and when?}
\]

However, accurate diagnosis is only useful if it can improve future behavior.

This project studies the next question:

\[
\boxed{
\text{Failure Attribution}
\rightarrow
\text{Training Credit}
\rightarrow
\text{Policy Improvement}
}
\]

Specifically, we investigate whether failure-attribution signals learned from existing causally labeled trajectories can transfer across tasks and agent architectures and provide useful supervision for post-training new agent policies.

---

# 3. Motivation

Most agent post-training methods rely on trajectory-level outcome rewards:

\[
R(\tau).
\]

For a failed trajectory,

\[
R(\tau)=0,
\]

but this does not tell us which intermediate decisions were responsible for the failure.

Consider:

\[
s_1
\rightarrow
s_2
\rightarrow
s_3
\rightarrow
s_4
\rightarrow
s_5
\rightarrow
\text{FAIL}.
\]

Suppose only the action at \(s_3\) caused the trajectory to become unrecoverable.

Trajectory-level optimization may implicitly penalize many actions:

\[
a_1,a_2,a_3,a_4,a_5,
\]

even though

\[
a_1,a_2
\]

were correct and

\[
a_4,a_5
\]

may simply be consequences of the earlier error.

Failure attribution instead provides:

\[
P(t\text{ caused failure}\mid\tau).
\]

This creates a natural temporal credit signal.

For multi-agent systems, attribution additionally provides:

\[
P(i,t\text{ caused failure}\mid\tau),
\]

where \(i\) identifies the responsible agent and \(t\) identifies the responsible step.

The central hypothesis is therefore:

\[
\boxed{
\text{Better failure attribution}
\Rightarrow
\text{better training credit}
\Rightarrow
\text{better agent learning}.
}
\]

---

# 4. Key Research Gap

Recent work has begun using hindsight information for agent training.

**OPID** extracts episode-level and critical-step skills from completed on-policy trajectories and uses them for self-distillation.

**AHEAD** analyzes completed trajectories, identifies error steps, generates corrective hints for those steps, and provides them to a privileged teacher during GRPO training.

**Critical Step Optimization (CSO)** identifies candidate critical decisions, proposes alternative actions, executes counterfactual continuations, and converts verified corrections into DPO pairs.

However, these works primarily construct supervision from the agent's current trajectories during training.

In contrast, **Who&When Pro** provides 12,326 failed trajectories with golden attribution labels constructed using controlled failure injection after exact replay of successful prefixes.

Who&When Pro primarily evaluates attribution itself rather than systematically studying how attribution should be converted into policy-learning supervision.

This motivates the main gap:

\[
\boxed{
\text{Can causally supervised failure attribution be learned once,
transferred to new agents,
and reused as a general post-training signal?}
}
\]

---

# 5. Core Hypothesis

We hypothesize that a failure attributor trained on causally labeled trajectories can learn transferable knowledge about agent failures:

\[
A_\phi:
\tau
\rightarrow
W,
\]

where \(W\) represents responsibility over the trajectory.

For a single-agent trajectory:

\[
W=
[w_1,\ldots,w_T],
\]

with

\[
w_t=P_\phi(t\text{ caused failure}\mid\tau).
\]

For a multi-agent trajectory:

\[
W=
[w_{i,t}],
\]

where

\[
w_{i,t}
=
P_\phi(
\text{agent }i
\text{ at step }t
\text{ caused failure}
\mid\tau
).
\]

This attribution signal can then serve as a common interface for multiple post-training algorithms:

\[
\boxed{
\tau
\xrightarrow{A_\phi}
W
\xrightarrow{\text{Post-training}}
\pi_{\theta'}
}
\]

---

# 6. Research Questions

## RQ1. Does attribution quality predict downstream learning utility?

Compare different attributors while holding the policy-learning algorithm fixed.

For example:

- Oracle attribution
- trained specialized attributor
- frontier-LLM attribution
- smaller LLM attribution
- heuristic attribution
- uniform credit
- random credit

Measure both:

\[
\text{Attribution Accuracy}
\]

and

\[
\text{Downstream Policy Improvement}.
\]

The main analysis asks whether:

\[
\operatorname{Accuracy}(A_\phi)
\]

correlates with:

\[
\Delta\operatorname{Performance}(\pi).
\]

An especially interesting result would be:

\[
\boxed{
\text{best diagnostic attribution}
\neq
\text{best training attribution}.
}
\]

This would suggest that *actionable attribution* is different from standard attribution benchmark accuracy.

---

## RQ2. What representation of attribution is most useful for learning?

Compare different forms of credit.

### Hard key-step attribution

\[
w_t=
\mathbf 1[t=t^*].
\]

### Soft responsibility distribution

\[
w_t=
P_\phi(t=t^*\mid\tau).
\]

### Local-window attribution

\[
w_t=
\begin{cases}
1,&t=t^*\\
\lambda,&|t-t^*|\le k\\
0,&\text{otherwise}.
\end{cases}
\]

### Failure-type-conditioned attribution

\[
W=
(w_t,m),
\]

where \(m\) represents reasoning, planning, perception, verification, action, etc.

### Multi-agent Who × When attribution

\[
w_{i,t}
=
P(i,t\text{ caused failure}\mid\tau).
\]

This tests whether fine-grained attribution improves learning over trajectory-level supervision.

---

## RQ3. Does attribution transfer across agents and tasks?

Train the attributor on one set of:

- agent architectures,
- benchmarks,
- modalities,

and evaluate it on trajectories generated by an unseen policy.

For example:

\[
A_\phi
\text{ trained on Who\&When Pro agents}
\]

then applied to:

\[
\tau^{\text{new}}
\sim
\pi_\theta^{\text{new agent}}.
\]

Two transfer questions should be separated.

### Attribution transfer

Does:

\[
A_\phi(\tau^{\text{new}})
\]

still correctly locate failures?

### Learning transfer

Even if attribution accuracy decreases somewhat, does using:

\[
A_\phi(\tau^{\text{new}})
\]

still improve the new policy?

The second question is ultimately more important.

---

## RQ4. Is failure attribution useful across different post-training algorithms?

The attribution signal should ideally not depend on one optimizer.

The two main realizations will be:

1. **Attribution-guided On-Policy Distillation**
2. **Attribution-guided Reinforcement Learning**

Optionally, DPO can serve as a third validation mechanism.

If attribution improves multiple training paradigms, it supports the stronger claim:

\[
\boxed{
\text{failure attribution is a general-purpose supervision signal}.
}
\]

---

# 7. Overall Framework

The complete framework consists of three layers.

## Layer 1: Failure Attribution

Given a completed trajectory:

\[
\tau=
(h_1,a_1,\ldots,h_T,a_T),
\]

the attributor predicts:

\[
A_\phi(\tau)
\rightarrow
W,m.
\]

For single-agent tasks:

\[
W=[w_1,\ldots,w_T].
\]

For multi-agent tasks:

\[
W=[w_{i,t}].
\]

The attributor is **hindsight-based**: it sees the complete trajectory.

It is not required during agent inference.

---

# 8. Attributor Training

## 8.1 Training Data

Primary source:

**Who&When Pro**

It provides causally constructed failure trajectories, where failures are introduced only after exact replay of a successful prefix, yielding controlled decisive-step labels.

Training examples can be represented as:

\[
D_{\text{attr}}
=
\{
(\tau_i,t_i^*,m_i)
\}_{i=1}^N.
\]

For multi-agent trajectories:

\[
D_{\text{attr}}
=
\{
(\tau_i,i_i^*,t_i^*,m_i)
\}_{i=1}^N.
\]

---

## 8.2 Architecture-Agnostic Trajectory Representation

Since source agents may differ from the target policy, raw internal prompts should not be treated as the canonical input representation.

Each step should ideally be normalized into:

\[
x_t=
(
q,
o_t,
r_t,
a_t,
e_t
),
\]

where:

- \(q\): task goal,
- \(o_t\): environment observation,
- \(r_t\): optional reasoning/intention,
- \(a_t\): action,
- \(e_t\): environment response/result.

Then:

\[
A_\phi(x_{1:T})
\rightarrow
W.
\]

This reduces dependence on:

- tool syntax,
- prompt templates,
- memory implementation,
- agent-specific action formats.

A key experimental question is whether this normalization improves cross-agent transfer.

---

# 9. Attributor Variants

The paper should deliberately compare several attribution approaches.

## A. Oracle Attribution

Use gold:

\[
t^*
\]

from Who&When-style controlled labels.

This represents an upper bound on the value of attribution.

---

## B. Frontier LLM Judge

Provide the full trajectory and ask a strong model to identify:

- responsible step,
- failure mode,
- optionally responsible agent.

This represents the common LLM-as-judge approach.

---

## C. Specialized Fine-Tuned Attributor

Train:

\[
A_\phi
\]

on Who&When Pro.

Possible model sizes:

- 1B
- 3B
- 7B

This is important because one motivation is **amortized attribution**:

\[
\text{expensive LLM analyzer every rollout}
\]

versus

\[
\text{small specialized attribution model}.
\]

---

## D. Heuristic Attributors

Examples:

- first environment error,
- last action before failure,
- maximum uncertainty step,
- maximum action entropy,
- repetition/loop detection.

These establish whether learned attribution is actually necessary.

---

## E. Random / Uniform Controls

Random:

\[
w_t\sim U.
\]

Uniform:

\[
w_t=\frac1T.
\]

These controls are critical because they isolate whether the improvement comes from attribution itself or merely additional training.

---

# 10. Post-Training Method I: Attribution-Guided OPD

This should be the primary training mechanism because it most directly uses attribution as supervision.

## Step 1: On-policy rollout

Current student:

\[
\pi_\theta
\]

generates:

\[
\tau
\sim
\pi_\theta.
\]

This ensures all states are sampled from the current policy distribution.

---

## Step 2: Hindsight attribution

After trajectory completion:

\[
A_\phi(\tau)
\rightarrow
W.
\]

Example:

\[
W=
[0.01,0.04,0.09,0.71,0.10,0.05].
\]

Step 4 receives most responsibility.

---

## Step 3: Privileged teacher

At state \(h_t\), the normal student observes:

\[
\pi_\theta(a\mid h_t).
\]

The teacher receives additional hindsight information:

\[
g_t,
\]

such as:

- this is a high-responsibility failure state,
- predicted failure mode,
- original failed action,
- optional failure critique.

Teacher:

\[
q_T(a)
=
\pi_T(a\mid h_t,g_t).
\]

Importantly, the teacher does **not** need to be a larger model.

The privileged advantage may come entirely from:

\[
g_t.
\]

A same-model self-distillation setting can therefore be tested:

\[
\pi_T
=
\pi_{\theta_{\text{old}}}.
\]

---

## Step 4: Student forward

The same logged prefix:

\[
h_t
\]

is fed to the student:

\[
q_S(a)
=
\pi_\theta(a\mid h_t).
\]

This requires a new model forward pass but **does not require environment replay**.

---

## Step 5: Attribution-weighted distillation

Define:

\[
\boxed{
\mathcal L_{\text{Attr-OPD}}
=
\sum_t
w_t
D_{\mathrm{KL}}
\left(
q_T(\cdot|h_t,g_t)
\|
\pi_\theta(\cdot|h_t)
\right).
}
\]

The attribution model therefore determines:

\[
\boxed{
\text{where and how strongly the teacher supervises the student}.
}
\]

For multi-agent systems:

\[
\mathcal L_{\text{Attr-OPD}}
=
\sum_{i,t}
w_{i,t}
D_{\mathrm{KL}}
(
q_{T,i,t}
\|
\pi_{\theta_i}
).
\]

This directly realizes:

\[
\text{Who}
+
\text{When}
\rightarrow
\text{Who should learn, and at which step}.
\]

---

# 11. Why OPD Does Not Require Counterfactual Replay

Suppose:

\[
t^*=4.
\]

The original trajectory already contains:

\[
h_4.
\]

Training only requires:

\[
h_4
\rightarrow
\text{Teacher Forward}
\]

and

\[
h_4
\rightarrow
\text{Student Forward}.
\]

The teacher/student decision distributions are compared at the logged state.

No browser/environment interaction is needed.

The student does **not** generate a new \(a_4\) and continue using the old trajectory's \(o_5\), since that would create an inconsistent trajectory.

Instead:

\[
\boxed{
h_{t^*}
\rightarrow
\text{current-decision distillation}
\rightarrow
\text{update}.
}
\]

After updating:

\[
\pi_\theta
\rightarrow
\pi_{\theta'},
\]

the next training iteration generates a completely new on-policy trajectory:

\[
\tau'
\sim
\pi_{\theta'}.
\]

---

# 12. Post-Training Method II: Attribution-Guided RL

The second method asks whether attribution can improve credit assignment without privileged distillation.

Standard trajectory-level RL produces an outcome advantage:

\[
A_\tau.
\]

Failure attribution provides:

\[
W=[w_1,\ldots,w_T].
\]

We construct:

\[
A_t^{\text{attr}}
=
f(A_\tau,w_t).
\]

A simple starting formulation is:

\[
A_t^{\text{attr}}
=
w_t A_\tau.
\]

The resulting objective is:

\[
\boxed{
\mathcal L_{\text{Attr-RL}}
=
-
\sum_t
A_t^{\text{attr}}
\log
\pi_\theta(a_t|h_t).
}
\]

More sophisticated mappings should be investigated because failure responsibility and reward credit are not necessarily identical.

Possible mappings:

### Linear

\[
A_t=w_tA_\tau.
\]

### Temperature-normalized

\[
\tilde w_t
=
\frac{
\exp(w_t/\tau)
}{
\sum_j\exp(w_j/\tau)
}.
\]

### Residual credit

Preserve standard trajectory advantage while adding attribution:

\[
A_t=
A_\tau+\lambda g(w_t).
\]

### Negative-only localization

Use attribution primarily for failed trajectories while leaving successful trajectories unchanged.

This is likely safer than aggressively redistributing positive reward.

---

# 13. Optional Method III: Attribution-Guided Preference Learning

For attributed failure state:

\[
h_{t^*},
\]

the original trajectory gives a known failed action:

\[
a^-.
\]

A teacher or counterfactual procedure proposes:

\[
a^+.
\]

Then construct:

\[
a^+\succ a^-.
\]

Preference loss:

\[
\mathcal L_{\text{Attr-DPO}}
=
w_t
\mathcal L_{\text{DPO}}
(
h_t,a^+,a^-
).
\]

This method is useful mainly as an additional validation that attribution benefits learning outside OPD.

It should probably not be the main method because CSO already performs verified critical-step DPO using counterfactual continuation.

---

# 14. Counterfactual Replay as an Optional Stronger Variant

Pure Attr-OPD does not verify whether the teacher's correction truly changes final task outcome.

A stronger version can replay to the attributed state:

\[
s_0
\rightarrow
\cdots
\rightarrow
s_{t^*},
\]

replace the original action:

\[
a^-
\rightarrow
a^+,
\]

and continue:

\[
s_{t^*}
\xrightarrow{a^+}
\cdots
\rightarrow
R'.
\]

This yields environment-verified correction.

However, replay is substantially more expensive and environment-dependent.

Therefore the recommended progression is:

### Main method

\[
\text{Attribution-Guided OPD}
\]

without replay.

### Stronger variant

\[
\text{Verified Attribution-Guided OPD/DPO}.
\]

This also creates a useful ablation:

\[
\text{unverified teacher}
\quad\text{vs.}\quad
\text{environment-verified teacher}.
\]

---

# 15. Recommended Agentic Benchmarks

The method should not be framed as Web-specific.

A good paper should ideally span multiple agentic settings.

## A. ALFWorld

Advantages:

- clean state transitions,
- multi-step interaction,
- relatively cheap rollouts,
- commonly used by OPID and AHEAD.

Useful for algorithm development.

---

## B. WebShop

Advantages:

- Web-style planning,
- simpler and more reproducible than WebArena,
- explicit task rewards,
- widely used in agent RL research.

This is likely the best second benchmark.

---

## C. Search / Research Agent

Possible choices:

- search-based QA,
- GAIA,
- DeepSearch-style tasks.

This tests whether attribution transfers beyond structured environments.

---

## D. Multi-Agent Benchmark

Potentially the most scientifically interesting extension.

Attribution becomes:

\[
P(i,t|\tau).
\]

This provides simultaneous:

\[
\boxed{
\text{agent-level credit}
+
\text{temporal credit}.
}
\]

It directly exploits the "Who" and "When" structure of failure-attribution benchmarks.

---

# 16. Experimental Design

The central experiment should be a matrix:

| Attribution Signal | OPD | RL |
|---|---:|---:|
| No attribution | ✓ | ✓ |
| Uniform attribution | ✓ | ✓ |
| Random attribution | ✓ | ✓ |
| Heuristic attribution | ✓ | ✓ |
| LLM attribution | ✓ | ✓ |
| Learned specialized attributor | ✓ | ✓ |
| Oracle attribution | ✓ | ✓ |

This allows clean isolation of:

\[
\text{Attributor Quality}
\times
\text{Learning Algorithm}.
\]

---

# 17. Main Evaluation Metrics

## Attribution Metrics

Depending on the benchmark:

### Step localization accuracy

\[
\operatorname{Acc}(t^*).
\]

### Top-\(k\) recall

\[
P(t^*\in\operatorname{TopK}(A_\phi)).
\]

### Distance error

\[
|t^*-\hat t|.
\]

### Responsibility calibration

For soft attribution:

- ECE
- Brier score
- NLL

### Multi-agent attribution

Measure:

- Who accuracy
- When accuracy
- Joint Who × When accuracy.

---

## Policy Metrics

### Final task success rate

Primary metric:

\[
SR.
\]

### Sample efficiency

Plot:

\[
\text{Success Rate}
\quad\text{vs.}\quad
\text{Environment Steps}.
\]

### Training efficiency

Measure:

- trajectories required,
- teacher calls,
- attributor inference cost,
- training FLOPs.

### Interaction efficiency

Average number of environment steps to success.

---

# 18. Most Important New Metric: Learning Utility of Attribution

Define:

\[
U(A)
=
\operatorname{Perf}
(
\pi_{\text{trained with }A}
)
-
\operatorname{Perf}
(
\pi_{\text{without attribution}}
).
\]

This gives each attributor a **learning utility score**.

Then compare:

\[
\operatorname{AttributionAccuracy}(A)
\]

against:

\[
U(A).
\]

The central scientific plot becomes:

\[
\boxed{
x=\text{Attribution quality},
\qquad
y=\text{Downstream learning gain}.
}
\]

This could be one of the strongest figures in the paper.

---

# 19. Transfer Experiments

Transfer should be a major component rather than a minor ablation.

## Agent transfer

Train:

\[
A_\phi
\]

on trajectories from agents:

\[
A,B,C.
\]

Evaluate on:

\[
D.
\]

Then measure:

1. attribution accuracy on \(D\),
2. downstream learning improvement of \(D\).

---

## Task transfer

Train on:

\[
\mathcal T_{\text{source}}
\]

and evaluate on:

\[
\mathcal T_{\text{target}}.
\]

Possible settings:

- train attribution on Web tasks → transfer to search,
- train on tool agents → transfer to embodied,
- leave-one-benchmark-out.

---

## Failure-type transfer

Train on some failure classes:

\[
\{
\text{planning},
\text{reasoning},
\text{verification}
\}
\]

and test whether knowledge transfers to related but unseen failure patterns.

---

# 20. Critical Ablations

## Attribution source

- Oracle
- fine-tuned model
- prompted LLM
- random
- uniform

---

## Credit representation

- hard key step
- soft posterior
- top-\(k\)
- local window
- full trajectory

---

## Teacher information

### T0

\[
T(h_t)
\]

No privileged information.

### T1

\[
T(h_t,\text{critical-step indicator})
\]

### T2

\[
T(h_t,\text{failure mode})
\]

### T3

\[
T(h_t,\text{failure critique})
\]

This tells us which attribution information actually creates useful privileged supervision.

---

## Distillation scope

Compare:

\[
t=t^*
\]

only,

versus:

\[
[t^*-k,t^*+k],
\]

versus all states weighted by:

\[
w_t.
\]

---

## Teacher size

Compare:

- same-model privileged teacher,
- stronger teacher model.

This tests whether gains come from:

\[
\text{better information}
\]

or simply:

\[
\text{larger model}.
\]

---

# 21. Baselines

The most important conceptual baselines are:

### Outcome-only RL

No explicit step credit.

---

### Uniform OPD

Teacher supervises every step equally.

---

### OPID

Uses on-policy hindsight skills and critical-step routing.

---

### AHEAD

Uses environment feedback and error-specific corrective hints for privileged self-distillation.

---

### Critical Step Optimization

Relevant mainly if DPO or counterfactual replay is included.

---

### LLM-as-Judge Attribution

Important baseline for determining whether a trained specialized attributor is actually useful.

---

# 22. Novelty Positioning

The paper should **not** claim:

> We are the first to distill critical steps.

That is already occupied by OPID/AHEAD-style methods.

It should also not claim:

> We are the first to post-train agents using critical states.

CSO already explicitly performs critical-step post-training.

Instead, the contribution should be:

### Contribution 1

**Failure attribution is treated as an independent, transferable training signal rather than merely a diagnostic output.**

---

### Contribution 2

**We introduce an attribution-to-credit interface that converts causal responsibility estimates into localized post-training supervision.**

\[
A_\phi(\tau)
\rightarrow
W
\rightarrow
\text{policy update}.
\]

---

### Contribution 3

**We systematically study the relationship between attribution accuracy and downstream learning utility.**

\[
\text{diagnostic accuracy}
\stackrel{?}{\Longrightarrow}
\text{optimization utility}.
\]

---

### Contribution 4

**We test whether an attributor trained on causally labeled failures transfers across agent architectures/tasks.**

---

### Contribution 5

**We demonstrate that the attribution signal can improve multiple post-training paradigms, such as OPD and RL.**

This is what makes the claim broader than another OPD variant.

---

# 23. Main Method Name

A neutral framework name would be preferable to an OPD-specific name.

Possible names:

### ACT

**Attribution-to-Credit Transfer**

### FACT

**Failure Attribution for Credit Transfer**

### FAIL2LEARN

**Failure Attribution for Agent Learning**

### TRACE

**Transferable Responsibility Attribution for Credit Estimation**

The framework could then contain:

- FACT-OPD
- FACT-RL
- FACT-DPO

as individual instantiations.

---

# 24. Expected Main Figure

The central architecture figure could be:

```text
        Causally Labeled Failure Data
                  │
                  ▼
        ┌──────────────────┐
        │ Failure Attributor│
        │       Aφ          │
        └──────────────────┘
                  │
          attribution model
                  │
──────────────────┼──────────────────
                  │
                  ▼
        New Agent Rollout
                  │
                  ▼
          τ ~ πθ
                  │
                  ▼
             Aφ(τ)
                  │
                  ▼
       Responsibility W
        /             \
       /               \
      ▼                 ▼
Privileged OPD      RL Credit
Teacher             Assignment
      │                 │
      └────────┬────────┘
               ▼
         Policy Update
               │
               ▼
              πθ'
```

The important point is that:

\[
\boxed{W}
\]

is the common interface between attribution and learning.

---

# 25. Minimum Viable Experiment

Before implementing the full framework, the first experiment should answer one simple question:

> **If we had perfect failure attribution, would agent post-training actually improve?**

Do not train the attributor first.

Use oracle attribution where possible.

Compare:

### Baseline

\[
\text{standard OPD / RL}
\]

### Oracle Attr-OPD

\[
w_t=\mathbf 1[t=t^*].
\]

If oracle attribution does not improve policy learning, then training a sophisticated attributor is unlikely to help.

If oracle attribution produces a substantial improvement, then there is a strong empirical motivation for the full project.

This should be **Experiment 0**.

---

# 26. Recommended Development Order

## Phase 1 — Oracle Feasibility

Implement:

\[
\text{Oracle Attr-OPD}
\]

on one simple benchmark.

Recommended:

\[
\boxed{\text{ALFWorld or WebShop}}
\]

Questions:

- Does localized distillation outperform uniform distillation?
- How large is the oracle upper bound?

---

## Phase 2 — Attributor Comparison

Implement:

- LLM judge
- fine-tuned attributor
- heuristics
- random/uniform.

Measure:

\[
\text{attribution quality}
\]

and:

\[
\text{learning utility}.
\]

---

## Phase 3 — Transfer

Train attributor on source-agent data.

Apply to unseen policy trajectories.

This is the main test of:

\[
\boxed{\text{transferable supervision}.}
\]

---

## Phase 4 — RL Extension

Use the same attribution output:

\[
W
\]

for RL credit assignment.

Show:

\[
\text{Attr-RL}
>
\text{Outcome-only RL}.
\]

This establishes optimizer generality.

---

## Phase 5 — Multi-Agent Extension

Use:

\[
w_{i,t}.
\]

Investigate whether attribution can correctly decide:

\[
\boxed{
\text{which agent should receive corrective learning}.
}
\]

This could substantially strengthen the paper if the first four phases work.

---

# 27. Main Risks

## Risk 1: Attribution transfer fails

A model trained on Who&When Pro may not generalize to a new agent architecture.

Mitigation:

- architecture-agnostic trajectory normalization,
- domain-adaptive fine-tuning,
- calibration on a small number of target trajectories.

---

## Risk 2: Attribution accuracy does not correlate with learning

This is scientifically interesting rather than necessarily fatal.

The paper can then show:

\[
\boxed{
\text{diagnostic attribution and actionable attribution differ}.
}
\]

This may motivate new training-aware attribution metrics.

---

## Risk 3: Privileged teacher corrections are wrong

Possible mitigations:

- teacher confidence filtering,
- ensemble teachers,
- environment verification for high-impact states,
- teacher-quality weighting.

---

## Risk 4: Attribution-weighted RL destabilizes training

Start with OPD because it provides a cleaner supervised signal.

Treat RL credit redistribution as the second method.

---

## Risk 5: Novelty overlap with AHEAD / OPID

Avoid making the paper about:

- critical-step distillation,
- corrective hints,
- generic hindsight supervision.

The paper must center on:

\[
\boxed{
\text{transferable learned failure attribution}
+
\text{its downstream learning utility}.
}
\]

---

# 28. Primary Scientific Claims to Aim For

An ideal final paper would establish the following.

### Claim 1

Failure attribution is useful beyond diagnosis:

\[
\text{attribution}
\rightarrow
\text{measurable policy improvement}.
\]

### Claim 2

Better attribution generally produces better learning:

\[
\operatorname{Quality}(A)
\uparrow
\Rightarrow
U(A)
\uparrow.
\]

### Claim 3

Soft responsibility can outperform binary key-step supervision.

### Claim 4

Attribution learned from one set of agents transfers to unseen agent architectures.

### Claim 5

The same attribution signal improves more than one post-training method.

### Claim 6

Attribution-guided supervision improves sample efficiency compared with trajectory-level learning.

---

# 29. Suggested Paper Contribution Statement

A concise contribution paragraph could eventually read:

> We investigate whether failure attribution can serve not merely as a diagnostic endpoint, but as transferable supervision for agent post-training. We introduce a unified attribution-to-credit framework that converts responsibility estimates over completed agent trajectories into localized learning signals. We study multiple attribution mechanisms, ranging from oracle and frontier-model judges to specialized attribution models trained on causally labeled failures, and evaluate their downstream utility under on-policy distillation and reinforcement learning. Across agent architectures and task families, we examine whether attribution quality predicts policy improvement, whether soft responsibility improves over trajectory-level credit, and whether attribution learned from one agent transfers to another.

---

# 30. One-Sentence Project Summary

\[
\boxed{
\textbf{
Train agents not only from the fact that they failed,
but from transferable knowledge of where, when, and potentially who caused the failure.
}
}
\]

The central conceptual shift is:

\[
\boxed{
\text{Failure Attribution}
:
\text{Evaluation Target}
\quad\longrightarrow\quad
\text{Training Supervision}.
}
\]

That should remain the identity of the project; OPD, RL, and DPO are mechanisms for testing whether that supervision is actually useful.