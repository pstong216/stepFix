# Report: Training an Error Attributor, and What Its Output Does for Multi-Agent Self-Correction

Dates: experiments 2026-09-08 to 2026-09-15. Data, scripts and per-decision rationale are in `STATUS.md`, `TODO.md` and `DECISIONS.md` (#12–31). This is the English version of `report_attribution_and_self_correction.md`.

Organisation: (1) training results for the attributor (SFT / GRPO / OPD); (2) the multi-agent systems (MAS) used in the correction experiments; (3) their initial failure rates; (4) what the attributor says about MAS failures; (5) how the attributor's output is delivered to the MAS, and what happens; (6) how tone and role change MAS behaviour. All numbers are final values after two grading fixes (DECISIONS #25/#26); every experiment comes with one real example.

---

## 1. The error attributor (Model A): SFT, GRPO and OPD

### 1.1 Setup

- Model: Qwen2.5-3B-Instruct + LoRA (r=16, α=32, all linear projections), bf16, 3 epochs, lr 1e-4, sequence cap 8,192 tokens (over-length examples are dropped, not truncated).
- Data: the official Who&When Pro text release (`Leoxx/whowhen_pro`, 6,257 traces, 8 frameworks). **ALFWorld (311 traces) is held out entirely** as the transfer test; the other 7 frameworks are split 90/10 stratified by framework → train 5,353 (4,767 of them ≤ 8,192 tokens actually trained on), val 593. The release's video split (eva 748 / dvd 168, text-renderable) and image/GUI splits were not used.
- Input: the official `all_at_once` judge prompt (task instructions + the 17-mode error taxonomy + user question + full transcript + step-coordinate format + response format). Loss is computed only on target tokens.
- Output: three lines (Agent Name / Step Number / Error Mode). Scored with the official `whowhen_eval.score`, whose rules are framework-specific (debate/dylan on the round only; mathchat uses 2·round + position).

### 1.2 One SFT training example (`math_debate_0122`, prompt 24,212 characters)

**Prompt head (task instructions + taxonomy excerpt)**

```
# Task

You are an expert at diagnosing failures in agentic systems.

You will be given the transcript of an agentic system attempting to answer a user question.
The system failed because of a decisive error somewhere in the transcript. Your job is to
identify the first decisive error: the step that most directly causes the system to go wrong
and eventually produce an incorrect answer.

Report which agent made that decisive error, the exact step coordinate where it occurred, and
the best matching error mode from the taxonomy below. Then briefly explain your reasoning.

## Error Mode Taxonomy

- P.1: **Visual Misidentification** - Wrong object, entity, or text recognition in image input. ...
- R.1: **Hallucination** - Generating claims that are NOT grounded in any observation the agent
  has retrieved. ... Distinguished from R.2: in R.1 the wrong claim is ABSENT from all
  observations; in R.2 the wrong claim is traceable to something the agent actually saw but misapplied.
- R.2: **Reasoning Error** - The agent has the right high-level frame but makes a localized
  reasoning mistake ...
  (17 modes: P.1–P.2 perception, R.1–R.4 reasoning, PL.1–PL.2 planning,
   A.1–A.4 action, V.1–V.2 verification, C.1–C.3 coordination)
```

The taxonomy text is not ours: the official `whowhen_eval/prompts.py` generates it from the release's `taxonomy.yaml` (17 codes with `name` and `description`); the gold `ground_truth.mode` labels use the same codes.

**Prompt middle (question + transcript, one block per step with coordinate and agent name)**

```
## User Question

Let $a,$ $b,$ and $c$ be positive real numbers such that $a + b^2 + c^3 = \frac{325}{9}.$
Find the minimum value of \[a^2 + b^3 + c^4.\]
After your reasoning, enclose your final answer in \boxed{}. For example: \boxed{42}

## Transcript

Step 0.0 | Agent: agent_0
[output]
Given \( a + b^2 + c^3 = \frac{325}{9} \), we are to find the minimum value of ...
Let's use the method of **Lagrange Multipliers**. ...
[/output]
Step 0.1 | Agent: agent_1
[output] ... [/output]
Step 0.2 | Agent: agent_2
[output] ... [/output]
Step 1.0 | Agent: agent_0
[output]
Let's carefully solve the problem using the sounder approach suggested by Agent 2. ...
[/output]
...
```

**Prompt tail (coordinate format + response format)**

```
## Step Coordinate Format

step R.P where R is the round and P is the turn's position within that round. The agent
cannot see the other agents' turns in the same round.

## Response Format

Please answer in the following format, exactly:
Agent Name: (the agent ID whose turn first introduces the error)
Step Number: (the step coordinate, exactly as used in the conversation above)
Error Mode: (one of the error modes listed)
Reason: (one or two sentences explaining the error)
```

**Training target (loss only on these three lines)**

```
Agent Name: agent_2
Step Number: 1.0
Error Mode: C.3
```

The raw gold label is `{"agents": ["agent_2", "agent_1"], "round": 1, "mode": "C.3"}` (C.3 = over-reliance on other agents). Debate is scored on the round only, so the target coordinate is the first coordinate of the gold round, `1.0`.

**Training-data composition after the 8,192-token cap** (train rows: all / actually trained): smolagents 2,949/2,699, mathchat 828/815, magentic-one 730/716, debate 401/227, macnet 235/233, dylan 137/**4**, metagpt 73/73. Debate and dylan traces are long (multi-round × 3–4 agents), so the cap removed 43% of debate and almost all of dylan; Model A has effectively never seen dylan, which is its weakest val framework.

### 1.3 SFT results

| model | data | agent | step | mode | all-3 correct |
|---|---|---|---|---|---|
| untuned Qwen2.5-3B | val (in-distribution, n=593, no ALFWorld) | 77.6 | 39.5 | 16.9 | 4.9 |
| untuned Qwen2.5-3B | ALFWorld (n=311) | 100 | 8.4 | 18.6 | 2.9 |
| untuned Qwen2.5-7B (same prompt, zero-shot judge) | val | 90.6 | 62.6 | 16.0 | 9.6 |
| untuned Qwen2.5-7B | ALFWorld | 100 | 42.1 | 19.9 | 12.5 |
| **SFT 3B** | **val (in-distribution, n=593)** | 97.6 | 95.6 | 82.8 | **81.6** |
| **SFT 3B** | **ALFWorld (held-out transfer, n=311)** | 100 | 53.4 | 26.0 | **20.3** |

val = the 10% hold-out of the 7 training frameworks (smolagents 327 / mathchat 92 / magentic-one 81 / debate 44 / macnet 26 / dylan 15 / metagpt 8), 0% ALFWorld; the ALFWorld rows are a framework never seen in training.

Training curve: loss 2.04 → 0.014; eval loss monotone 0.083 → 0.024, no overfitting. Paired McNemar, SFT-3B vs the 7B judge: 438 : 11 discordant traces on all-3, p=4.7e-114.

Reading:
- The attribution skill is genuinely learned (untuned 3B: 4.9%).
- Scale alone buys no error-type accuracy: untuned 3B → 7B moves mode from 16.9% to 16.0%. The 17-way classification is learned from supervision, not read off the prompt.
- Under transfer, step localisation keeps about half its accuracy (53%) while mode falls to the 17-way guessing floor (~0.26): classification is the real transfer gap. SFT-3B on ALFWorld (20.3%) beats both the untuned 3B (2.9%) and the untuned 7B judge (12.5%): the learned skill transfers well enough to outperform a zero-shot judge twice its size.

### 1.4 GRPO results

Setup: trl 1.12 `GRPOTrainer`, 8 samples per prompt, three independent 0/1 rewards = the official scorer's agent / step / mode correctness, β=0.04, same parser as evaluation. Prompts ≤ 4,096 tokens (3,639 of the train set; dylan absent, debate 13).

| arm | start / data | result |
|---|---|---|
| 1 | SFT init, all prompts | **no gradient**: 85–95% of groups have 8 identical rewards (SFT has memorised the training set), so the group-relative advantage is zero |
| 2 | base init, all prompts | in-distribution step 0.40 → 0.58, mode unchanged; on ALFWorld statistically identical to the untuned base |
| 3 | SFT init, 714 "hard" prompts (reward variance under SFT sampling) | signal exists, but held-out mode **0.828 → 0.811** (0–2 fixed, 11–12 broken, p ≤ 0.013). Cause: selection bias — mathchat-R.3 is 43% of the hard set and RL learned the shortcut "mathchat → R.3"; stopped early at ~115 steps |

Conclusion: at this scale (~1,000 prompts, 200 steps, LoRA r=16) a sparse 0/1 verifiable reward does not teach the 17-way taxonomy and can erode it; SFT's dense supervision does. GRPO track closed (untried fixes: mode-balanced mining, mode-weighted reward, larger scale).

### 1.4b OPD results (the third training paradigm)

On-policy distillation, GKD-style: student = the 3B attributor (agent-text-only SFT init, merged) + fresh LoRA; teacher = the 7B attributor (same input format, frozen). The student samples its three-line answer on 5,018 in-distribution training prompts; loss = per-token full-vocabulary KL(teacher ‖ student) at the sampled positions; no gold labels; half an epoch; two arms (uniform token weights / Error Mode line weighted ×3).

| 3B attributor | val all-3 | ALFWorld step / all-3 |
|---|---|---|
| init (SFT) | 82.0 | 62.7 / 23.2 |
| + OPD, uniform | 80.9 (p=0.31) | 64.3 / 22.5 (n.s.) |
| + OPD, mode-weighted | 80.4 (p=0.06) | 62.1 / 22.2 (n.s.) |
| teacher, 7B | 83.8 | 91.3 / 38.9 |

Conclusion: OPD did not transfer the 7B's out-of-distribution ability to the 3B — both arms sit at their initialisation on both splits. The training log explains why: the per-token KL between student and teacher on training prompts is ~0.01 nats; the two models already agree on the training set, so on-policy sampling exposes no disagreement to distil. Across the three paradigms: SFT learns attribution; GRPO (sparse 0/1 reward) does not improve it and can erode mode; OPD (stronger same-format teacher, in-distribution prompts) is a no-op. What actually improved Model A was data (agent-text-only input) and scale (7B).

### 1.5 Input ablations and compressed-input retraining ("do we really need the full trace?")

Inference-time ablations first (Model A unchanged, input edited), then one retrained Model A per compression variant (the clean answer):

| Model A training input | chars kept | val all-3 | ALFWorld step | ALFWorld all-3 |
|---|---|---|---|---|
| full trace (reference) | 100% | 81.6 | 53.4 | 20.3 |
| **without environment / tool observations** | 75% | 81.5 (p=1) | **64.3** (p=4.5e-5) | 22.8 (p=0.17) |
| same, restricted to the reference's **identical 4,767** training rows | 75% | 82.0 | **62.7** (p=3.4e-4) | 23.2 |
| only lines containing numbers / answers | 65% | 72.5 (p=1e-8) | 37.3 | 13.2 |
| first line of every step | 10% | 49.2 | 12.5 (answers "step 1" on 82% of traces) | 5.1 |

**What "without environment / tool observations" removes.** The official renderers wrap text the agent did *not* generate in `[observation] … [/observation]` tags: ALFWorld's environment feedback, smolagents' code-execution logs. The `no_observations` variant deletes only those blocks; the agent's thinking (`[think]`), actions (`[action]`) and outputs (`[output]`) are untouched. The first two steps of one ALFWorld trace (`alfworld_alfagent_0001`, 13,742 → 11,564 characters):

```
full trace                                            observations removed
------------------------------------------------      ------------------------------------------------
Step 1 | Agent: agent                                 Step 1 | Agent: agent
[think]                                               [think]
I need to find two pencils to put in a drawer.        I need to find two pencils to put in a drawer.
Pencils are commonly found on desks or shelves ...    Pencils are commonly found on desks or shelves ...
[/think]                                              [/think]
[action]                                              [action]
go to desk 1                                          go to desk 1
[/action]                                             [/action]
[observation]                                         (deleted)
You arrive at desk 1. On the desk 1, you see a
book 1, a pencil 2, and a pencil 1.
[/observation]
Step 2 | Agent: agent                                 Step 2 | Agent: agent
[think]                                               [think]
I found two pencils on desk 1 as required ...         I found two pencils on desk 1 as required ...
[/think]                                              [/think]
[action]                                              [action]
take pencil 1 from desk 1                             take pencil 1 from desk 1
[/action]                                             [/action]
[observation]                                         (deleted)
You pick up the pencil 1 from the desk 1.
[/observation]
```

Same for smolagents: the agent's Thought + Code inside `[output]` stays; the `Execution logs: … [5 rows x 46 columns] Index([...])` inside `[observation]` goes (`databench_smolagents_0480`, 10,445 → 9,032 characters).

Which frameworks are affected: only **alfagent (ALFWorld, 311/311, 82% of characters kept) and smolagents (val 327/327, 83% kept)** contain `[observation]` blocks. The renderers for debate / dylan / mathchat / magentic-one / macnet / metagpt do not emit the tag (in mathchat the `user_proxy`'s returned code result is its own `[output]` as an agent, not an observation), so their prompts are byte-identical between the two variants. "Removing observations" therefore changes two things: at **training** time, smolagents — the largest training framework, ~55% — becomes shorter and loses its execution logs; at **test** time, ALFWorld loses its environment feedback. Both are single-agent-plus-environment traces, which is why the gain appears on ALFWorld transfer: the model learned on smolagents to localise errors from the agent's thinking and actions alone, and that habit transfers to the same trace type.

**What "restricted to the identical 4,767 training rows" means.** Examples over 8,192 tokens are dropped at training time. The full-trace training set loses 586 of 5,353 rows that way and actually trains on **4,767**; without observations the prompts are ~25% shorter, only 337 are over-length, and **5,016** rows train. That is a confound: the ALFWorld gain could come from "249 extra rows" rather than "no observations". The third row removes it with `--keep_ids`: the observation-free training set is restricted to exactly the 4,767 row ids the full-trace model trained on — same rows, each just shorter — with everything else unchanged. The result is statistically identical to the 5,016-row run (step 18:23, p=0.53) and still significant against the reference (46:17, p=3.4e-4), so the gain is the input format.

Reading:
- The attributor needs the **agents' own text**. Environment / tool observations carry no information in-distribution and are noise out of distribution: removing them costs nothing on val and adds +9–11 pp of step localisation on transfer (data-size confound closed). **This is the only intervention that narrowed the transfer gap**; the attributor's canonical input should be "agent text only".
- Every compression that removes agent prose loses accuracy, monotonically in the amount removed, and the error *type* is the first casualty: the nature of the error is written in the reasoning body, not in the first line or in lines that contain numbers.
- Side finding: the position of the decisive error depends on the task type — 76% in the second half of the trace for debate / math-chat frameworks, 82% in the first half for ALFWorld; with only the first half shown, step accuracy almost equals the fraction of traces whose gold step is visible. **Online attribution** (catching the error while the task runs) is therefore feasible for embodied tasks and structurally infeasible for debate-style ones — a property of the task, not of the model.

---

### 1.6 Scale test: 7B attributor (2×2: model size × input format, identical 4,767 training rows)

| Model A | val all-3 | ALFWorld step | ALFWorld mode | ALFWorld all-3 |
|---|---|---|---|---|
| 3B, full trace | 81.6 | 53.4 | 26.0 | 20.3 |
| 3B, observations removed | 82.0 | 62.7 | 26.7 | 23.2 |
| 7B, full trace | 81.8 | 86.5 | 34.1 | 32.8 |
| **7B, observations removed** | **83.8** | **91.3** | **39.9** | **38.9** |

Paired tests (ALFWorld, n=311): scale (7B vs 3B, both full trace) +12.5 pp all-3, p=4e-8; format (7B without observations vs 7B full) +6.1 pp, p=9e-4 — larger than at 3B (+2.9, p=0.15), and at 7B mode also improves significantly (23:5). The two effects add: 20.3 → 23.2 → 32.8 → 38.9. In-distribution all four models sit at 82–84%; neither scale nor format matters there.

Reading: the attribution task saturates in-distribution at 3B; out of distribution both scale and "agent text only" matter and stack — removing observations is not a crutch for small models, it helps *more* as the model gets stronger. The canonical Model A going forward is **7B, agent-text-only input**: it localises the decisive step in 91% of never-seen embodied traces and classifies the error type in 40%, the first time any run leaves the 0.26 guessing floor.

---

## 2. The multi-agent systems (MAS) used for the correction experiments

Every system is played entirely by **Qwen2.5-7B-Instruct (frozen)** on the same **593 tasks** (math 192 / scibench 225 / mmlu_pro 117 / gpqa 59, drawn from the Who&When Pro task pool). Each system emits the official Who&When Pro schema of the corresponding framework, so the attributor, renderers and scorer are reused unchanged. Three collaboration topologies:

| framework | topology | protocol | code |
|---|---|---|---|
| **debate** | peers | 3 agents. Round 0: independent answers. Round 1: each agent sees its own and its peers' previous answers and revises. Majority vote. | `src/error_attr/selfplay/debate_runner.py` |
| **mathchat** | reasoner + tool proxy (AutoGen MathChat reproduction) | `assistant` reasons and writes Python; `user_proxy` actually executes it and returns `(Result of the code run: ...)`, or says "Continue…" when there is no code; `assistant` emitting `\boxed{}` terminates; ≤ 5 turns | `mathchat_runner.py`, sandbox `pyexec.py` |
| **magentic-one style** | hierarchical | `orchestrator` writes one instruction per round and routes to a specialist: `coder` writes code / `computerterminal` runs the latest code block; only the orchestrator may state the final answer; ≤ 6 rounds, forced answer at the budget | `magentic_runner.py` |

No web or file tools, so only self-contained reasoning benchmarks are used.

**Debate's two system prompts (verbatim)**

```
round 0: You are one of several independent expert agents solving a problem. Think step by
step, then give your final answer in exactly the format the task specifies.

round 1: You are one of several expert agents in a debate. You previously answered the problem;
you now see the other agents' answers from the last round. Reconsider carefully -- keep your
answer if you are confident, change it only if another agent's reasoning is genuinely more
convincing. Think step by step, then give your final answer in exactly the format the task specifies.
```

**One mathchat trajectory (`selfplay_mathchat_math_0147`, gold 351, system answered 360)**

```
[user_proxy r0p0]  (AutoGen MathChat's fixed opening: use Python, print results, three solution
                    routes, put the answer in \boxed{}; then the problem)
[assistant  r0p1]  "To determine the number of cake-eating days from Amaretta's birthday (July 27) to
                    Enzo's birthday (September 3) each year from 2008 to 2016, we need to calculate ...
                    ```python ... ```"
[user_proxy r1p0]  "(Result of the code run:\n360\n)"
[assistant  r1p1]  "The total number of cake-eating days ... is \(\boxed{360}\)."
```

**One magentic-one trajectory (`selfplay_magentic_scibench_0554`, gold 0, system answered 1468.8)**

```
[orchestrator r0p0 -> coder]  "Write the formula for the rate of flow outward through a surface using
                               the divergence theorem. Then express the divergence of the velocity field ..."
[coder        r0p1]           "```python\nfrom sympy import symbols, div, integrate\n... div_v = div(v, (x, y, z)) ..."
[orchestrator r1p0 -> computerterminal]  "Evaluate the integral of the divergence over the volume of the cylinder ..."
[computerterminal r1p1]       "Execution failed. Output:\nError: ... AttributeError: 'Tuple' object has no attribute 'expand'"
[orchestrator r2p0 -> coder]  "Correctly calculate the divergence of the velocity field and then integrate it ..."
[coder        r2p1]           "```python\n# Correct divergence of the velocity field\ndiv_v_correct = z + 2*y**2 + 2*x**2 ..."
[orchestrator r3p0 -> FINAL]  "FINAL: \boxed{1468.8}"
```
(The true divergence is 2y and the flux through the symmetric cylinder is 0; on its second attempt the coder simply wrote down a wrong divergence.)

---

## 3. Initial MAS failure counts / rates

Grading rule: parse `\boxed{}` (brace-matched); correct if the normalised strings match **or** both sides parse as numbers within 1% relative error (SciBench's own rule is 5%).

| benchmark (tasks) | debate correct | mathchat correct | magentic-one correct |
|---|---|---|---|
| math (192) | 90 (47%) | 93 (48%) | 52 (27%) |
| scibench (225) | 84 (37%) | 92 (41%) | 57 (25%) |
| mmlu_pro (117) | 74 (63%) | 67 (57%) | 45 (38%) |
| gpqa (59) | 16 (27%) | 13 (22%) | 12 (20%) |
| **total / 593** | **264 (45%)** | **265 (45%)** | **166 (28%)** |
| **usable failures** (answered, wrong) | 329 | 326 | 415 |

- Debate and mathchat are on par; the hierarchical magentic system is clearly worse, with a typical pattern of "coder forgets to print → terminal returns nothing → orchestrator repeats the same instruction for several rounds → forced answer at the budget" (A.4 looping / C.1 orchestration errors in the Who&When taxonomy).
- The tasks are hard by construction: the pool is drawn from Who&When Pro, i.e. tasks on which other systems had already failed.
- Each framework contributes a seeded sample of **150 failures** to the correction experiments. Of debate's 150, 32 turned out to be correct after the grading fix (the original exact-match grader had judged `87.79` vs `+87.8` and `0.5` vs `\frac{1}{2}` wrong), so debate's analysis is restricted to **118 true failures**; mathchat / magentic samples were drawn after the fix, n=150.

---

## 4. What the attributor says about MAS failures

Procedure: render the system's own failed trajectory into the same `all_at_once` prompt used in training (Section 1.2's format, with the Qwen team's trajectory in place of the Who&When one); Model A decodes three lines greedily; 150/150 parse.

### 4.1 A complete example (debate, `selfplay_debate_mmlu_pro_0311`)

**Task**: "What tool can be used to perform SNMP enumeration? A. Whois … G. IP Network Browser H. Nmap … J. Tcpdump". Gold **G**; the team answered **H** (all three agents reasoned their way to Nmap in round 0 and confirmed each other in round 1).

**Transcript section of the attributor prompt (excerpt; 14,920 characters in total)**

```
## Transcript

Step 0.0 | Agent: agent_1
[output]
To determine which tool can be used for SNMP (Simple Network Management Protocol) enumeration,
let's analyze each option:

A. Whois - Used for querying information about internet domain names and IP addresses. Not for SNMP.
B. Nslookup - Used for determining network configuration and hostnames. Not for SNMP.
C. Wireshark - A network protocol analyzer ... not primarily an enumeration tool.
...
G. IP Network Browser - A utility for viewing and managing IP networks. Ca...
[/output]
Step 0.1 | Agent: agent_2
[output] ... [/output]
Step 0.2 | Agent: agent_3
[output] ... [/output]
Step 1.0 | Agent: agent_1
[output]
\boxed{H}
[/output]
Step 1.1 | Agent: agent_2
[output]
The task requires choosing the correct tool ... After considering the options and the
explanations provided by the other agents, the best choice remains Nmap ...
[/output]
Step 1.2 | Agent: agent_3
[output] ... my conclusion remains unchanged. The most appropriate tool ... is Nmap ... [/output]
```

**Model A's output (verbatim)**

```
Agent Name: agent_1
Step Number: 0.0
Error Mode: R.1
```

i.e. the first decisive error is agent_1's round-0 turn (its description of IP Network Browser is fabricated — R.1 hallucination); the other two agents followed.

### 4.2 Diagnosis distributions on the three frameworks (150 traces each)

| | debate | mathchat | magentic-one |
|---|---|---|---|
| **who** | spread over the three agents | `assistant` 150/150 | `coder` 80 / `orchestrator` 68 / `terminal` 2 |
| **which step** | concentrated in round 1 (the revision round) | step 1 (first answer) 84% | round 0 (first plan or first code block) 98/150 |
| **top-3 error types** | C.3 over-reliance on other agents 49% · R.2 reasoning error 33% · R.4 task misunderstanding 16% | R.4 task misunderstanding 57% · R.3 calculation error 24% · R.2 reasoning error 13% | R.1 hallucination 46% · R.4 23% · R.2 13% (plus 12 C.1 orchestration / C.2 communication) |

The two trajectories above are diagnosed as: mathchat → `assistant / 1 / R.4` (wrong approach in the first answer; the code computed 360 along that approach); magentic → `coder / 0.1 / R.1` (the coder's divergence formula has no basis).

Reading: the attributor reads the collaboration structure — the peer debate's typical disease is following the crowd, the tool-proxy pair's is pure reasoning / calculation error, the hierarchical system's is hallucination plus orchestration. On 232 traces with comparable gold labels, feedback built from Model A's predictions and feedback built from gold labels produced indistinguishable outcomes (2–4 traces flip per condition, p ≥ 0.25), so its output can be used directly as feedback content.

---

## 5. How the attributor's output is delivered to the MAS

### 5.1 From three lines to feedback text

Fixed templates; the error type is expanded to "taxonomy name + first sentence of its definition". For the Section 4.1 example, the five information levels read:

| condition | feedback text |
|---|---|
| **incorrect_only** (no-information control) | Your team's answer to this task was incorrect. Please review your team's work and provide a corrected final answer. |
| **reason_only** | Your team's answer to this task was incorrect. Somewhere in your team's work, the following error occurred: Hallucination -- Generating claims that are NOT grounded in any observation the agent has retrieved. Please review your team's reasoning and provide a corrected final answer. |
| **vague_reason** | Your team's answer to this task was incorrect. The error occurred somewhere in the first quarter of the interaction (roughly steps 1-2 of 6). The error was of this kind: Hallucination -- … Please review that part of your team's work and provide a corrected final answer. |
| **precise_reason** | Your team's answer to this task was incorrect. Specifically, at step 0.0, in agent_1's turn, the following error occurred: Hallucination -- … Please review that step and provide a corrected final answer. |
| **precise_only** | Your team's answer to this task was incorrect. Specifically, at step 0.0, in agent_1's turn, an error occurred. Please review that step and provide a corrected final answer. |

Three **deliberately wrong** controls use the same templates: `wrong_location` (a different agent's real turn in a different round — internally consistent but wrong), `placebo_reason` (a randomly substituted error type), `random_attribution` (random agent / step / type).

### 5.2 Injection protocols

**one_shot**: "# Task + ## Your Team's Attempt (this attempt was INCORRECT) + full trajectory + ## Coordinator Feedback + feedback text + respond with only the corrected final answer" is given to the same Qwen-7B in a single generation.

**Framework-native "one more round"** (main result): the feedback becomes the system's next-round input.

- debate: each agent receives (with the round-1 system prompt):

```
Problem:
{task}

Your previous answer:
{this agent's previous turn}

Other agents' answers (last round):
[agent_2]: {…}
[agent_3]: {…}

## Coordinator Feedback
{feedback text}

Give your (possibly revised) final answer in exactly the format the task specifies.
```
  Each of the three agents answers once; majority vote.

- mathchat: `user_proxy` appends one message to the same dialogue — `## Coordinator Feedback\n{feedback text}\n\nContinue solving. Put your corrected final answer in \boxed{}.` — and `assistant` gets up to two more turns (it may run code once more).
- magentic-one: `orchestrator` receives the full team transcript + `## Coordinator Feedback\n{feedback text}\n\nYour team's previous final answer was judged INCORRECT. Decide how to proceed.`; it may re-delegate for 1–2 rounds and must then emit `NEXT_SPEAKER: FINAL`.

### 5.3 Results: recovery rate (%)

| condition | debate (n=118 true failures) | mathchat (n=150) | magentic-one (n=150) | external reader* (n=2,328) |
|---|---|---|---|---|
| task_only (task text only) | — | — | — | 14.3 |
| no_feedback (trajectory, not told it failed) | — | — | — | 28.5 |
| incorrect_only | 5.1 | 7.3 | 3.3 | 42.4 |
| reason_only | 5.1 | 4.7 | 5.3 | 43.6 |
| vague_reason | 6.8 | 5.3 | 6.0 | 41.8 |
| precise_reason | 4.2 | 7.3 | 4.0 | 43.5 |
| precise_only | 5.9 | 6.7 | 4.7 | 40.5 |
| wrong_location (wrong) | 8.5 | 6.0 | 3.3 | — |
| placebo_reason (wrong) | 5.1 | 3.3 | 7.3 | — |
| random_attribution (wrong) | 6.8 | 4.0 | 4.7 | — |

\* External reader = a **fresh** Qwen-7B that reads the failed trajectory of **another system** (Who&When Pro; the four gradable frameworks mathchat / magentic-one / debate / dylan), receives the same feedback, and re-answers under the same "answer only" output constraint.

Statistics:
- **For the erring team itself**: in no framework does any correct-content condition beat "your answer was incorrect" (all paired McNemar p ≥ 0.34), and wrong content does no worse than correct content; one_shot and "one more round" are indistinguishable (p ≥ 0.18). The team keeps its old answer 60–65% of the time; when it changes, it is right only 10–15% of the time — below the 33% chance of merely switching options on a 4-way multiple-choice item.
- **For the external reader** (paired on the same 2,328 traces): seeing the trajectory +14 pp (no_feedback vs task_only, p=1e-33); being told it is wrong another +14 pp (incorrect_only vs no_feedback, 351:27, p=5e-73); adding the error type **+1.2 pp** (reason_only vs incorrect_only, 78:50, p=0.017); adding the exact step without a reason **−1.9 pp** (precise_only vs incorrect_only, 34:79, p=3e-5). The originally reported "why ≫ where" gap (reason_only vs precise_only, 111:38, p=2e-9) is two-thirds "a bare location pointer hurts" and one-third "the error type helps a little".

**The actual correction records for the Section 4.1 trajectory** (8 conditions, same trace):

| condition | team's new answer | recovered? |
|---|---|---|
| incorrect_only | H | no (unchanged) |
| reason_only | H | no (unchanged) |
| vague_reason | G | **yes** |
| precise_reason | H | no (unchanged) |
| precise_only | G | **yes** |
| expert_incorrect_only | G | **yes** |
| expert_precise_reason | G | **yes** |
| deferential_system_precise_reason | H | no |

This trace shows why the aggregate is null: with the same diagnosis (R.1 / agent_1 / 0.0), location-without-reason recovers and location-with-reason does not — the team's reaction is close to random and unrelated to content.

Reading (with the baselines): **for an external reader, the outcome label carries the effect (+14 pp), the error type adds ~1 pp, and a bare location subtracts ~2 pp; for the erring team, nothing about the content matters.** Both recipients respond to the signal "this is wrong", not to the diagnosis. This closes the research plan's route "use the attributor's output as feedback content to drive team self-correction" (at 7B), but not the attributor itself: it is accurate, it transfers, and its *location* output is exactly what a rollback procedure needs (Summary, item 5).

---

## 6. How tone and role change MAS behaviour

### 6.1 Manipulations

Content fixed; only the presentation (first sentence / system prompt) changes:

| condition | change |
|---|---|
| **expert_incorrect_only** | first sentence becomes "An independent expert reviewer has verified that your team's final answer is WRONG and has audited the full transcript. This diagnosis is reliable and should be acted on." — followed by **no diagnosis at all** |
| **expert_precise_reason** | the same authoritative lead + the full precise_reason diagnosis |
| **deferential_system_precise_reason** (debate only) | the system prompt "keep your answer if you are confident, change it only if another agent's reasoning is genuinely more convincing" is replaced by "Your team's previous answer has been verified as incorrect by an independent reviewer … Do NOT simply keep your previous answer: re-derive the solution from scratch, taking the reviewer's diagnosis into account" |

Tested first on debate's 150, replicated on **all** 173 remaining fresh debate true failures, then on mathchat's and magentic's samples (each extended to all of their fresh true failures: 326 and 415).

### 6.2 Results (paired McNemar, "A-only : B-only")

| contrast (same traces, paired) | debate (n=291) | mathchat (n=326) | magentic (n=415) | **pooled 1,032** |
|---|---|---|---|---|
| authoritative lead, no content vs neutral "you were wrong" | 10.7 vs 6.5%, 15:4, p=0.012 | 10.1 vs 5.5%, 21:6, p=0.006 | 8.0 vs 5.1%, 24:12, p=0.065 | **60:22, p=3e-5** |
| authoritative + diagnosis vs neutral + diagnosis | 10.0 vs 6.2%, 13:3, p=0.013 | 8.6 vs 5.5%, 18:8, p=0.08 | 7.2 vs 6.7%, 16:14, p=0.86 | **47:25, p=0.013** |
| authoritative + diagnosis vs authoritative alone (does content add to authority?) | 7:9 | 12:17 | 11:14 | 30:40, p=0.28 (negative direction) |
| neutral + diagnosis vs neutral alone (does content add anything?) | 8:9 | 12:12 | 21:14 | 41:35, p=0.57 |
| system prompt "re-derive from scratch" (debate only, 118) | 5.9% ≈ neutral; answer-change rate 49%, correct-given-changed only 12% | — | — | — |

**Behaviourally**: the authoritative lead raises the team's answer-change rate from ~37% to ~50%, but the probability that a changed answer is correct does not move (12–19%); "re-derive from scratch" has the highest change rate (49%) and the lowest correctness among changes (3–12%). Gains concentrate on the multiple-choice sets (gpqa / mmlu_pro) and are near zero on math.

### 6.3 Reading

- This is the **only** robust effect in the whole self-correction study: same sign in all three topologies (+3–5 pp each), significant alone in two of them, pooled over 1,032 paired traces p=3e-5; every replication was pre-specified on disjoint traces before its result was seen.
- Its mechanism is **social**, not **informational**: the speaker's authority loosens the team's attachment to its previous answer, while the diagnosis adds nothing on top of the authority (30:40 — if anything, slightly negative).
- For a 7B team, "who erred, at which step, with what error type" is not a lever; "how authoritatively it is told that it is wrong" is — and even that only makes it guess again, not guess better.

---

## Summary and next steps

1. **Attributor**: SFT learns a genuine, transferable attribution skill (in-distribution 81.6%; transfer step localisation 53%); GRPO and OPD do not improve on it at this scale. What did: input format ("agent text only": in-distribution unchanged, transfer localisation +9–11 pp, confound closed) and scale (7B: transfer step 91%, mode 40%, all-3 39%). The two effects are additive.
2. **Self-correction**: 7B teams in three collaboration structures are insensitive to feedback **content** — wrong content equals correct content equals none. The one effective lever is **authoritative framing**, and its mechanism is willingness to abandon the previous answer, not information.
3. **External readers** respond mainly to the outcome label (+14 pp); the error type is worth ~1 pp; a bare location pointer is harmful (−2 pp). Every feedback study needs the "told wrong, nothing else" zero.
4. **Where the attributor's value lies**: as an external judge and as a training / intervention signal — it is accurate, it transfers, and it knows *where*. It should not be used as content to tell a team what went wrong.
5. **Proposed next step — Attribution-Guided Rollback**: use the attributed step to truncate the failed trajectory, let a *fresh* instance (no memory of the old answer) regenerate that step conditioned on the diagnosis, and resume the protocol from there. This follows from findings 2–4 (the erring team is anchored; a fresh reader can use the outcome signal and, marginally, the diagnosis; a location handed to a reader distracts, but a location used to cut and regenerate never has to be interpreted). Controls: random-step rollback (does *where* matter?), rollback without diagnosis (does *why* matter?), equal-compute rerun from scratch (efficiency), and the existing feedback-injection arms. Verified rollback continuations would also supply the reliable per-step teacher that attribution-guided distillation was missing (appendix).

## Appendix A: two grading fixes made along the way

- The `\boxed{}` regex truncated nested braces (`\boxed{\frac{1}{64}}` → `\frac{1}`); replaced by brace-matched parsing.
- Grading was exact string match after normalisation (`87.79` vs `+87.8` counted wrong); replaced by "string match or numeric agreement within 1%".

Both changed many absolute numbers (every external-reader condition +11 pp; 86 of the debate pool's 862 "failures" were actually correct) but **did not change the direction or significance of any conclusion** — every comparison is paired, so the same trace is graded the same way under every condition. Re-runs and splices are documented in DECISIONS #25/#26.

## Appendix B: the policy-side Attr-OPD by-product

Before the request was clarified to mean OPD *for Model A*, attribution-guided on-policy distillation of the **debate team itself** was run (Research Plan Sec. 10): the team's failures on 355 train tasks were attributed by Model A; a privileged teacher (the same model, or a 14B, given the erring agent's exact turn prompt plus hindsight — team wrong, this turn decisive, error type, original response) regenerated that turn; the team's LoRA was distilled toward it at the same prefix without hindsight; arms attr / random-turn / uniform; greedy team accuracy on 238 held-out tasks. Teachers were correct at the attributed turn only 6–9% (self) and 14% (14B) of the time, and every unverified arm was at or below the untrained team (7B base 47.1%: attr 45.4%, random 42.4%, uniform 41.2%; 14B-teacher CE arms 39.5–42.4%, significantly worse; verified-only arms = base; 3B student likewise null). The by-product's conclusion — the sign of attribution-guided distillation is set by the teacher's correctness at the attributed step — motivates the rollback proposal above.
