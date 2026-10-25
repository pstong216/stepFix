# Self-evolution without rollouts: proxy filters for self-generated corrections (3B, ALFWorld, Oct 1–2 2026)

Code: `scripts/proxy_eval_se3b.py` (signals, AUC), `scripts/proxy_arms_se3b.py` (selection rules → training targets),
`scripts/run_proxy_arms*.sh`, `scripts/run_pure_self2.sh`, `scripts/self_evolve3b_report.py` (paired read-out).
Logs: `reports/alfworld/self_evolve3b/queue.log`, per-arm dirs under `reports/alfworld/self_evolve3b/proxy_arms/`.

## 1. Common setup
- Student: Qwen2.5-3B-Instruct + LoRA, SFT on 100 expert trajectories; greedy, budget 50. Baseline **136/274** (unseen 56/134, seen 80/140).
- Failures: 319 own failed train trajectories (`student3b_low100_train_b50.jsonl`, the attr_nofb subset).
- State: one pointer step t per failure (Model A unless stated; `m5_plain` uses the M5 progress-estimator pointer).
- Candidates: the student itself scores every admissible command at t. *Hindsight* teacher = prompt contains the whole failed
  trajectory (future actions/observations); *plain* = ordinary policy prompt. Two candidates = top-2 excluding the logged action a⁻.
- Training: one state → one-hot (or verified-distribution) target over the admissible commands, KL(target‖student), lr 1e-5,
  1 epoch, batch 8, LoRA continued from the student adapter (`alfworld_opd_plus.py train`).
- Evaluation: 274 held-out games (valid_unseen 134 + valid_seen 140), greedy; paired exact-binomial McNemar on discordant games.
- Rollout labels (for analysis only): every candidate was rolled out by the student (≤20 steps) in the rollout-verified arms, so
  each candidate has a "would pass verification" label; proxy rules never use it except `fut_pass`/`fut_fail`.

## 2. Per-candidate signals (exact definitions)
Entities = regex `[a-z]+ \d+` over lower-cased text (e.g. `mug 2`, `cabinet 3`). `obj` = goal object type from `parse_goal`.
`known(t)` = entities in the initial observation ∪ all `obs`/`next_obs` of steps < t.

**One-step environment evidence** (`evidence()` in `env_evidence_filter.py`): replay a₀..a_{t−1} in a fresh env, execute the
candidate c once, read o⁺.
- `nothing_happens` = o⁺ starts with "nothing happens"
- `n_new` = |entities(o⁺) − known(t)|
- `goal_revealed` = some new entity has type `obj`
- `subgoal` = o⁺ matches `you (pick up|clean|heat|cool|move) the <obj>` or starts with "you turn on the desklamp"
- `no_info` = nothing_happens ∨ (n_new = 0 ∧ o⁺ does not match `you (pick up|clean|heat|cool|move|open|turn on)`)
- **`env_score` = 2·goal_revealed + 2·subgoal + min(n_new, 5)/5 − 2·no_info**

**Failed-log future** (`future_feats()` in `proxy_eval_se3b.py`): later = {j > t : a_j = c}.
- `appears_later` = later ≠ ∅; t₂ = min(later); `gap` = t₂ − t
- at t₂ apply `evidence()` to the *logged* next_obs with known(t₂): `productive_later` = goal_revealed(t₂) ∨ subgoal(t₂)
- `progress_after` = milestone level (rule classifier `classify_full`) after t₂ exceeds the level at t₂
- **`fut_score` = 2·goal_revealed(t₂) + 2·subgoal(t₂) + min(n_new(t₂),5)/5 − 2·no_info(t₂) + progress_after**, 0 if never taken later

**Hindsight audit** (`hindsight()`): for c = "go to X" / "open X", scan steps ≥ t of the failed log for the same go/open of X
(skipping go-to's that found X closed); `seen_there` = obj among entities of that next_obs.
`audit` = 1.0 supports / 0.0 contradicts / 0.5 silent (never visited) / 0.5 n/a (not go/open).

**Model signals** (state-level unless noted): hindsight prob `hind_p`, plain prob `plain_p` (per candidate), entropies `H_hind`,
`H_plain`, `KL_hp` = KL(hindsight ‖ plain) over admissible commands, `hind_minus_plain` = log hind_p − log plain_p.

**Detour detector** (`alfworld_detour_detector.py`): LoRA from the student answering "Necessary: yes/no" after the policy prompt +
"Action: c"; trained on 6,000 steps of the student's own WON trajectories labelled by replay-verified detour pruning
(`compress_trajectories.py`: a step is necessary iff it survives pruning and the pruned action list still wins on exact env
replay). `p_necessary` = P(yes).

## 3. Selection rules (per state, over the two candidates)
| rule | which candidate | keep the state if | cap |
|---|---|---|---|
| verified (rollout) | the one(s) whose student continuation wins | any wins | — (86) |
| all_top1 | hindsight top-1 | always | 319 |
| hind_top | hindsight top-1 | — | 86 highest `hind_p` |
| env_filter | veto `no_info`; max (`env_score`, `hind_p`) | `env_score` > 0 | — (232) |
| env_top | as env_filter | — | 86 highest (`env_score`, `hind_p`) |
| fut_filter | max (`fut_score`, `hind_p`) among `appears_later ∧ productive_later ∧ audit ≠ contradicts` | any such | — (85) |
| fut_pass / fut_fail | fut_filter's pick | would pass / would fail rollout | 46 / 39 |
| combo_filter | veto `no_info` and `contradicts`; max s, **s = env_score + fut_score + 2·(audit − 0.5)** | s > 0 | — (137) |
| combo_top | as combo_filter | — | 86 highest (s, `hind_p`) |
| det_top / det_filter | max `p_necessary` | — / p > 0.5 | 86 / 180 |
| kl_top / ent_top | hindsight top-1 | — | 86 highest `KL_hp` / `H_plain` |
| placebo | a random admissible action ≠ a⁻ | the 86 rollout-verified states | 86 |

## 4. Yield on the 319 states (no cap) and rollout pass rate of what each rule keeps
| rule | targets | would pass rollout |
|---|---|---|
| rollout verified | 86 | 100% |
| fut_filter | 85 | 54% |
| combo_filter | 137 | 45% |
| det_filter | 180 | 24% |
| env_filter | 232 | 28% |
| all_top1 | 319 | 18% |

## 5. Results (274 games; paired tests in `self_evolve3b_report.py`)
Rollout-verified, hindsight candidates, by step rule: Model A 148, self-attribution 150, random-early 144, M5 142,
max-entropy 132; **placebo 141**; none beats the placebo. Reference: plain candidates + rollout (earlier) 158;
external 3B SFT teacher (549 demos) + Model A 169.

Proxy-selected, no rollout (Model A pointer states, hindsight candidates unless noted):
| arm | targets | pass% | unseen | seen | total | vs student | vs placebo | vs rollout-verified (148) | vs ext. teacher (169) |
|---|---|---|---|---|---|---|---|---|---|
| **env_top** | 86 | 60% | 78 | 94 | **172** | | | | |
| **combo_top** | 86 | 53% | 68 | 96 | **164** | 38/10 p=6e-5 | 36/13 p=0.001 | 26/10 p=0.011 | 18/23 p=0.53 |
| fut_pass | 46 | 100% | 74 | 91 | 165 | | | | |
| **m5_plain/combo_top** (no external component, plain candidates) | 86 | n/a | 68 | 93 | 161 | 38/13 p=6e-4 | 34/14 p=0.006 | 23/10 p=0.035 | 21/29 p=0.32 |
| fut_filter | 85 | 54% | 68 | 92 | 160 | 34/10 p=4e-4 | 29/10 p=0.003 | 23/11 p=0.058 | 15/24 p=0.2 |
| det_top | 86 | 29% | 71 | 87 | 158 | 32/10 p=9e-4 | 29/12 p=0.012 | 21/11 p=0.11 | 18/29 p=0.14 |
| attr_plain/combo_top (plain candidates) | 86 | n/a | 68 | 89 | 157 | | | | |
| fut_fail (proxy-approved, all fail rollout) | 39 | 0% | 64 | 86 | 150 | 20/6 p=0.009 | 24/15 p=0.2 | 19/17 p=0.87 | 15/34 p=0.009 |
| sft_hard (objective ablation on the 86 verified states) | 86 | 100% | 64 | 84 | 148 | | | = KL 148 | |
| kl_top | 86 | 21% | 51 | 77 | 128 | | | | |
| hind_top | 86 | 23% | 45 | 70 | 115 | | | | |
| ent_top | 86 | 17% | 46 | 69 | 115 | | | | |
| env_filter (wide) | 232 | 28% | 55 | 59 | 114 | 28/50 p=0.017 | | | |
| all_top1 (no filter) | 319 | 18% | 26 | 34 | 60 | | | | |

Proxy quality on all 3,190 rollout-labelled candidates (AUC / within-state accuracy where exactly one candidate won):
env_score 0.743/0.773, goal_revealed 0.717/0.723, n_new 0.681, fut_score 0.677/0.645, audit 0.677/0.673,
productive_later 0.671, p_necessary 0.633/0.568; hind_p 0.526, plain_p 0.536, H 0.52–0.53, KL_hp 0.527 (state-level signals
cannot separate two candidates of the same state: within-state 0.5 by construction). Veto `no_info` keeps 67% of candidates and
83% of rollout winners (win rate kept 19.7% vs vetoed 8.1%).

Flip probe (argmax changes on 150 student states): external teacher 36%, placebo 22%, rollout-verified hindsight arms 13–19%,
combo_top 18%, fut_filter 17%, hind_top 17% (→115), env_filter 42% (→114), all_top1 37% (→60). Change magnitude does not
determine outcome; target content does.

## 6. Caveats
- Each arm is one training run; 274 evaluation games. Training-seed replications of env_top / combo_top / placebo are running
  (`scripts/run_replications_oct2.sh`). The compression-SFT line showed large sample-to-sample swings (keep1 165 → 136 on a
  second 200-episode sample; pruned 146 → 160), so single-run differences of ≤15 games should be read as trends.
- All proxy arms except `m5_plain` use the external Model-A pointer; `m5_plain/combo_top` (161) is the fully self-contained
  configuration.
- `goal_revealed` / `subgoal` rely on ALFWorld goal parsing; `n_new`, `no_info`, `nothing_happens`, the future and audit signals
  are environment-generic given an entity tokenizer.

## 7. Pre-registered multi-round protocol (chains; `scripts/self_evolve_round.sh`, `scripts/run_rounds_plan.sh`)
Three chains start from the same SFT student and follow the identical per-round procedure; the only difference is the action filter.
1. **Collect**: the chain's current student plays the same 1,500 train games (T = 1.0, budget 50, the original b50 protocol).
2. **Estimator refresh**: the M5 progress estimator is re-trained on this student's own wins (continuing from the previous
   round's estimator; 4,000 states, lr 1e-4) — no external component.
3. **Pointer**: M5 rule (first step whose next state is predicted ≥1 step farther from success) on the failures of the first
   1,000 collected games.
4. **Candidates**: the current student's own top-2 admissible commands at the pointer, plain prompt (no hindsight), a⁻ excluded.
5. **Filter** (the manipulated factor): `env` = one-step env evidence, 86 states with the highest `env_score` (no rollout);
   `rollout` = student continuation ≤20 steps, keep winners (verified KL); `placebo` = 86 random pointer states, uniform random
   admissible action ≠ a⁻.
6. **Train**: KL on the targets, lr 1e-5, 1 epoch, LoRA continued from the chain's previous adapter. 7. **Eval**: the fixed 274
   held-out games (never used for training), greedy; paired McNemar vs the chain's previous round and vs the placebo chain at the
   same round. Logged per round: train-game win rate, #failures, #targets (rollout yield), flip rate vs previous adapter.
Fixed in advance: K = 86 for env/placebo; delta = 1.0; seeds (collection seed 0, placebo RNG = round index); 3 rounds planned;
a chain "works" if it improves over its own previous round and over the placebo chain at the same round.
Round 1: env chain = `m5_plain/env_top` (M5 pointer, plain candidates, env_top86); rollout and placebo chains built on the same
m5_plain states/candidates. Outputs under `reports/alfworld/self_evolve3b_rounds/<chain>/r<k>/`, log `.../queue.log`.

## 8. Replications (Oct 2)
- Training-seed replications on the SAME targets: env_top 172 → **169** (74/95); combo_top 164 → **164** (72/92). Training noise ≈ ±3 games;
  the proxy-selection results are stable. (placebo seed-1 pending.)
- Compression-SFT, different 200-episode samples: keep1 165 / 136 / 152, pruned 146 / 160, raw 139 (raw_s1 pending). Sample-to-sample
  swing ≈ ±15; keep1 vs pruned not distinguishable; compression ≈ +15 over the student on average. The Oct-1 "keep one detour" claim is withdrawn.

## 9. Step choice under the best filter (Oct 2 evening; plain candidates, env_top86, no rollout, no hindsight)
| pointer | 274 games | note |
|---|---|---|
| Model A (7B, external) | 172 / 169 (two training seeds) | |
| M5 (self-trained progress estimator) | **167** | fully self-contained configuration |
| random step in the first 3 | **168** | no attributor at all |
| 3B Who&When attributor (89% of pointers at step 0) | **168** | |
Within the "early, before the detour" window the pointer does not matter once the action filter is good. Other additions:
placebo training-seed replication 150 (seed 0: 141) → placebo mean ≈146, so single-run differences ≤10 games are noise;
combo_filter uncapped (137 targets) 159 ≈ combo_top86 164/164 (1.6× the targets at the same quality; no gain from the extra ones).

Placebo spread: 141 (Model-A states), 150 (same, training seed 1), **118** (M5 states, 75% at step 0; random actions at opening
states damage the opening policy). Chain env R1 (m5_plain/env_top) 167 vs placebo on the same M5 states 118: the gain is action content.

## 10. Literal on-policy distillation control (Oct 2)
KL to the hindsight teacher's FULL soft distribution over admissible commands, no filter, no verification, same lr/epochs:
on the 86 rollout-verified states **120/274** (45/75) — below the student (136) and below the random-action placebo (141/150) on the
same states. Distilling the hindsight self-teacher as-is is harmful; the gain in this setting comes entirely from selecting the action
with evidence the student can also access (one-step env outcome, logged future). hind_soft319 (all 319 pointer states) **93/274** (37/56): more unfiltered hindsight distillation, worse.

## 11. Multi-round chains (protocol §7) — running log
| chain | R1 | R2 | R3 |
|---|---|---|---|
| env (M5 pointer refreshed on own wins, plain candidates, env_top86, no rollout, no external component) | 167 (72/95) | **178 (78/100)**; train-game wins 993/1500 = 66% (student 59%); 334 failures in the first 1000 games; retention of R1 wins 150/167 | pending |
| rollout (same, rollout verification) | 161 (69/92; 134 verified) | pending (resumes on the next hold) | pending |
| placebo (same states protocol, random actions) | 118 (54/64) | pending | pending |
env R2 vs R1 28/17 p=0.14; vs student 57/15 p=6.5e-7; vs placebo R1 68/8; vs external-teacher R1 (169) 31/22 p=0.27; vs the
external-teacher M5 R2 chain (188 on these 274) 32/42 p=0.3.
