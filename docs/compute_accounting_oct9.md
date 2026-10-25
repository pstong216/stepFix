# Compute accounting: ALFWorld 774-game comparison at matched environment budgets

Snapshot: 2026-10-09, about 03:50 CDT. This is a read-only analysis of logs, SLURM job-step records (sacct) and output files.
§7 gives the source of every number. Estimates are marked *est.*

## 0. Conventions

- **Budget points.** R1 / R2 / R3 are cumulative training episodes:
  - Ours and RFT: 1,323 / 2,823 / 4,323.
  - SERL and GiGPO: steps 10 / 22 / 34 = 1,280 / 2,816 / 4,352.
  - HINT-SD: ckpt1-3 = 1,328 / 2,824 / 4,324.
- **Common start.** All five methods start from the same SFT student (`student3b_low100`, Qwen2.5-3B-Instruct + LoRA). Its cost (343 s of SFT) is shared and excluded.
- **Evals are excluded for every method.** Every checkpoint is scored with the same `eval774` harness on the shared hold GPUs. One 774-game eval costs 1.5-2.2 GPU process-hours over completed attempts (0.6-1.4 h fair-share); §3 lists ours for reference.
- **GPU-hours are reported two ways:**
  - **proc-wall** is the sum, over every GPU process, of its wall-clock (sacct step Start to End).
    - For SERL and GiGPO, each of which ran alone on one dedicated GPU, this equals the sum of `timing_s/step`.
    - That sum matches their checkpoint-directory timestamps to the second. It excludes 3-6 min of one-time startup.
  - **fair-share** charges each process ∫ 1/n(t) dt, where n(t) is the number of our python processes on the same 1-GPU hold job at time t.
    - Every script pins `cuda:0`, and the scheduler helper `pick` probes GPU 0, so all of a job's processes share that one GPU.
    - Fair-share adds up to each GPU's busy time, so it removes the double counting that GPU sharing causes. For dedicated runs it equals proc-wall.
    - Exclusive GPU time itself was **not logged** (there is no GPU-utilization accounting), so fair-share is an estimate.
- **Hardware is not uniform.**
  - SERL ran on a dedicated L40S (agd06).
  - GiGPO ran on a dedicated A100-SXM4-40GB (agb02).
  - Ours, RFT and HINT-SD ran on shared hold GPUs: A100, A40 or L40S, picked per process by free memory, with up to 3-11 co-resident processes.
  - The GPU-hours are therefore the cost of each run as it was executed, not hardware-normalised FLOPs.
- **Attempts.** Only completed attempts are counted. Crash and OOM restarts are listed separately in §3.

## 1. Summary: GPU-hours to reach each budget (evals excluded)

| Method | R1 | R2 | R3 | 774 score (R1 / R2 / R3) |
|---|---|---|---|---|
| **Ours** (proc-wall / fair-share) | 3.38 / 3.28 | 8.01 / 5.35 | 11.09 / 7.23 | 537 / 561 / 591 |
| SERL (dedicated L40S) | 4.23 | 8.84 | *est.* 13.4 (13.2-13.6) | 447 / 532 / pending |
| GiGPO (dedicated A100-40GB) | 3.34 | *est.* 6.8 (6.8-7.0) | *est.* 10.3 (10.3-10.8) | 475 / pending (514 at step 14) / pending |
| HINT-SD online (proc-wall / fair-share) | 2.40 / 1.59 | 4.90 / 3.66 | 7.78 / 4.46 | 367 / 341 / 344 |
| RFT (proc-wall / fair-share) | 4.75 / 4.46 | 9.39 / 6.56 | running (R3 collection started 03:29) | 431 / 428 / running |

Our one-step evidence runs on CPU, so it is reported separately: 0.50 / 0.99 / 1.18 CPU process-hours cumulative.
SERL's R3 and GiGPO's R2/R3 values extrapolate the observed time per step:
- Central value: the mean over steps after 10.
- Range: from the mean over all logged steps and the mean over the last 8 steps.
- At the snapshot SERL had finished 24/34 steps (9.58 h) and GiGPO 16/34 (5.08 h).

## 2. Full table (cumulative to each budget point)

### 2a. Environment interaction and GPU time

| Method | Point | Episodes | Policy env steps | Verification env steps | GPU-h proc-wall | GPU-h fair-share |
|---|---|---|---|---|---|---|
| Ours | R1 | 1,323 | 38,053 | 820 candidates (3,088 env.step incl. prefix replay) | 3.38 | 3.28 |
| Ours | R2 | 2,823 | 72,377 | 1,362 (7,668) | 8.01 | 5.35 |
| Ours | R3 | 4,323 | 103,661 | 1,704 (10,120) | 11.09 | 7.23 |
| SERL | s10 | 1,280 | 39,966 | 0 | 4.23 | 4.23 |
| SERL | s22 | 2,816 | 80,125 | 0 | 8.84 | 8.84 |
| SERL | s34 | 4,352 | *est.* 120.5k | 0 | *est.* 13.4 | *est.* 13.4 |
| GiGPO | s10 | 1,280 | 38,988 | 0 | 3.34 | 3.34 |
| GiGPO | s22 | 2,816 | *est.* 76.7k | 0 | *est.* 6.8 | *est.* 6.8 |
| GiGPO | s34 | 4,352 | *est.* 114.4k | 0 | *est.* 10.3 | *est.* 10.3 |
| HINT-SD | c1 | 1,328 | 39,647 | 0 | 2.40 (own log: 2.35) | 1.59 |
| HINT-SD | c2 | 2,824 | 89,109 | 0 | 4.90 (4.76) | 3.66 |
| HINT-SD | c3 | 4,324 | 136,769 | 0 | 7.78 (7.60) | 4.46 |
| RFT | R1 | 1,323 | 38,053 (same collection as ours) | 0 | 4.75 | 4.46 |
| RFT | R2 | 2,823 | 78,420 | 0 | 9.39 | 6.56 |

### 2b. Training compute and generated tokens

| Method | Point | Backprop samples | Backprop tokens (fwd+bwd) | Loss tokens | Extra forward-only passes in training | Generated tokens |
|---|---|---|---|---|---|---|
| Ours | R1 | 182 rows (82 KL corrections + 100 SFT replay) | 1.42M | 13.4k | none | 28.1k aux (Model A 9.8k + teacher 18.3k); policy 0 |
| Ours | R2 | 336 (136 + 200) | 2.74M | 24.4k | none | 47.7k |
| Ours | R3 | 486 (186 + 300) | 3.47M | 31.8k | none | 62.1k |
| SERL | s10 | 39,966 turn-samples | 25.54M | 0.325M | old-log-prob over the same tokens; teacher (a student copy re-synced every 10 steps) over a re-prompted hindsight copy of every sample (tokens not logged) | 0.325M policy action tokens |
| SERL | s22 | 80,125 | 52.85M | 0.650M | same | 0.650M |
| SERL | s34 | *est.* 120.5k | *est.* 80.0M | *est.* 0.98M | same | *est.* 0.98M |
| GiGPO | s10 | 38,988 | 25.34M | 0.321M | old-log-prob + reference (KL) over the same tokens | 0.321M |
| GiGPO | s22 | *est.* 76.7k | *est.* 52.5M | *est.* 0.63M | same | *est.* 0.63M |
| GiGPO | s34 | *est.* 114.4k | *est.* 79.6M | *est.* 0.95M | same | *est.* 0.95M |
| HINT-SD | c1 | 1,563 target steps (166 optimizer steps) | 0.97M | 10.3k | EMA-teacher forward (prompt + feedback + action) 1.02M | 47.6k analyzer; policy 0 |
| HINT-SD | c2 | 3,674 (353) | 2.23M | 24.9k | 2.34M | 112.7k |
| HINT-SD | c3 | 5,701 (541) | 3.44M | 38.6k | 3.61M | 174.4k |
| RFT | R1 | 10,853 SFT rows | 7.32M | 88.4k | none | 0 |
| RFT | R2 | 23,320 | 15.55M | 188.9k | none | 0 |

Notes:
- **Our row counts understate our compute.** A KL correction row backprops prompt + command for *every* admissible command, in chunks of 8 padded to the chunk maximum.
  - That is 29.1 / 35.1 / 24.6 sequences per row in R1 / R2 / R3.
  - KL rows account for 1.37M / 1.28M / 0.69M tokens per round; the 100 SFT replay rows add only 44.5k / 45.7k / 45.0k.
  - Tokens are therefore the comparable unit, not rows.
- **SERL and GiGPO train per turn.** One "sample" is one turn: prompt (10-step history + admissible list) plus the action.
  - Samples per step = 128 × `episode/length/mean`. This agrees with `perf/total_num_tokens` ÷ (mean prompt + mean response length) to within 0.4%.
  - The loss is applied to the action tokens only.
- **Policy forward tokens during rollouts.**
  - Ours, RFT and HINT-SD act by scoring every admissible command, so they generate no action tokens. Per env step they spend prompt + all command tails:
    - Ours: 31.9M / 64.5M / 93.6M.
    - RFT: 31.9M / 65.8M.
    - HINT-SD: 34.4M / 76.4M / 117.2M (my recount with the same formula; its own counter logged 135.7M at c3, probably including batch padding).
  - SERL and GiGPO roll out 25.5M (s10) and 52.9M (s22) prompt+response tokens. These go through vLLM generation and then the old-log-prob forward.
- **Not counted:** Model A's prefill (the whole failed trajectory) and our teacher's two scoring passes over the admissible commands (with and without hindsight).

## 3. Our GPU time by stage and round (hours, proc-wall / fair-share)

| Stage | R1 (reused `selfattr_a3bfinal/r1` + SFT collection) | R2 | R3 |
|---|---|---|---|
| Collection (policy rollouts, T = 1) | 2.92 / 2.92: 1 process, alone on an A40, Sep 20, 1,323 games (7.9 s/episode) | 4.02 / 1.75: 4 shards × 375 games | 2.68 / 1.71: 4 shards |
| Model A attribution (frozen 3B) | 0.03 / 0.03 (544 failures) | 0.11 / 0.05 (428) | 0.04 / 0.04 (375) |
| Hindsight self-teacher | 0.25 / 0.25 (410 states, 1 process) | 0.38 / 0.15 (271 states, 3 shards) | 0.27 / 0.08 (171 states, 3 shards) |
| One-step evidence (CPU, no GPU) | 0.50 CPU-h (1 process, 820 candidates) | 0.49 CPU-h (12 processes, 542) | 0.19 CPU-h (12 processes, 342) |
| Training (`alfworld_opd_plus.py train`) | 0.18 / 0.08 (trainer elapsed 629 s) | 0.12 / 0.12 (429 s) | 0.10 / 0.05 (319 s) |
| **Round GPU total** | **3.38 / 3.28** | **4.64 / 2.07** | **3.08 / 1.88** |
| Failed or unproductive attempts (not in totals) | +0.02 (cancelled Model A); +0.35 (the R1 collector wrote nothing after game 1,323 at 09:44:45 and was cancelled at 10:05) | 0 | +0.41 / +0.21 (21 collector and 15 teacher crash-restarts) |
| 774 eval (excluded) | 1.54 / 0.84 | 1.54 / 0.83 | 1.62 / 0.62 |

- Collection is 87% of our GPU time: 9.62 of 11.09 proc-wall hours, 6.38 of 7.23 fair-share.
- The method-specific GPU stages (Model A + teacher + training) total 1.47 proc-wall / 0.85 fair-share hours over three rounds.
- Notes on R3:
  - Its teacher covered 171 of the 249 attributed states (the pool problem found by `r3_poolcheck`); 171 is what the 591 run actually used.
  - The later `r3_fixpool` retrain (574) is not part of this chain and is not counted.
  - The R3 driver sat idle from 20:39 to 22:03. Idle time costs nothing here because only process time is counted.

**Baseline stage breakdowns**
- **SERL** to s10 / s22:
  - gen (vLLM rollout, including env stepping): 0.79 / 1.73 h
  - old-log-prob: 0.62 / 1.29 h
  - update_actor (actor fwd+bwd plus the teacher forward on re-prompted copies): 2.78 / 5.74 h
- **GiGPO** to s10: gen 0.78, old-log-prob 0.45, ref 0.34, update_actor 1.75 h.
- **HINT-SD** to c1 / c2 / c3, from its own per-step `sec` fields:
  - rollout: 1.58 / 3.24 / 5.42 h, of which env-worker time is 0.35 / 0.65 / 0.81 h
  - analyzer: 0.40 / 0.80 / 1.25 h
  - loss: 0.36 / 0.72 / 0.93 h
- **RFT:**
  - R1 = the same 1,323-game collection (2.92) + training 1.83 / 1.54 (trainer elapsed 6,525 s).
  - R2 = collection 3.54 / 1.81 + training 1.10 / 0.28 (3,955 s, on a heavily shared L40S).

## 4. Verification cost: one-step evidence vs rollout verification (3B, K = 20%)

| | R1 | R2 | R3 | 774 (R1 / R2 / R3) |
|---|---|---|---|---|
| One-step (ours): candidates / env.step calls incl. prefix replay / compute | 820 / 3,088 / 0.50 CPU-h | 542 / 4,580 / 0.49 CPU-h | 342 / 2,452 / 0.19 CPU-h | 537 / 561 / 591 |
| Rollout verification (ablation `selfattr_rollsub_a3b20`, verify-only): env steps / GPU-h proc-wall (fair-share) | 32,559 / 3.42 (1.47) | 26,555 / 2.68 (1.31) | 29,959 logged, *est.* ≈33.1k in total / 2.58 (2.48) | 483 / 390 / 435 |

- How one-step executes a candidate: one env reset, a replay of the logged prefix (t steps), then the candidate.
  - Only the candidate step yields new information.
  - Across the three rounds that is 1,704 new env steps (1.6% of our 103,661 collection steps), or 10,120 env.step calls counting the replays (9.8%).
- One rollout-verification R3 shard restarted after 36 of its 94 targets, and that first chunk's env steps were never logged. I pro-rated it from the shard's logged chunk.

## 5. Takeaways

- **Backprop compute is where ours is cheap.**
  - Through R3, ours backprops 486 rows = 3.47M tokens.
  - SERL and GiGPO: about 115-120k turn-samples ≈ 80M tokens (*est.*). That is about 23× more tokens and about 240× more samples.
  - RFT at R2: 15.5M tokens vs our 2.74M (5.7×).
  - Online HINT-SD: about the same tokens as ours (3.44M vs 3.47M), spread over 12× more rows (5,701).
- **Total GPU-hours are the same order for all methods**, because every method's cost is dominated by rollouts. To R3:
  - ours: 11.1 proc-wall / 7.2 fair-share
  - GiGPO: ≈10.3 (*est.*)
  - SERL: ≈13.4 (*est.*)
  - HINT-SD: 7.8 / 4.5
  - At R2, RFT is 9.4 / 6.6 vs ours 8.0 / 5.3.
  - Ours is not cheaper than GiGPO in GPU-hours, and the paper should not claim it is.
- **Our GPU time is mostly our rollout implementation.**
  - 87% of it is collection, done with an unbatched HF-transformers scorer that runs one env per process.
  - verl-agent's batched vLLM rollout needs 0.79 h per 1,280 episodes.
  - Projection, not measured: a collector that matched that rate would bring us to about 2.7 h of collection + 1.5 h of method stages ≈ 4 h to R3.
  - The method-specific GPU stages (Model A + teacher + training) are 1.47 proc-wall / 0.85 fair-share hours over all three rounds.
- **Verification is nearly free.**
  - One-step evidence: 1,704 extra env steps (1.6%; 9.8% counting deterministic replays), about 1.2 CPU process-hours, and no GPU.
  - The rollout-verification ablation spent about 92k env steps (+89% env interaction) and 8.7 GPU process-hours over three rounds, and still scored lower (483 / 390 / 435 vs 537 / 561 / 591).
- **Generated tokens.**
  - Ours, RFT and HINT-SD generate no policy tokens (admissible-command scoring).
  - Our only generated text is Model A's attribution plus the teacher's analysis: 62k tokens through R3.
  - For comparison: about 0.95-0.98M generated action tokens for SERL/GiGPO (*est.* at 4.35k episodes), and 174k analyzer tokens for HINT-SD.
- **Episodes are matched, env steps are not.** Ours uses the fewest env steps at every point (tied with RFT at R1, which reuses the same collection), because its improving student finishes episodes sooner: 20.9-28.8 steps per episode, against 29.9-31.8 for HINT-SD and about 26-31 for SERL/GiGPO.
  - At R3 ours is 103.7k steps, or 113.8k including the one-step replays.
  - SERL ≈120.5k (*est.*), GiGPO ≈114.4k (*est.*), HINT-SD 136.8k.
- **Caveats for the paper table.**
  1. GPU types differ (dedicated L40S for SERL, dedicated A100 for GiGPO, a shared A100/A40/L40S mix for the rest).
  2. HF and vLLM implementations differ in efficiency.
  3. SERL s34, GiGPO s22/s34 and RFT R3 are still running; their compute numbers are extrapolations.
  4. Our frozen Model A attributor had a one-time training cost of 3 h 51 min (Sep 9, Who&When data; `reports/attributor_no_observations_train.log`). It is not ALFWorld-specific and is not included above.

## 6. What could not be determined

- **True exclusive GPU time** of processes on shared GPUs. No GPU-utilization accounting exists, so fair-share is an estimate. It could be too low for processes that are latency-bound but GPU-light, or too high under contention.
- **Token count of SERL's teacher pass.** Re-prompt lengths are not logged; `serl/reprompt_sample_fraction` = 1.0, so every sample gets that extra forward.
- **Optimizer-update counts for SERL and GiGPO.** Not logged.
- **HINT-SD's own forward-token counter per checkpoint.** Only its final total (135.7M) is logged; the per-checkpoint values in §2b are my recount.
- **Final scores and compute** for SERL s34, GiGPO s22/s34 and RFT R3 (still running).

## 7. Sources

**Our stage times, and the RFT and HINT-SD proc-wall / fair-share numbers**
- `sacct -u pan00389 -S 2026-10-04 -p --format=JobID,JobName,Start,End,ElapsedRaw,NodeList,State,AllocTRES,SubmitLine`.
- Each `srun --overlap` step's SubmitLine names the script and its `--out` round directory, which maps the step to its stage and round.
- 774 eval steps are matched by `--tag h500_<TAG>`.

**R1 collection**
- `sacct -L` step agate 1436314.17 (agc04, A40; it was the only step on that GPU during the collection).
- `reports/alfworld/chain_student3b_low100.log`.
- The mtime of `reports/alfworld/student3b_low100_train_b50.jsonl`: 06:49:38 to 09:44:45.

**Trainer elapsed**
- `reports/alfworld/self_attr_rounds/a3bfinal_k20/r{1,2,3}/model/train_log.jsonl`.
- `rft_full/r{1,2}/model/train_log.jsonl`.

**Episodes and env steps**
- Ours and RFT: `n_steps` summed over `traj.jsonl` (R2/R3) and `student3b_low100_train_b50.jsonl` (R1).
- HINT-SD: `a3b_hintsd_online/rollouts.jsonl`.
- SERL/GiGPO: Σ 128 × `episode/length/mean`.

**One-step evidence**
- Rows of `selfattr_a3bfinal/r1/proxy/onestep.jsonl` (820), `a3bfinal_k20/r2/proxy/onestep.jsonl` (542) and `a3bfinal_k20/r3/proxy/onestep.jsonl` (342).
- env.step calls = Σ(t+1), following the replay in `scripts/proxy_eval_se3b.py:cmd_onestep`.
- CPU time runs from the `teacher.log` mtime to the `onestep*.log` mtimes.

**Rollout verification**
- The "DONE distill" `env_steps` / `elapsed_s` lines in `selfattr_rollsub_a3b/r1/rollout_s*.log` (reused for a3b20 R1) and `selfattr_rollsub_a3b20/r{2,3}/rollout_s*.log`.
- Plus the sacct `alfworld_attr_opd.py distill` steps.

**SERL / GiGPO**
- `/scratch.global/pan00389/error-attribution/serl_work/outputs/{full_1gpu,gigpo_1gpu}/train.log`: per-step `timing_s/*`, `perf/total_num_tokens`, `episode/length/mean`, `response_length/mean`.
- The `checkpoints/global_step_*` mtimes, which confirm that Σ `timing_s/step` equals wall time.
- `run_serl_igpu.sh` / `run_gigpo_a100qw.sh` (16 tasks × 8 rollouts, `ppo_epochs` 1, validation off).

**HINT-SD**
- `a3b_hintsd_online/summary.json`.
- `train_log.jsonl`: per-step `sec`, `gen_tokens`, `n_targets`, `n_target_tokens`.
- ckpt adapter mtimes: 19:43:48, 22:17:02, 01:09:48.

**Backprop and generated tokens**
- Recounted on CPU with the Qwen2.5-3B tokenizer and the exact policy prompt (`history_at` + `render_prompt(10)` + chat template).
- Training files: `targets*.jsonl`, `replay.jsonl`, `rft_full/r*/train.jsonl`, and HINT-SD's `targets.jsonl` + `rollouts.jsonl`.
- Generated text: `modelA_pred.jsonl` (field `raw`) and `self/targets.jsonl` (field `analysis`).
- Cross-check: the recount reproduces HINT-SD's logged 38,583 loss tokens and 5,701 rows exactly.

**774 scores**
- `reports/alfworld/self_attr_rounds/queue.log` BASELINE lines: a3bk20_r1/r2/r3, serl_s10/s22, gigpo_s10/s14, hsdon_c1-3, rftfull_r1/r2.
- RFT R2 = 428 was logged at 03:29:11 and GiGPO s14 = 514 at 03:35:57.

**Analysis scripts**
- `parse_verl_steps_ca9.py`, `sacct_stages_ca9.py`, `count_tokens_ca9.py`, `hsd_cum_ca9.py`, in this session's scratchpad (ephemeral).
- Rerunning the sacct command above plus these scripts regenerates every number.
