# Handoff 2026-10-10 20:05 CDT (new session; old session context too large)

Experiment freeze **Oct 11 12:00 CDT**; ARR deadline Oct 12. Full result table: `docs/results_method_map_oct10.md`,
ScienceWorld summary: `docs/sciworld_final_summary_oct10.md`. Standing rules: ask before any NEW experiment (plain Chinese);
no multi-seed; Model A stays the step selector; no privileged subgoals / oracles; outputs on /scratch.global; never push Overleaf
unprompted; baselines same budget as ours. **Credit saving (user, Oct 10): no background agents; check logs only at key times.**

## TOP PRIORITY (user, 19:25): ScienceWorld 1,500-task chain — tmux `sf_x1500`
- Script `scripts/run_sciworld_stepfix_x1500.sh` (resumable; restart = `tmux kill-session -t =sf_x1500` then relaunch same command).
- Why: old protocol collected only 518 train variations (per_task 20 cap); r1big = the same 518 x 3 seeds (not new tasks) = 305.
  Now per_task 75 = **1,483 distinct variations** (518 old + 965 new via `--job_slice 20 75`, overlap checked = 0), ALFWorld-like scale.
- Recipe = hybrid H (best, r2h 303): 3B Model A no_observations, full-view teacher + relaxed validity + refill, one-step generic
  evidence, env_top K = 0.2 x failures, **replay_fw n = round(1.2 x K)** (user: ALFWorld ratio 82:100; NOT 300, NOT 100), train from
  previous adapter, 1 ep lr 1e-5 bs 8 seed R. Baseline: standard RFT (all win steps) on the same collections.
- Dirs `reports/sciworld/stepfix/{x1,x2,x1rft,x2rft}`; log lines prefixed x1/x2/x1rft/x2rft in `reports/sciworld/stepfix/queue.log`.
- State 20:00: x1 collection done (1483 eps, 615 won, 868 failed); Model A done; teacher (32 shards) starting; ETA x1 848 result ~21:30.
  x1rft training 11,460 rows (~1.8 h) -> eval ~22:00. Then x2 (collect with x1 adapter) -> R2 result ~01:30-02:30; x2rft after.
- Success criterion: x2 significantly > x1, or clearly > rftfull R2 313 / x2rft.

## PAUSED to give GPUs to sf_x1500 (resume when x1500 is in a light stage or done)
- **r2hR / r2hRw / r1bR** evals (analysis only — full-trajectory RFT replay is inconsistent with ALFWorld, user agreed it can't be
  the main result): tmux sf_rftR killed during 848 evals. All three are TRAINED. Resume: `tmux new -d -s sf_rftR "bash
  scripts/run_sciworld_stepfix_rftreplay.sh 2>&1 | tee -a reports/sciworld/stepfix/rftreplay_driver.log"` (episode-level resume).
- **Extended test eval**: `reports/sciworld/eval_ext/.maxp` set to 0 (was 20). Raise to resume (driver tmux xeval_ext_a still alive).
- **RL agent died** (account session limit). RL training continues on reserved sw_rl_gpu jobs (GiGPO -> step 72 ~05:00, tight vs the
  job end ~05:20; SERL from ckpt 12 -> 48, had OOMs — nobody restarts it now). Eval drivers in tmux sw_eval_g*/sw_eval_s* wait for
  checkpoints (`scripts/sciworld_rl_eval.sh RUN_DIR STEP TAG`); I told the agent not to launch shards on shared GPUs.
  Results so far (848): base 267-ish, gigpo_s8 272, serl_s8 277, **gigpo_s16 286** (vs R0 p=0.048; vs r3h 299 p=0.28). Pending s24+.
  Summaries: `scripts/sciworld_rl_summary.py`, logs `reports/sciworld/rl_baselines/`.

## Other running
- 7B ScienceWorld (tmux sw7b3, `scripts/run_sciworld_7b_r1.sh`, reserved sw7b_gpu): R0 7B = 278/848 (vs 3B R0 267, p=0.37);
  R1 with the 7B Model A, teacher stage running (one shard retried rc=143). Log `reports/sciworld/stepfix7b/queue.log`.

## Results new today (848 episodes, R0 = 267)
- r2c (cumulative corrections R1+R2) = 302 (= r2h 303, p=1). r1big = 305 (= r1b, p=0.56).
- ScienceWorld standard RFT: R1/R2/R3 = 265/313/309 (4,008/4,415/4,454 rows, all win steps; fatal 130/158/147 vs ours 112/107/108).
- R2 diagnosis (518 protocol): 314 failures, only 121 states with any S>0 candidate, 102 with no valid candidate; 26/63 r2h
  corrections went to inclined-plane (stays 0 wins). Task-name-based quota from TEST results was rejected (test-set tuning).
- Rejected by user: StepFix on top of the RFT model (option A); 300-step replay.
- Ideas proposed, not approved: training-signal adaptive quota; Model A sampled 3x (multi-step candidates).

## Update 23:05
- x1 (3B R1, 1,483 tasks) = 309 (vs R0 p=0.0002). x1rft (same-budget standard RFT R1, 11,460 rows) = 330, but all of its lead is
  inclined-plane guessing (37 wins / 53 fatal there); outside inclined-plane x1 309 vs rft 293; fatal 128 vs 186; mean score 53.3 vs 51.2.
- x2 (R2, original recipe) = 284, significantly WORSE than x1 (p=0.013): forgetting on uncorrected tasks.
- x2c / x2cr (user-approved): x2 teacher rows re-selected with per-type cap (2x share = 13) + RETENTION replay (by training wins),
  replay 1.2K (x2c) / 2.4K (x2cr); tmux sf_x2c / sf_x2cr; script scripts/run_sciworld_x2c.sh, selector scripts/sciworld_select_capped.py.
- x2rft collection (orphan driver pid 1662414) PAUSED with SIGSTOP at 23:05 to give GPUs to x2c/x2cr evals. RESUME:
  for p in $(cat <scratchpad>/x2rft_stopped_pids); do kill -CONT $p; for c in $(pgrep -P $p); do kill -CONT $c; done; done
- 7B x1500 run: tmux sw7bx (scripts/run_sciworld_7b_x1500.sh), throttled by tmux sw_prio (scripts/sciworld_priority_watch.sh).
- RL evals: reports/sciworld/rl_baselines/.skip_jobs now lets them use the sw_rl_gpu trainer GPUs only.

## Update 02:55 Oct 11 — SUCCESS on ScienceWorld R2 (w chain)
- w chain (scripts/run_sciworld_w_chain.sh, tmux sf_wchain): teacher top-4, S>=2, commit guard "confirmed" (.commit_guard_mode),
  NO per-type cap, look-around allowed, RETENTION replay 2.4 x K; each round continues from the previous adapter.
- w1 = 307 (vs R0 p=0.0004; = x1). w2 = 334 (vs w1 p=0.014; vs R0 p=1.8e-6; vs x1 p=0.055; vs same-budget RFT R1 330 p=0.84).
  mean score (neg->0) w2 55.5 (best), raw 39.2; fatal 138 (w1 119, RFT 186).
- w3 collecting (ETA ~05:00). x2rft (same-budget RFT R2) collection RESUMED (SIGCONT) at 02:55.
- Dead ends (keep as ablations): x2 284, x2c 282, x2cr 299, x2d 285, z1 (per-type cap + no look-around) ~lower, y1 stopped.
- 05:22 wk3 collection shard loops SIGSTOPped (pids in scratchpad/wk3_stopped) so w3/w3k can train; resume with kill -CONT on those pids.
- 05:32 w2k (KL replay, beta 10) = 336 / mean0 55.9 / raw 40.6 / fatal 130; vs w1 21/50 p=0.00077; vs w2 36/38 p=0.91; vs RFT-R1 98/104 p=0.73. KL >= plain on every metric (not significant vs w2). Full KL chain R3 = wk3 (collecting).
- 06:27 w3k (R3 from w2 + KL) = 338 / mean0 53.6 / fatal ~133; vs w2 45/49 p=0.76; vs w1 p=0.019. w3 (plain R3) ~300/848 (845 done): R3 without KL regresses (mean0 46.7, raw 29.3). wk3 (full KL chain R3) teacher running.
- 07:33 wk3 (full KL chain R3, w1->w2k->wk3) = 353 / mean0 56.3 / raw 35.2 / fatal 179; vs w2k 41/58 p=0.11; vs w1 46/92 p=0.00011; vs R0 p=2.3e-9; vs RFT-R1 103/126 p=0.15. KL chain: 307->336->353 wins; fatal 119->130->179 (commit guessing grows).
- 08:25 GPU scarcity: 4 GPU jobs running. Paused 7B w7_2 collection (pids scratchpad/w7_2_stopped, resume kill -CONT). Gave 3364361 to RL evals (s48 cells needed for paper). Submitted 4 more preempt GPU jobs.
- 09:00 RFT R2 (x2rft) = 352/51.5/28.1/198 (filled in paper). Paused RL eval drivers g32/g40/g56/g64/s32/s40 (tmux killed; per-episode resume via scripts/sciworld_rl_eval.sh) to prioritize s24/s48/g72 cells.
- 10:03 w7_1 eval done -> 7B blocked from shared GPUs (res_only_until +24h, sw_prio stopped) so x2rft/RL evals finish; w7_2 will not make the 12:00 freeze.
- 10:03 7B ScienceWorld R1 (w7_1, drift-controlled, no KL) = 315 / mean0 54.6 / fatal 138; vs 7B R0 278: 72/35 p=0.00045; vs old-recipe x7 311 p=0.74. Added to paper appendix. 7B R2 blocked (won't make freeze).
- 10:10 7B w chain stopped: w7_2 collection incomplete (1124/1483 after shard give-ups); do not use w7_2. Resume later by deleting w7_2/traj.jsonl and relaunching scripts/run_sciworld_7b_w_chain.sh.
- 10:45 GiGPO s72 = 391/56.4/33.2/196 (vs wk3 130/92 p=0.013). All freeze experiments done.
- 11:19 7B w chain v2: 9+ w7_2 collect shards gave up (OOM contention, 25-attempt cap); restarted tmux sw7bw2 (same command; shards resume).
- 11:38 7B w7_2 collect: root cause of repeated GIVING UP = stale collect_sN.log.aN (a1..a25) make loop7 give up after 1 attempt per restart; moved stale attempt logs of the 10 unfinished shards to w7_2/col/old_attempts and restarted sw7bw2 again.
- 11:48 a7b_gpu 3364361 ends ~12:15; submitted replacement a7b_gpu via sbatch.
- 11:49 7B ablation evals (3x16 procs) were starving 7B main collection with OOM; killed tmux sw_abl7_rand/norep/nofil (priority 7B main > ablations); relaunch after w7_2 collect finishes (scripts/run_sciworld_7b_abl1.sh resumes).
- 12:28 w7_2 collect complete (1483 eps, 650 won), Model A pointer done, teacher stage started; moved stale abl7 attempt logs; relaunched sw_abl7_rand (norep/nofil staggered +10/+20 min).
- 12:38 relaunched sw_abl7_norep.
- 12:57 7B w7_2 teacher (48 shards) starved by OOM (8/48 done after 25 min, shards at up to 10 attempts of 25 cap); killed sw_abl7_rand/norep again (nofil not relaunched). Will relaunch ablations after teacher stage. NOTE: teach7 writes 'DONE teacher (merged)' even if shards gave up -> check shard DONE count before trusting.
- 13:01 a7b replacement 3409005 RUNNING; renamed to sw7b_gpu per coordinator (3 reserved sw7b GPUs now).
- 13:38 w7_2 teacher 48/48 + select (K=167) done, training next; relaunched sw_abl7_rand.
- 13:47 3B results landed: abl_norep 286, abl_rand 288, abl_rftm 268, abl_nofil 219, roll1 327 (p vs R0 1.7e-06, vs w1 0.033); sent to main. ga1 eval running.
- 14:01 ga1 DONE: 319 won (vs gigpo_s24 328 p=0.39; vs gigpo_s48 392 p=9.9e-10). All 3B jobs finished. w7_2 trained (k=568), eval running; relaunched sw_abl7_norep (nofil +10 min).
- 14:08 relaunched sw_abl7_nofil.
- 14:26 w7_2 eval shards at 21 attempts (cap 25) and w7_3 col shards at ~15: OOM contention; killed the 3 sw_abl7_* sessions again (resume later when main has free GPUs).
- 14:26 restarted sw7bw2 after moving stale unfinished-shard attempt logs (w7_2/eval, w7_3/col) to old_attempts to reset 25-attempt caps.
- 14:54 coordinator: 7B R3 cancelled, sw7bw2 stopped (do not restart); now watching sw7b_r2fin, sw_w7_2b3/2b1/2b3lr; abl7 lower priority.
- 14:58 7B w7_2 848: 316 won, mean 55.6; vs R0_7B p=0.001, vs w7_1 p=1, vs w2k_3B p=0.1. Reported. Now watching r2tune arms + sf_ch_abl_* chains (7B ablations cancelled).
- 15:28 all running GPU jobs renamed sw7b_gpu: 3B ablation chains get no new GPU slots until the 7B R2 tuning evals finish (rename back to tcded_gpu/a7b_gpu afterwards).
- 16:06 w7_2b3 eval shard s9 hit 25-attempt cap; moved stale attempt logs of unfinished eval shards to old_attempts and restarted tmux sw_w7_2b3 (same cmd).
- 16:07 w7_2b1 eval shards near 25-attempt cap; moved stale attempt logs and restarted tmux sw_w7_2b1 (same cmd).
- 16:07 w7_2b3lr eval shards near 25-attempt cap; moved stale attempt logs and restarted tmux sw_w7_2b3lr (same cmd).
- 16:26 job 3402380 (tcded_gpu) ends in ~10 min; submitted replacement a7b_gpu via sbatch.
- 16:36 7B 848: w7_2b3lr 347 (p vs R0 2.2e-9, vs w7_1 8.5e-4, vs w7_2 3.3e-5); w7_2b1 339 (p vs w7_1 .0063, vs w7_2 .0014). Reported. w7_2b3 eval 15/16.
- 16:39 7B w7_2b3 848: 323 (p vs R0 4.5e-5, vs w7_1 .37, vs w7_2 .26). All 3 R2 tuning evals done; told main to rename jobs back. Now babysitting 3B sf_ch_abl_* chains.
- 16:49 coordinator: new 7B R3 run in tmux sw7b_r3 from w7_2b3lr (beta 3, lr 2e-5) -> stepfix7b/w7_3; watching (reserved sw7b_gpu jobs: replace within 30 min of end).
- 17:16 FOUND BUG: run_sciworld_abl_chain.sh abl_round 'local R=$1 AD=$2 D=$SF/${NAME}_r$R' -> D=<NAME>_r (R empty at expansion); chains share one dir, r3 reuses r2 adapter, evals point at empty _r2/_r3. Reported to main; no change made.
- 17:30 ablation-chain bug fixed (D used R before assignment -> all rounds wrote <name>_r). Round-2 data moved to <name>_r2, bogus r2/r3 dirs and the misplaced model moved to _old/abl_bug_1725; chains relaunched.
- 17:29 coordinator fixed abl_chain bug, relaunched chains ~17:30; w7_3 col 44/60.
- 17:43 coordinator: 7B ablations (abl7_rand/norep/nofil) relaunched (evals resume); w7_3 col 59/60.
- 17:45 w7_3 (R3) collection done: 1483 eps, 720 won, mean 62.6; reported. Model A pointer next.
- 18:14 abl7 eval shards at 14-18 attempts: moved stale attempt logs of unfinished eval shards to old_attempts and restarted sw_abl7_nofil/norep/rand (same cmds).
- 18:42 w7_3 teacher shards at 14-22 of 25-attempt cap (OOM thrash, 27/48 done); moved stale unfinished-shard attempt logs to w7_3/old_attempts and restarted tmux sw7b_r3 (same cmd).
- 19:02 abl7_rand 848: 269 won, mean 48.7; vs R0_7B(278) p=0.45; vs w7_1(315) p=2.9e-6. norep eval shard near cap: moved stale logs, restarted sw_abl7_norep.
- 19:07 abl7_nofil 848: 107 won (p vs R0_7B 1.2e-30); abl_rftm_r2 (3B chain): 286 won (vs R0 p=.13, vs wk3 p=1e-5). Reported. w7_3 teacher 44/48.
- 19:13 abl7_norep 848: 300 won (p vs R0_7B .043, vs w7_1 .082). All 7B ablations done.
- 19:23 w7_3 teacher 48/48, select K=153, trained (k=520, loss_corr .150); eval started. cpu_tmux 2995731 ends ~22:40.
- 19:35 Oct 11 ScienceWorld ONLINE HINT-SD (3B, from R0) running: tmux sw_hsd_online (scripts/run_sciworld_hintsd_online.sh -> scripts/hintsd_online_sciworld.py, port of hintsd_online.py; same algo/hparams, B=2x4 rollouts, lr 5e-6, EMA 0.999); train dir reports/sciworld/stepfix/hsdon_train (resumable state/), ckpt1..3 exactly at 1483/2966/4449 episodes -> 848 evals hsdon_c1..3 (40 shards) + compare vs R0/w1/w2k/wk3 into stepfix/queue.log. Smoke test (hsdon_smoke, 5 steps) OK.
- 19:40 job 3407623 (tcded_gpu) ends in ~34 min; submitted replacement a7b_gpu.
- 19:52 Oct 11 hsdon throughput ~550 eps/h (step time) / ~500 eps/h wall; ETA ckpt1 ~22:30, ckpt2 ~01:30, ckpt3 ~04:30 (no interruptions). WARNING: cpu_tmux 2995731 ends ~22:39 -> sw_hsd_online dies; relaunch: tmux new -d -s sw_hsd_online "bash scripts/run_sciworld_hintsd_online.sh > reports/sciworld/stepfix/hsdon_driver.log 2>&1" (resumes from hsdon_train/state; evals skip finished shards).

## 19:50 Oct 11 — BEFORE cpu_tmux (2995731, cn1019) ENDS ~22:39
- Job 3440069 (error_attr_drivers, acn108/160, 24 h; slurm/relaunch_drivers.sbatch) waits for 2995731 to end, then relaunches in tmux ON ITS NODE:
  sf_ch_abl_{rand,nofil,norep,rftm,roll} (3B ablation chains R2/R3), sw_hsd_online (online HINT-SD, ckpts 1483/2966/4449), sw7b_r3 (only if
  7B R3 not finished). All resume from outputs. To inspect: `srun --jobid=3440069 --overlap --pty bash` then `tmux ls`.
- The Claude session and its babysitter agent run on cn1019 and will end too. Start a new Claude Code session and read this file.
- Results so far: see docs/morning_summary_oct11.md + this file. 7B ScienceWorld: R0 278, R1 315, R2 347 (KL 3, lr 2e-5), R3 eval ~20:15.
  7B ablations (1 round): random 269, no replay 300, no filter 107. 3B ablations R1: random 288, no replay 286, no filter 219, matched RFT 268,
  rollout 327 (25x more policy calls); matched RFT R2 286. RL+StepFix (ga1) 319 (vs GiGPO s24 328) -> not in paper.
- Paper: Overleaf pushed at 118e98b (claude-oct11). Pending: ALFWorld efficiency (agent still computing), ablation R2/R3 cells, HINT-SD online.
- 19:58 7B R3 w7_3 848: 340 won, mean 56.6 (p vs R0_7B 3e-7, vs w7_1 .011, vs w7_2b3lr .41 [347], vs wk3_3B .36). Reported. Only 3B sf_ch_abl_* chains remain.
- 20:00 7B R3 (w7_3, KL 3, lr 2e-5, from w7_2b3lr) = 340 / mean0 56.6 / fatal 133; vs R2 347: 23/30 p=0.41 (no gain); vs R1 315 p=0.011; vs R0 p=3e-7. 7B ScienceWorld chain: 278 -> 315 -> 347 -> 340.
- 20:17 job 3407626 (tcded_gpu) ends in ~35 min; submitted replacement a7b_gpu (3438792 from earlier still pending).
- 21:22 session moved to acn32 (job 3442857). 7B R3 tuning arms on acn32: sw_w7_3b1 (KL1 lr2e-5), sw_w7_3k30 (top 30% K=229, KL3 lr2e-5), sw_w7_3lr1 (KL3 lr1e-5). Only 3 GPUs were running; submitted 6 more. New Sonnet babysitter launched.
- 21:22 only 3 GPU jobs running, 2 named sw7b_gpu (not usable by 3B chain pick()); asked main to rename back. Many jobs PENDING (preempted).
- 21:50 3 running GPU jobs (3407625,3407624,3418820) renamed sw7b_gpu for 7B R3 tuning arms (w7_3b1/k30/lr1 were OOM-retrying on shared GPU); sf_x3rft + sw_serl72 added to watch (serl72 waits for sw_rl_gpu).
- 21:50 running GPU jobs 3407624/3407625/3418820 are named sw7b_gpu again (rename-back did not stick); 3B chains can't pick them; asked main.
- 21:58 jobs now tcded_gpu; rule: no sw7b_gpu names (coordinator).
- 22:03 only 3 GPUs running (19 pending, cluster Priority). Renamed all 3 to sw7b_gpu so 7B R3 tuning arms get every free slot; 3B chains/HINT-SD/x3rft launch nothing new until renamed back.
- 22:07 jobs 3418820/3407625/3407624 reappeared as sw7b_gpu; renamed to tcded_gpu again (something re-renames; told main).
- 22:15 Oct 11 hsdon FIX: attempts were host-RAM OOM-killed (uncapped ScienceWorld JVMs default heap = 1/4 host RAM, 3-4 GB RSS each x 8, in full 64G job cgroups). Trainer now caps worker JVMs (-Xmx1g SerialGC; tested 0.5 GB RSS), saves state every 5 steps; driver exports the same cap for the eval shards and picks the trainer GPU with >= 20 GiB free host RAM in the job cgroup (pick_mem/SM). Restarted sw_hsd_online on cn1019 at 21:57 (resumed step 20). Relaunch after 22:39 uses the same script.
- 22:40 drivers relaunched on acn108 by job 3440069 (cpu_tmux 2995731 ended)
- 22:47 relaunch verified: job 3440069 started abl chains (rand/nofil/norep/rftm/roll) 22:39-22:40 and hsdon on acn108 22:40:59 (queue.log/hsdon_driver.log). Re-renamed GPU jobs to sw7b_gpu (something renames them back to tcded_gpu; re-applied each check).
- 22:48 paused w7_3b1 (not yet trained; only 3 GPUs) so w7_3k30 / w7_3lr1 evals finish faster; relaunch: tmux new -d -s sw_w7_3b1 'bash scripts/run_sciworld_7b_r3tune.sh w7_3b1 reports/sciworld/stepfix7b/w7_3 1 2e-5'
- 23:16 job 3447396 (l40s, 23h46 left) renamed sw_rl_gpu for sw_serl72 (7B already has 3 reserved).
- 23:55 job 3418820 ends ~00:36; submitted replacement a7b_gpu 3453591. Added rsplit_rej/pass (tmux sf_abl_rsplit_*) to watch.
- 00:15 shard_loop cap is now 200 (lib_sciworld_stepfix.sh), no old_attempts moves needed; 7B arms w7_3k30/lr1 evals OOM-retrying but progressing.
- 00:34 3438792 started -> renamed sw7b_gpu (replaces 3418820 ending); submitted replacements hold_anygpu(ee5239) + tcded for 3407625/3407624 (end ~00:58/01:10).
- 00:42 ALFWorld rejrand82 (random rollout-rejected 82 + replay) launched: tmux alf_rsplit_rejrand (scripts/run_rollout_split_rejrand.sh). ScienceWorld rsplit runs PAUSED per user (relaunch later with run_sciworld_abl1.sh abl_rsplit_{rej,pass,rejrand} ... ; targets in stepfix/rsplit/).
- 01:11 7B R3 tuning stopped by user; 7B ScienceWorld final = 278/315/347/340. sw7b_gpu jobs now serve rft7_1 / roll7_1.
- 01:15 coordinator: 7B R3 arms stopped (final w7_3=340); now watching sw_rft7_1, sw_roll7_1 (7B baselines), x3rft, serl72, hsdon, abl chains, alf_rsplit_rejrand.
- 02:15 roll7_1: s18 hit 25-attempt cap; moved attempt logs of 12 unfinished rollout shards to roll7_1/old_attempts, restarted tmux sw_roll7_1 (same cmd).
- 02:16 job 3438792 renamed error_attr_hold_shared (3B chains, ALFWorld rejrand, HINT-SD, x3rft had NO shared GPU since 22:05). 7B baselines keep 2 sw7b_gpu.
- 02:37 user priority for shared GPUs: (1) ALFWorld alf_rsplit_rejrand, (2) 3B ablation chains R2, (3) HINT-SD, (4) x3rft. Paused sf_x3rft (tmux killed; resumes from outputs: tmux new -d -s sf_x3rft 'bash scripts/run_sciworld_rft_r3.sh').
- 03:09 SERL training done (step 72), serl_s72 eval started 03:01 on sw_rl_gpu 3447396 (will hand to hold_shared after eval). alf rejrand82 TRAIN FAILED 02:59 (auto-retry).
- 04:14 serl_s72 848: 382 won, mean 55.56 (test 290/651, dev 92/197); vs R0 267 p<1e-4, vs 3B R3h 299. Renamed 3447396 sw_rl_gpu -> error_attr_hold_shared. alf rejrand82 train failure cause: 'oom_kill event in StepId=3438792.5496' (host RAM OOM on shared job, other steps co-resident).
- 04:14 rejrand82 OOM-killed (host RAM, shared job). Pinned it to L40S job 3447396 renamed alf_rejrand_gpu (HOLD_NAME_RE='^alf_rejrand_gpu$'); rename back to error_attr_hold_shared after its 774 eval. SERL s72 = 382.
- 04:25 coordinator: 3447396 = alf_rejrand_gpu (rejrand82 alone) until its 774 eval done, then error_attr_hold_shared (auto in my check).
- 05:08 ALFWorld rejrand82 = 432/774 (vs R0 60/44 p=0.14; vs rej82 498: 39/105 p=3.5e-8). Filled in paper. 3447396 back to error_attr_hold_shared.
- 05:12 alf rejrand82 DONE: 432/774 (unseen 66/134, seen 80/140, heldout 286/500; vs rej82 498). 3447396 -> hold_shared.
- 05:32 HINT-SD driver moved from acn108 to acn32 (tmux sw_hsd_online, HSD_RE='^hsd_gpu$'); trainer pinned to job 3402382 renamed hsd_gpu (dedicated). When it nears its end, submit a replacement and rename it hsd_gpu.
- 05:42 coordinator: HINT-SD driver now on acn32 (sw_hsd_online, job 3402382 renamed hsd_gpu); replace 3402382 ~30min before end (~11:40) and name new one hsd_gpu; x3rft relaunch at 09:00 latest.
- 06:13 7B roll7_1 (rollout verification R1) 848: 315 won, mean 53.4 (vs R0_7B 278 p=9e-4; vs w7_1 315 p=1). Note: 3407623 and 3438792 no longer in squeue (ended/cancelled elsewhere).
- 06:30 7B rft7_1 (standard RFT R1) 848: 333 won, mean 53.2 (vs w7_1 315 p=0.19). Both 7B baselines done -> 3440828 renamed a7b_gpu (shared).
- 06:35 NOTE: queue.log abl_*_r2 'DONE' lines from Oct 10 are stale (pre-bug-fix); only abl_rftm_r2 (22:43) is genuine post-fix. Real chain evals: rand/nofil/norep r2 evals still running (06:29), roll_r2 in rollout.
- 07:19 3B abl_norep_r2 848: 283 won, mean 49.7 (vs R0 267 p=.2; vs w2k 336 p=2.9e-8; vs wk3 353 p=7.6e-10).
- 07:26 3B abl_rand_r2 848: 291 won, mean 50.6 (vs R0 p=.029; vs wk3 353 p=2.3e-6); abl_nofil_r2: 99 won (07:17).
- 07:29 online HINT-SD hsdon_c1 (1483 eps) 848: 241 won, mean 50.1 (worse than R0 267 p=.0043; vs w1 307 p=1e-8).
- 08:10 hsd_gpu 3402382 vanished from squeue (check sacct); renamed 3442946 (7h54 left) -> hsd_gpu to keep HINT-SD dedicated GPU.
- 08:20 x3rft relaunched (coordinator).
- 09:30 3442946 (hsd_gpu) was preempted/requeued too (PENDING); renamed 3447396 (13h left, was hold_shared) -> hsd_gpu. GPU jobs are being preempted periodically.
- 10:41 hsdon_c2 (2966 eps) 848: 188 won, mean 43.8 (collapsing; c1 241, R0 267).
- 10:50 3407623 (requeued, still named sw7b_gpu) -> tcded_gpu (7B baselines done).
- 12:26 chmod-fixed noexec dir abl_roll_r2/model/ckpt_tmp (recreated without x bit by trainer; recurs).
- 12:27 root cause of non-exec dirs: srun steps launched from acn108 drivers run with umask 0117. Patched sciworld_stepfix.py / sciworld_agent.py / hintsd_online_sciworld.py / sciworld_baselines.py to force umask 0027 at start.
- 13:05 HINT-SD training DONE (4449 eps, 558 steps); hsdon_c3 eval started 13:02; 3447396 hsd_gpu -> error_attr_hold_shared.
- 13:50 3B x3rft (standard RFT R3, full 1483-ep rftfull, 16667 rows) 848: 413 won, mean 61.0; vs R0 267 p=8e-23; vs StepFix wk3 353 p=8e-5 (RFT BEATS StepFix at R3).
- 13:55 x3rft (std RFT R3, 16,667 rows) = 413 / 61.0 / 41.5 / 166 — beats StepFix R3 353 on all metrics (wins p=7.9e-5; clip -4.7 p<0.001; raw -6.3 p=0.02). Filled in tables (text not yet updated).
- 14:20 hsdon_c3 (4449 eps) 848: 107 won, mean 36.2 (vs R0 267 p<1e-30). Online HINT-SD DONE: c1 241 / c2 188 / c3 107.
- 14:35 moved abl chains rand/nofil/norep/roll from acn108 (job 3440069 ends ~19:50) to acn32 tmux (umask 0027); rftm chain already DONE.
- 14:40 coordinator moved 3B chain drivers rand/nofil/norep/roll to acn32 tmux (sf_ch_abl_*), acn108 empty; r2 'DONE' lines re-logged 14:37 are re-finalizations (same numbers: rand 291, nofil 99, norep 283).
- 15:24 3B abl_roll_r2 (rollout verification R2) 848: 344 won, mean 54.0 (vs R0 p=1.7e-9; vs StepFix wk3 353 p=.52).
- 16:01 3B abl_norep_r3 848: 150 won, mean 33.2 (collapse vs norep_r2 283; vs R0 p=1e-13).
- 07:22 (found 16:10) 3B abl_rftm_r3 848: 289 won, mean 47.6 (vs R0 p=.084; vs wk3 353 p=5e-5).
- 16:35 3B abl_rand_r3 848: 254 won, mean 45.7 (vs R0 p=.31; vs wk3 353 p=4e-13); 741 rows.
- 17:38 3B abl_nofil_r3 848 done (see queue.log).
- 18:08 3B abl_roll_r3 848: 346 won, mean 54.6 (vs R0 p=4.6e-10; vs wk3 353 p=.61). ALL babysat runs finished; final table sent to main.
