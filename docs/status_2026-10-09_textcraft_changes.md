# TextCraft R1_v3 (user decision Oct 9 ~00:45)
Changes vs the R1 protocol (reports/textcraft/final/r1_v3):
1. Model A sees everything: same weights (attributor_no_observations), rendered with observations (variant full) + full task statement incl. crafting commands. NOTE: ALFWorld runs used the no-observation input (its training format; removing observations raised ALFWorld step accuracy 53.4 -> 64.3 in the attributor study). Cross-dataset consistency to be checked (offline pointer comparison on ALFWorld proposed).
2. Teacher sees the whole failed trajectory: no 10-step prefix cap, no 25-step future cap, no 160-char observation cut (TextCraft-only implementation; ALFWorld runs used the truncated HindsightHistory).
3. Margin rule (env_top_margin).
4. Replay: fail_weighted by crafting depth (more quota for deeper tasks), complete trajectories.
Evidence F unchanged: the "get" dominance came mainly from false corrections (46/76 get->bigger get at already-good steps; removed by the margin rule) and from the real error mix (too few / missing ingredients = 2/3 of first problems); replay of complete deep trajectories supplies craft steps.

## Results (Oct 9 ~02:45)
| | R0 | R1 (old) | R1_v2 (margin + depth replay) | R1_v3 (+ full-info Model A + full-view teacher) |
|---|---|---|---|---|
| TEST success /810 | 454 | 439 | 459 (vs R0 61/56 p=0.71; vs old R1 54/34 p=0.04) | **282** (vs R0 31/203 p=4e-32) |
| get share of actions | 0.30 | 0.54 | 0.54 | 0.85 |
| true corrections / 76 | — | 30 | 53 | 66 |
| get replacing a craft | — | 13 | 22 | 55 |
Full-info Model A on TextCraft DID point better (step-0 share 0.52 -> 0.18; at the first failed action 242/379 vs 39; hand check 6/15 exact, 11/15 on a wrong action vs 5/15, 8/15) — but it points at the failed CRAFT (proximate), not the earlier under-count GET (root). The correction there is always "get more X" (teacher proposed a craft in 52/758 candidates; 7 with F>0), so 55 craft->get one-hot targets taught "stop crafting": 383/528 failures run out of budget looping on gets. TextCraft-specific; fixes (localize the under-count get / per-type balance / craft credit) need the user's decision.

## R1_v4 = the final recipe (standard Model A pointer + full-view teacher + env_top_margin + fail_weighted replay by depth) — Oct 9 ~13:15
**470/810 (58.0%)**, best variant: vs R0 454 66/50 p=0.16 (goal-level 32/29 p=0.80); vs old R1 439 60/29 p=0.0013; vs v2 459 40/29 p=0.23; vs v3 282 206/18.
Depth 1/2/3/4: 176/263/31/0 (R0 179/246/29/0). Get share 0.52 (R0 0.30, v3 0.85). Kept 76: 57 true corrections, 25 = oracle next action; 75 get / 1 craft (19 get replacing a craft vs 55 in v3).

## R2_v4 (old final recipe, round 2): 417/810 — significantly worse than R1_v4 (470; 44/97 p=1e-5) and R0 (454; 50/87 p=0.002)
Collection by R1_v4 at T=1.0: 869/1512 (R0 910). K=83, 65 true corrections, 79 get / 4 craft. Get share of actions 0.30 -> 0.52 -> 0.68; failures ending in the same get 4x: 3 -> 60 -> 106. Action-only "get more" corrections compound into get loops across rounds -> motivates the per-step plan-then-act redesign (reports/textcraft/plan).
