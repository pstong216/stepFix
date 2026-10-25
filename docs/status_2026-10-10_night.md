# Oct 10 night log (continued from status_2026-10-07_overnight.md)
- 04:15: tcded job 3185163 renamed to tcded_rft and reserved for the ScienceWorld RFT baselines; the StepFix pickers match only ^tcded_gpu$.
- 04:15: ScienceWorld r2f (refined recipe R2) = 299/848.
  - vs R0: 78/46, p=0.005; vs r1b: 34/34, tie; vs old-recipe R2 (308): 24/33, p=0.29.
  - The refined recipe gives no gain at R2. Suspected cause: margin + fail-proportional quota moved corrections away from the high-yield lifespan focus fixes; 18 of 63 corrections were redistributed.
- 7B RFT R3 training resumed from its checkpoint (2400/16760) after its job expired; checkpointing works.
- 7B new-recipe ablations running (a7b_randv3, a7b_rollv3). rand-v3 R2 trained (50 random corrections + 100 fail-weighted replay); its eval is running.
