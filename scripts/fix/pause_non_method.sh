#!/bin/bash
# User (Oct 5 23:35): everything yields to FINALIZING THE METHOD.  Pause (resumable from files) all runs that do not decide the method:
# env-vs-rollout R2 chains (rollout/rollsub/eqv/placebo), final ablation runner rig_hind3 + arm builder, exact-ETO, old clean runners.
# Keep: A1 R1 late arm, main chain R2 + same-trajectory R2 controls (fix / teacher-point), random-step control, WebShop generic round, held-out daemon A.
cd /users/8/pan00389/multi-agent/error-attribution
for s in hind_rollout hind_rollsub hind_eqv hind_placebo rig_hind3 hind_arms eto_fair rig_clean1 rig_clean2 heldout500_b; do tmux kill-session -t $s 2>/dev/null; done; sleep 2
for p in $(pgrep -f 'run_final_chains_hind.sh (rollout|rollsub|eqv|placebo)|run_eto_fair.sh|build_hind_arms.sh'); do kill $p 2>/dev/null; done
for p in $(pgrep -f 'self_attr_round.sh selfattr_(rollout|rollsub|eqv|placebo) '); do kill $p 2>/dev/null; done; sleep 2
for p in $(pgrep -f 'tag selfattr_(rollout|rollsub|eqv|placebo)_r[23]|tag rg_hind_(placebo100|veto_only|teacherconf100|future100|all410|env_top_k308_75pct)_s1|alfworld_expert_collect|traj_dpo.py'); do kill $p 2>/dev/null; done; sleep 3
R=reports/alfworld/rigorous; for a in hind_placebo100 hind_veto_only hind_teacherconf100 hind_future100 hind_all410 hind_env_top_k308_75pct; do [ -f "$R/$a/DONE" ] || rm -f "${R:?}/${a:?}/CLAIM"; done
echo "=== $(date '+%F %T') user: method finalization first -> paused env-vs-rollout R2 chains, final ablations, exact ETO (all resumable)" | tee -a reports/alfworld/self_attr_rounds/queue.log
