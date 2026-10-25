#!/bin/bash
# Oct 5 05:35: the plain log rule broke round 2 (flags the wrong take, corrections = "go elsewhere", student stops taking).
# Stop ALL five final chains (running bashes hold the old script inode), archive the s1/s2/rollout round-2 pointer-dependent
# outputs, relaunch every chain with the step-back rule (collect / Model A / replay resume from files).
cd /users/8/pan00389/multi-agent/error-attribution; SA=reports/alfworld/self_attr_rounds
for s in final_chain_s1 final_chain_s2 final_chains; do tmux kill-session -t $s 2>/dev/null; done
for p in $(pgrep -f 'run_final_chains_resume.sh|scripts/fix/resume_chain.sh|scripts/self_attr_round.sh selfattr'); do kill $p 2>/dev/null; done
sleep 2
for p in $(pgrep -f 'tag selfattr(_s1|_s2|_rollout|_placebo)?_r[23]'); do kill $p 2>/dev/null; done
sleep 8
echo "remaining chain processes: $(pgrep -f 'scripts/self_attr_round.sh|tag selfattr(_s1|_s2|_rollout|_placebo)?_r[23]' | wc -l)"
for ch in selfattr_s1 selfattr_s2 selfattr_rollout; do D=$SA/$ch/r2; A=$D/archive_logrule_plain; mkdir -p "$A"
  for f in pointer.jsonl pointer.raw.jsonl pointer.log diag.raw.jsonl teacher.log onestep.log distill.log targets_env_top model model_rollout proxy self eval_unseen.jsonl eval_seen.jsonl eval_unseen.release.jsonl eval_seen.release.jsonl eval.log train.log ADAPTER ATTR; do
    [ -e "$D/$f" ] && mv "$D/$f" "$A/"; done
  mkdir -p $D/self $D/proxy; echo "$ch r2 archived: $(ls $A | tr '\n' ' ')"; done
echo "=== $(date '+%F %T') chains: step selector switched to the step-back log rule; s1/s2/rollout round 2 redone (plain-rule outputs archived), main R3 / placebo R2 restarted" | tee -a $SA/queue.log
tmux new-session -d -s final_chain_main "bash scripts/fix/resume_chain.sh selfattr '' 0 error_attr_hold_igpu_qw"; sleep 45
tmux new-session -d -s final_chain_s1 "bash scripts/fix/resume_chain.sh selfattr _s1 1 any"; sleep 45
tmux new-session -d -s final_chain_s2 "bash scripts/fix/resume_chain.sh selfattr _s2 2 any"; sleep 45
tmux new-session -d -s final_chain_rollout "bash scripts/fix/resume_chain.sh selfattr_rollout '' 0 any"; sleep 45
tmux new-session -d -s final_chain_placebo "bash scripts/fix/resume_chain.sh selfattr_placebo '' 0 any"
echo relaunched
