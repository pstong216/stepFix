#!/bin/bash
# Restart every final-recipe driver so it runs with EVIDENCE=generic (all stages resume from files; GPU steps in flight are re-run).
cd /users/8/pan00389/multi-agent/error-attribution
for s in hind_main hind_placebo hind_rollout hind_rollsub hind_eqv hind_r2fix hind_r2tp hind_r1late; do tmux kill-session -t $s 2>/dev/null; done; sleep 2
for p in $(pgrep -f 'run_final_chains_hind.sh|r2_fix_vs_late.sh|r2_teacher_point.sh|r1_late.sh'); do kill $p 2>/dev/null; done
for p in $(pgrep -f 'self_attr_round.sh selfattr(_rollout|_rollsub|_eqv|_placebo)? [12] '); do kill $p 2>/dev/null; done; sleep 2
for p in $(pgrep -f 'tag selfattr(_rollout|_rollsub|_eqv|_placebo)?_hind(late|fix|tp)?_r[12]|selfattr_hindlate/r1/self'); do kill $p 2>/dev/null; done; sleep 5
echo "left: $(pgrep -f 'selfattr[a-z_]*hind' | wc -l)"
echo "=== $(date '+%F %T') user: ONE generic evidence template for all datasets (EVIDENCE=generic; vocabulary-free first/late selectors); final drivers restarted (resume from files). R1 top-100 under generic = 96/100 identical to the specific one" | tee -a reports/alfworld/self_attr_rounds/queue.log
tmux new-session -d -s hind_main "bash scripts/run_final_chains_hind.sh main"; sleep 20
for c in placebo rollout rollsub eqv; do tmux new-session -d -s hind_$c "bash scripts/run_final_chains_hind.sh $c"; sleep 20; done
tmux new-session -d -s hind_r2fix "bash scripts/fix/r2_fix_vs_late.sh"; tmux new-session -d -s hind_r2tp "bash scripts/fix/r2_teacher_point.sh"; tmux new-session -d -s hind_r1late "bash scripts/fix/r1_late.sh"
echo relaunched
