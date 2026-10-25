#!/bin/bash
# User decision (Oct 5 01:45): one training seed first for every ablation arm; seeds 2-3 later.  Restart all rigorous runners
# with SEEDS=1 (per-arm work already done is reused; CLAIM markers of unfinished arms are cleared).
cd /users/8/pan00389/multi-agent/error-attribution; R=/users/8/pan00389/multi-agent/error-attribution/reports/alfworld/rigorous
for s in rigorous_c rigorous_d rigorous_main3 rigorous_c3 rigorous_d3 rigorous_sel rigorous_sel2 rigorous_sel3; do tmux kill-session -t $s 2>/dev/null; done
sleep 3
# stop their running GPU steps (training / evaluation of rigorous arms), all resumable
for p in $(pgrep -f 'tag rg_[A-Za-z]'); do kill $p 2>/dev/null; done
for p in $(pgrep -f 'rigorous/[A-Za-z0-9_]*/seed[0-9]/model '); do kill $p 2>/dev/null; done
sleep 5
for d in "${R:?}"/*/; do a=$(basename "$d"); [ -f "$d/DONE" ] && continue; rm -f "${R:?}/${a:?}/CLAIM"; done
echo "runners stopped, claims cleared; relaunching with SEEDS=1"
launch() { tmux new-session -d -s "$1" "RQ=$2 SEEDS='1' bash scripts/run_rigorous_arms_v3.sh"; sleep 75; }
launch rig_c1   $R/queue_c.txt          # B-set: V_plus_R, V_plus_B, V_verified125, (B_alone done), RFT
launch rig_c2   $R/queue_c.txt
launch rig_sel1 $R/selector_queue.txt   # log rule, log rule (step back)
launch rig_sel2 $R/selector_queue.txt
launch rig_m1   $R/main_queue.txt       # main table: placebo, teacher-confidence, future, all410, rollout-verified, teachers
launch rig_m2   $R/main_queue.txt
launch rig_d1   $R/queue_d.txt          # K-curve
echo "=== $(date '+%F %T') rigorous runners relaunched with ONE seed (7 runners)" | tee -a reports/alfworld/self_evolve3b/queue.log
