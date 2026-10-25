#!/bin/bash
# Oct 9 22:10: ScienceWorld R0 TRAIN collection with the bug-fixed interface (--snap safe), student10 (300 demos), T=1.0, budget 50,
# 10 variations per task (300 episodes), 3 round-robin shards on idle TextCraft GPUs (the TextCraft plan3 R0 eval keeps priority: 14 GB minimum).
# Preparatory: the first stage of a ScienceWorld StepFix round; awaiting the user's go for the round itself.
export HOLD_NAME_RE='^tcded_gpu$'
source scripts/lib_baselines.sh; O=reports/sciworld/r0fix; mkdir -p $O
for i in 0 1 2; do ( until grep -q "DONE collect" $O/collect_s$i.log 2>/dev/null; do $(S 14000) $P -u scripts/sciworld_agent.py collect --adapter reports/sciworld/student10/final --split train --per_task 10 --budget 50 --temperature 1.0 --seed 1 --snap safe --shard $i 3 --out $O/traj_s$i.jsonl --tag r0fix >> $O/collect_s$i.log 2>&1 || sleep 30; done ) & sleep 20; done; wait
cat $O/traj_s?.jsonl > $O/traj.jsonl; echo "=== $(date '+%F %T') sciworld r0fix train collection DONE: $(grep -c '"won": true' $O/traj.jsonl)/$(grep -c . $O/traj.jsonl)" | tee -a reports/sciworld/queue.log
