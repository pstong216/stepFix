#!/bin/bash
# Oct 9 (user: "修一下bug 看看我们在ScienceWorld 还可不可行"): re-evaluate the SAME student10 on the SAME 197 dev episodes
# (per_task 7, seed 0, budget 50, greedy) with the snapping bug fixed (--snap safe); 4 round-robin shards on any free GPU.
export HOLD_NAME_RE='^a7b_gpu$'   # 21:33: keep off the TextCraft GPUs; 21:45: 22 GB minimum so it never squeezes next to the 7B RFT training (not resumable)
source scripts/lib_baselines.sh; O=reports/sciworld/refix
for i in 0 1 2 3; do ( until grep -q "DONE collect" $O/eval_s$i.log 2>/dev/null; do $(S 22000) $P -u scripts/sciworld_agent.py collect --adapter reports/sciworld/student10/final --split dev --per_task 7 --budget 50 --temperature 0 --seed 0 --snap safe --shard $i 4 --out $O/eval_dev_safe_s$i.jsonl --tag refix >> $O/eval_s$i.log 2>&1 || sleep 30; done ) & sleep 20; done; wait
cat $O/eval_dev_safe_s?.jsonl > $O/eval_dev_safe.jsonl; echo "=== $(date '+%F %T') sciworld refix eval DONE: $(grep -c '"won": true' $O/eval_dev_safe.jsonl)/$(grep -c . $O/eval_dev_safe.jsonl)" | tee -a reports/sciworld/queue.log
