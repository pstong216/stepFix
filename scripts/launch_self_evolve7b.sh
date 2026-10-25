#!/bin/bash
# Wait for the 7B zero-shot collection (two shards; resume a shard on the new hold if its A100 job ended) and the new A40 holds.
cd /users/8/pan00389/multi-agent/error-attribution; O=reports/alfworld/self_evolve7b; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until [ "$(squeue -j 2467981 -h -o %T)" = "RUNNING" ] && [ "$(squeue -j 2467982 -h -o %T)" = "RUNNING" ]; do
  grep -q "^DONE" <(tail -3 $O/collect_a.log) && grep -q "^DONE" <(tail -3 $O/collect_b.log) && [ -n "$READY" ] && break; sleep 60; done
for s in a:0:2467981 b:150:2467982; do IFS=: read sh off job <<< "$s"
  until grep -q "^DONE" <(tail -3 $O/collect_$sh.log) && ! tmux has-session -t se7_c$sh 2>/dev/null; do
    if ! tmux has-session -t se7_c$sh 2>/dev/null; then
      srun --jobid=$job --overlap -n1 -c4 --mem=0 $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-7B-Instruct --split train --game_offset $off --n_games 150 --budget 50 --temperature 1.0 --out $O/r1_train_$sh.jsonl --tag se7b_r1 >> $O/collect_$sh.log 2>&1
    else sleep 30; fi
  done
done
cat $O/r1_train_a.jsonl $O/r1_train_b.jsonl > $O/r1_train.jsonl; cat $O/r1_train_a.release.jsonl $O/r1_train_b.release.jsonl > $O/r1_train.release.jsonl
echo "=== $(date '+%F %T') collected $(wc -l < $O/r1_train.jsonl) episodes, won $(grep -c '"won": true' $O/r1_train.jsonl)" >> $O/queue.log
bash scripts/run_self_evolve7b.sh 2467981 2467982
