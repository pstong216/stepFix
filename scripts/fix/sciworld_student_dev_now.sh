#!/bin/bash
# Run the ScienceWorld round-1 student dev baseline now (same command/paths as sciworld_round.sh step 29, so the driver reuses it).
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; R=reports/sciworld/round1; STU=reports/sciworld/student3/final
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pick() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge 10000 ] 2>/dev/null && echo $best; }
until grep -q "DONE collect" $R/eval_student.log 2>/dev/null; do H=""; until H=$(pick); [ -n "$H" ]; do sleep 60; done
  srun --jobid=$H --overlap -n1 -c4 --mem=0 $P -u scripts/sciworld_agent.py collect --adapter $STU --split dev --per_task 7 --budget 50 --temperature 0 --out $R/eval_dev_student.jsonl --tag sw_student >> $R/eval_student.log 2>&1 || sleep 60; done
echo "=== $(date '+%F %T') SCIWORLD r1 student (early run): dev $(grep DONE $R/eval_student.log | tail -n1 | sed 's/DONE collect //') success | mean score $($P -c "import json;r=[json.loads(l) for l in open('$R/eval_dev_student.jsonl')];print(round(sum(e['final_score'] for e in r)/len(r),1))")" | tee -a reports/sciworld/queue.log
