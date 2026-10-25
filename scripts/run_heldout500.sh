#!/bin/bash
# Daemon: greedy evaluation on the 500 held-out training games (ids 1500-1999, never used by SFT/OPD/self-evolution) for every
# adapter listed in reports/alfworld/ext/heldout_queue.txt (lines "tag<TAB>adapter_path"; adapters that do not exist yet are
# retried on the next scan).  Output reports/alfworld/ext/eval_heldout_<tag>.jsonl (resumable), one log line per finished tag.
# Picks whichever GPU hold has more free memory.  Adds the 274 + 500 = 774-game pooled result when the 274 exist next to the adapter.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; Q=${HQ:-reports/alfworld/ext/heldout_queue.txt}; L=reports/alfworld/eval_extended.log
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "=== $(date '+%F %T') $*" | tee -a $L reports/alfworld/self_evolve3b/queue.log; }
freeof() { srun --jobid=$1 --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1; }
pick() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge "${1:-10000}" ] 2>/dev/null && echo $best; }
while true; do
  did=0
  while IFS=$'\t' read -r tag ad; do
    [ -z "$tag" ] && continue; out=reports/alfworld/ext/eval_heldout_$tag.jsonl
    [ "$(grep -c . $out 2>/dev/null)" -ge 500 ] && continue
    [ "$ad" != "base" ] && [ ! -f "$ad/adapter_config.json" ] && continue
    H=""; until H=$(pick); [ -n "$H" ]; do sleep 60; done
    A=""; [ "$ad" != "base" ] && A="--adapter $ad"
    srun --jobid=$H --overlap -n1 -c4 --mem=0 $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct $A --split train --game_offset 1500 --n_games 500 --budget 50 --temperature 0 --out $out --tag h500_$tag >> reports/alfworld/ext/heldout_$tag.log 2>&1 || sleep 30
    did=1
    if [ "$(grep -c . $out)" -ge 500 ]; then d=$(dirname $(dirname $ad)); u=$d/eval_unseen.jsonl; s=$d/eval_seen.jsonl
      extra=""; [ -f $u ] && [ -f $s ] && extra=" | 274: $(( $(grep -c '"won": true' $u) + $(grep -c '"won": true' $s) )) | 774 total: $(( $(grep -c '"won": true' $u) + $(grep -c '"won": true' $s) + $(grep -c '"won": true' $out) ))"
      log "heldout500 $tag: $(grep -c '"won": true' $out)/500 (student 280/500)$extra"; fi
  done < $Q
  [ $did = 0 ] && sleep 300
done
