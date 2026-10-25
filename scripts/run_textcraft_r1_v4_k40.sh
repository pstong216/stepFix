#!/bin/bash
# TextCraft R1_v4_k40 (user-approved Oct 9): identical to R1_v4 except K = round(0.4 x 379) = 152.
#   REUSES r1_v4's teacher candidates (targets_s*.jsonl) and one-step evidence (onestep_s*.jsonl) unchanged (read in place) and r1_v4's
#   replay file; env_top_margin keeps every qualifying state up to 152; same trainer / hyper-parameters from R0's LoRA; 810-task TEST eval.
source scripts/lib_baselines.sh; set -o pipefail
F=reports/textcraft/final; R0=$F/r0; R1=$F/r1; V4=$F/r1_v4; V=$F/r1_v4_k40; T=$R1/traj.jsonl
STU=reports/textcraft/student100/final; mkdir -p $V; Q=$F/queue.log
tlog() { echo "=== $(date '+%F %T') $*" | tee -a $Q; }
cover() { $P scripts/textcraft_round.py cover "$@" > /dev/null 2>&1; }
tlog "TextCraft R1_v4_k40 driver start"
$P scripts/textcraft_round.py cover --glob "$V4/targets_s*.jsonl" --key idt --ids $V4/ids_fail.jsonl | tee -a $Q || { tlog "k40: r1_v4 teacher coverage FAILED -- stopping"; exit 1; }
$P scripts/textcraft_round.py cover --glob "$V4/onestep_s*.jsonl" --key idt --ids $V4/ids_fail.jsonl | tee -a $Q || { tlog "k40: r1_v4 one-step coverage FAILED -- stopping"; exit 1; }
K=$($P -c "print(round(0.4 * $(grep -c . $V4/ids_fail.jsonl)))"); tlog "k40: K = round(0.4 x $(grep -c . $V4/ids_fail.jsonl)) = $K"
[ -s $V/targets_env_top_margin/train.jsonl ] || $P scripts/textcraft_round.py select --traj $T --out_dir $V4 --k $K --rule env_top_margin --dest $V | sed 's/^=== textcraft/=== [k40] textcraft/' | tee -a $Q
[ -s $V/replay.jsonl ] || cp $V4/replay.jsonl $V/replay.jsonl; cmp -s $V/replay.jsonl $V4/replay.jsonl || { tlog "k40: replay differs from r1_v4 -- stopping"; exit 1; }
att=0
until [ -f $V/model/adapter/adapter_config.json ]; do
  att=$((att + 1)); CH=$([ $att -le 3 ] && echo 8 || echo 2)
  $(S 24000) env TC_LP_CHUNK=$CH $P -u scripts/textcraft_round.py train --data $V/targets_env_top_margin/train.jsonl:1.0 $V/replay.jsonl:1.0 --adapter $PWD/$STU --lr 1e-5 --epochs 1 --batch 8 --seed 0 --out $V/model > $V/train.log 2>&1 && { echo $CH > $V/model_chunk.txt; for k in $(seq 60); do ls $V/model/adapter > /dev/null 2>&1; [ -f $V/model/adapter/adapter_config.json ] && break; sleep 2; done; } || { tlog "R1_v4_k40 TRAIN FAILED (attempt $att, chunk $CH; retry in 3 min)"; sleep 180; }
done
tlog "R1_v4_k40 trained with command-scoring chunk $(cat $V/model_chunk.txt): $(tail -n 1 $V/train.log)"
k=0; for spec in 0:270 270:270 540:270; do off=${spec%:*}; n=${spec#*:}; o=$V/eval_test_s$k.jsonl
  ( until cover --glob $o --key task --split test810 --offset $off --n $n; do $(S 12000) $P -u scripts/textcraft_agent.py collect --adapter $V/model/adapter --split test810 --offset $off --n $n --temperature 0 --out $o --tag tcf_r1v4k40 >> ${o%.jsonl}.log 2>&1 || sleep 60; done ) &
  k=$((k + 1)); sleep 20; done; wait
$P scripts/textcraft_round.py cover --glob "$V/eval_test_s*.jsonl" --key task --split test810 --out $V/eval_test.jsonl | tee -a $Q || { tlog "R1_v4_k40 eval coverage FAILED -- stopping"; exit 1; }
$P scripts/textcraft_agent.py report $R0/eval_test.jsonl $V4/eval_test.jsonl $V/eval_test.jsonl | tee -a $Q
for ref in "$R0:R0" "$V4:R1_v4"; do $P scripts/textcraft_round.py compare --r0 ${ref%:*}/eval_test.jsonl --r1 $V/eval_test.jsonl --names ${ref#*:},R1_v4_k40 | tee -a $Q; done
tlog "TEXTCRAFT R1_v4_k40 DONE"
