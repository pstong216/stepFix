#!/bin/bash
# TextCraft R1_v4 = the FINAL recipe as on ALFWorld (user-approved Oct 9).  REUSES the R1 collection and R1's STANDARD Model A pointer
# (attributor_no_observations, variant no_observations: r1/pointer.jsonl + r1/diag_modelA.raw.jsonl, 379 failures, K = 76).
#   teacher   = FullHindsight full-view informed teacher (v3's), at R1's standard pointer states -> new candidates (top-2 excluding a-)
#   evidence  = one-step F (textcraft_evidence, unchanged), recomputed for the new candidates (deterministic env)
#   selection = env_top_margin (candidate F strictly above the taken action's F), K = 76
#   replay    = fail_weighted by depth with carry-over (as v3)
#   train     = same trainer / hyper-parameters from R0's LoRA; command scoring in chunks of 8 (original) for the first 3 attempts, chunks
#               of 2 (memory only, same math) after that -- the chunk actually used is written to model_chunk.txt
#   eval      = the same 810-task greedy TEST eval
# Shard before resume, one file per shard, coverage checks before every next stage.
source scripts/lib_baselines.sh; set -o pipefail
F=reports/textcraft/final; R0=$F/r0; R1=$F/r1; V2=$F/r1_v2; V3=$F/r1_v3; V=$F/r1_v4; T=$R1/traj.jsonl
STU=reports/textcraft/student100/final; mkdir -p $V; Q=$F/queue.log
tlog() { echo "=== $(date '+%F %T') $*" | tee -a $Q; }
cover() { $P scripts/textcraft_round.py cover "$@" > /dev/null 2>&1; }
tlog "TextCraft R1_v4 (final recipe) driver start"
# ---- R1's standard pointer, unchanged
for f in ids_fail.jsonl K.json pointer.jsonl pointer.raw.jsonl diag_modelA.raw.jsonl; do [ -s $V/$f ] || cp $R1/$f $V/$f; cmp -s $V/$f $R1/$f || { tlog "v4: $f differs from R1 -- stopping"; exit 1; }; done
$P scripts/textcraft_round.py cover --glob "$R1/modelA_pred_s*.jsonl" --key id --ids $V/ids_fail.jsonl | tee -a $Q || { tlog "v4: R1 Model A coverage FAILED -- stopping"; exit 1; }
# ---- full-view informed teacher at R1's pointer states (3 GPU shards)
for i in 0 1 2; do ( until grep -q "DONE informed_teacher" $V/teacher_s$i.log 2>/dev/null; do $(S 16000) $P -u scripts/textcraft_round.py teacher --traj $T --out_dir $V --adapter $STU --shard $i 3 --full_view >> $V/teacher_s$i.log 2>&1 || sleep 60; done ) & sleep 20; done; wait
$P scripts/textcraft_round.py cover --glob "$V/targets_s*.jsonl" --key idt --ids $V/ids_fail.jsonl | tee -a $Q || { tlog "v4 teacher coverage FAILED -- stopping"; exit 1; }
# ---- one-step evidence (4 CPU shards on this node)
for i in 0 1 2 3; do ( until grep -q "DONE onestep" $V/onestep_s$i.log 2>/dev/null; do $P -u scripts/textcraft_round.py onestep --traj $T --out_dir $V --shard $i 4 >> $V/onestep_s$i.log 2>&1 || sleep 10; done ) & done; wait
$P scripts/textcraft_round.py cover --glob "$V/onestep_s*.jsonl" --key idt --ids $V/ids_fail.jsonl | tee -a $Q || { tlog "v4 one-step coverage FAILED -- stopping"; exit 1; }
# ---- selection + replay
$P scripts/textcraft_round.py select --traj $T --out_dir $V --rule env_top_margin | tee -a $Q
$P scripts/textcraft_round.py replay_fw --traj $T --out $V/replay.jsonl --n 100 --carry | tee -a $Q
# ---- train (same trainer / hyper-parameters, from R0's LoRA)
att=0
until [ -f $V/model/adapter/adapter_config.json ]; do
  att=$((att + 1)); CH=$([ $att -le 3 ] && echo 8 || echo 2)
  $(S 24000) env TC_LP_CHUNK=$CH $P -u scripts/textcraft_round.py train --data $V/targets_env_top_margin/train.jsonl:1.0 $V/replay.jsonl:1.0 --adapter $PWD/$STU --lr 1e-5 --epochs 1 --batch 8 --seed 0 --out $V/model > $V/train.log 2>&1 && { echo $CH > $V/model_chunk.txt; for k in $(seq 60); do ls $V/model/adapter > /dev/null 2>&1; [ -f $V/model/adapter/adapter_config.json ] && break; sleep 2; done; } || { tlog "R1_v4 TRAIN FAILED (attempt $att, chunk $CH; retry in 3 min)"; sleep 180; }
done
tlog "R1_v4 trained with command-scoring chunk $(cat $V/model_chunk.txt): $(tail -n 1 $V/train.log)"
# ---- TEST eval (810, greedy)
k=0; for spec in 0:270 270:270 540:270; do off=${spec%:*}; n=${spec#*:}; o=$V/eval_test_s$k.jsonl
  ( until cover --glob $o --key task --split test810 --offset $off --n $n; do $(S 12000) $P -u scripts/textcraft_agent.py collect --adapter $V/model/adapter --split test810 --offset $off --n $n --temperature 0 --out $o --tag tcf_r1v4 >> ${o%.jsonl}.log 2>&1 || sleep 60; done ) &
  k=$((k + 1)); sleep 20; done; wait
$P scripts/textcraft_round.py cover --glob "$V/eval_test_s*.jsonl" --key task --split test810 --out $V/eval_test.jsonl | tee -a $Q || { tlog "R1_v4 eval coverage FAILED -- stopping"; exit 1; }
$P scripts/textcraft_agent.py report $R0/eval_test.jsonl $R1/eval_test.jsonl $V2/eval_test.jsonl $V3/eval_test.jsonl $V/eval_test.jsonl | tee -a $Q
for ref in "$R0:R0" "$R1:R1_old" "$V2:R1_v2" "$V3:R1_v3"; do $P scripts/textcraft_round.py compare --r0 ${ref%:*}/eval_test.jsonl --r1 $V/eval_test.jsonl --names ${ref#*:},R1_v4 | tee -a $Q; done
tlog "TEXTCRAFT R1_v4 DONE"
