#!/bin/bash
# User-approved (Oct 6 23:05): why do the final recipe's R1 corrections WITHOUT replay give 440 when older corrections-only arms gave ~500?
#   D-a  the final R1's 164 corrections, OLD trainer (alfworld_opd_plus_v1_backup_oct6.py: bf16 gradient accumulation)
#   D-b  the old main_env_top86 corrections (A0 steps, plain-student candidates, ALFWorld-specific evidence; 500/499/505), NEW trainer
#   D-c  the final R1's top-86 corrections (same ranking), NEW trainer
# All: SFT start, lr 1e-5, batch 8, 1 epoch, no replay, 774 evaluation.
source scripts/lib_baselines.sh
run() { local name=$1 data=$2 trainer=$3 D=$SA/$1; mkdir -p $D; [ -s $D/train.jsonl ] || cp $data $D/train.jsonl
  until [ -f $D/model/adapter/adapter_config.json ]; do $(S 24000) $P -u scripts/$trainer train --data $D/train.jsonl:1.0 --student Qwen/Qwen2.5-3B-Instruct --adapter $PWD/$STU --lr 1e-5 --epochs 1 --batch 8 --seed 0 --out $D/model > $D/train.log 2>&1 || { log "$name: TRAIN FAILED (retry)"; sleep 180; }; done
  [ -s $E/eval_heldout_$name.jsonl ] || eval774 $D $name; }
log "trainer diagnostics D-a / D-b / D-c: start"
run diag_a_oldtrainer $SA/selfattr_a3bfinal/r1/targets_env_top/train.jsonl alfworld_opd_plus_v1_backup_oct6.py &
sleep 90; run diag_b_newtrainer $SA/selfattr_plain/r1/main_env_top/train.jsonl alfworld_opd_plus.py &
sleep 90; run diag_c_top86 $SA/diag_c_top86/train.jsonl alfworld_opd_plus.py &
wait; log "trainer diagnostics DONE"
