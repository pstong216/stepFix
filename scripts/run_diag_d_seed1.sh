#!/bin/bash
# User-approved (Oct 6 23:16, "加 D-d"): D-d = the final R1's 164 corrections, NEW trainer, no replay, training seed 1 instead of 0
# (only the training order changes) -- is the 440 of a3bfinal_noreplay an accident?
source scripts/lib_baselines.sh; D=$SA/diag_d_seed1; mkdir -p $D
cp $SA/selfattr_a3bfinal/r1/targets_env_top/train.jsonl $D/train.jsonl; log "diag_d_seed1: start (164 corrections, new trainer, seed 1)"
until [ -f $D/model/adapter/adapter_config.json ]; do $(S 24000) $P -u scripts/alfworld_opd_plus.py train --data $D/train.jsonl:1.0 --student Qwen/Qwen2.5-3B-Instruct --adapter $PWD/$STU --lr 1e-5 --epochs 1 --batch 8 --seed 1 --out $D/model > $D/train.log 2>&1 || { log "diag_d_seed1: TRAIN FAILED (retry)"; sleep 180; }; done
[ -s $E/eval_heldout_diag_d_seed1.jsonl ] || eval774 $D diag_d_seed1
