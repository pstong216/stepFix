#!/bin/bash
# User-requested (Oct 7 19:45, "如果有空余7b的模型可以开始跑起来了"): the FINAL recipe (20% corrections + 100 replay) with a 7B student.
# R0 = Qwen2.5-7B-Instruct + LoRA (r=32), SFT on the SAME 100 expert trajectories as the 3B student, 1 epoch, lr 1e-4, effective batch 16
# (bs 4 x grad_accum 4; 3B used bs 8 x 2). Then R1-R3 with scripts/run_k20_round.sh (Model A stays the frozen 3B attributor; the teacher is the
# 7B student itself with hindsight). 774 evaluation each round.
export MODEL=Qwen/Qwen2.5-7B-Instruct GPU_MIN=20000 TRAIN_MIN=36000
source scripts/lib_baselines.sh; C=a7b_final; D=$SA/$C/r0; mkdir -p $D/model
until [ -f $D/model/final/adapter_config.json ]; do $(S 36000) $P -u scripts/alfworld_sft.py --traj reports/alfworld/expert_all_train_full.jsonl --max_games 100 --epochs 1 --bs 4 --grad_accum 4 --model $MODEL --out $D/model > $D/sft.log 2>&1 || { log "$C r0: SFT FAILED (retry)"; sleep 180; }; done
[ -e $D/model/adapter ] || ln -s final $D/model/adapter
log "$C r0: 7B SFT student ready ($D/model/adapter)"
[ -s $E/eval_heldout_a7b_r0.jsonl ] || eval774 $D a7b_r0
for R in 1 2 3; do bash scripts/run_k20_round.sh $C $R $SA/$C/r$((R-1))/model/adapter a7b_r$R 1 || { log "$C r$R FAILED"; exit 1; }; done
log "$C R0->R3 DONE"
