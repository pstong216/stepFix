#!/bin/bash
# Oct 2 replications on the fresh holds. Hold A (2644179): training-seed replication (seed 1) of the best proxy arms on the SAME
# targets (env_top, combo_top) and of the placebo -- how much of the spread is training noise?  Hold B (2644180): compression-SFT
# variance -- finish pruned_s1 seen eval, keep1 with a third sample (seed 2), raw seed 1.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; SE=reports/alfworld/self_evolve3b; C=reports/alfworld/compress_traj
STU=reports/alfworld/student3b_low100/final; export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SA="srun --jobid=2644179 --overlap -n1 -c4 --mem=0"; SB="srun --jobid=2644180 --overlap -n1 -c4 --mem=0"
log() { echo "=== $(date '+%F %T') $*" | tee -a $1; }
train_eval() { local S=$1 DATA=$2 D=$3 TAG=$4 SEED=$5; mkdir -p $D
  $S $P -u scripts/alfworld_opd_plus.py train --data $DATA --student Qwen/Qwen2.5-3B-Instruct --adapter $PWD/$STU --lr 1e-5 --epochs 1 --batch 8 --seed $SEED --out $D/model > $D/train.log 2>&1 || { echo "TRAIN FAILED $D"; return 1; }
  for sp in valid_unseen valid_seen; do $S $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $D/model/adapter --split $sp --budget 50 --temperature 0 --out $D/eval_${sp#valid_}.jsonl --tag $TAG >> $D/train.log 2>&1; done
  echo "unseen $(grep -c '"won": true' $D/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen.jsonl)/140"; }
( for r in env_top combo_top; do R=$(train_eval "$SA" $SE/proxy_arms/attr_$r/train.jsonl $SE/proxy_arms/attr_${r}_seed1 px_${r}_s1 1); log $SE/queue.log "3B proxy attr/$r TRAIN-SEED-1 replication: $R (seed 0: $(grep "3B proxy attr/$r:" $SE/queue.log | tail -n1 | grep -o 'unseen.*140'))"; done
  R=$(train_eval "$SA" $SE/placebo/train.jsonl $SE/placebo_seed1 placebo_s1 1); log $SE/queue.log "3B placebo TRAIN-SEED-1 replication: $R (seed 0: unseen 61/134 seen 80/140)" ) &
( D=$C/pruned_s1; $SB $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $D/model/adapter --split valid_seen --budget 50 --temperature 0 --out $D/eval_seen.jsonl --tag cmp_pruned_s1 >> $D/train.log 2>&1
  log $C/queue.log "compress-SFT pruned_s1 (seed 1) completed: unseen $(grep -c '"won": true' $D/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen.jsonl)/140 (seed-0 pruned 61/85)"
  $P scripts/build_compress_sft.py --n 200 --seed 2 --tag _s2 > /dev/null
  R=$(train_eval "$SB" $C/keep1_s2/train.jsonl $C/keep1_s2 cmp_keep1_s2 0); log $C/queue.log "compress-SFT keep1_s2 (seed 2 sample): $R (seed 0: 70/95 = 165; seed 1: 57/79 = 136)"
  R=$(train_eval "$SB" $C/raw_s1/train.jsonl $C/raw_s1 cmp_raw_s1 0); log $C/queue.log "compress-SFT raw_s1 (seed 1 sample): $R (seed 0 raw: 57/82 = 139)" ) &
wait; log $SE/queue.log "OCT2 REPLICATIONS DONE"
