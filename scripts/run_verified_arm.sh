#!/bin/bash
# Train on VERIFIED real fixes only (attr arm) vs matched placebo; DPO lr 1e-6, 1 and 3 epochs; flip probe; eval 274.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; O=reports/alfworld/self_evolve7b; M=$O/queue.log
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SE_DIR=$O SE_TRAJ=$O/r1_train.jsonl SE_MODEL=Qwen/Qwen2.5-7B-Instruct SE_ADAPTER=
until grep -q "DONE verify" $O/attr/verify_s0.log && grep -q "DONE verify" $O/attr/verify_s1.log; do sleep 30; done
$P scripts/build_verified_pairs.py >> $M 2>&1
srun --jobid=2467981 --overlap -n1 -c4 --mem=0 $P -u scripts/self_evolve.py refs --arms attr_verified,placebo_verified > $O/refs_verified.log 2>&1
tr() { local a=$1 ep=$2 J=$3; srun --jobid=$J --overlap -n1 -c4 --mem=0 $P -u scripts/alfworld_opd_plus.py train --data $O/$a/dpo_pairs.jsonl --student Qwen/Qwen2.5-7B-Instruct --adapter "" --lr 1e-6 --epochs $ep --beta 0.1 --out $O/$a/m_ep$ep > $O/$a/train_ep$ep.log 2>&1; }
(tr attr_verified 1 2467981; tr attr_verified 3 2467981) & (tr placebo_verified 1 2467982; tr placebo_verified 3 2467982) & wait
srun --jobid=2467981 --overlap -n1 -c4 --mem=0 $P -u scripts/measure_flip.py attr_verified-ep1=$O/attr_verified/m_ep1/adapter attr_verified-ep3=$O/attr_verified/m_ep3/adapter placebo_verified-ep1=$O/placebo_verified/m_ep1/adapter placebo_verified-ep3=$O/placebo_verified/m_ep3/adapter > $O/flip_verified.log 2>&1
ev() { local a=$1 ep=$2 J=$3; for sp in valid_unseen valid_seen; do srun --jobid=$J --overlap -n1 -c4 --mem=0 $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-7B-Instruct --adapter $O/$a/m_ep$ep/adapter --split $sp --budget 50 --temperature 0 --out $O/$a/eval_${sp#valid_}_ep$ep.jsonl --tag ver_${a}_$ep >> $O/$a/eval_ep$ep.log 2>&1; done
  echo "=== $(date '+%F %T') verified $a ep$ep: $(wc -l < $O/$a/dpo_pairs.jsonl) pairs | unseen $(grep -c '"won": true' $O/$a/eval_unseen_ep$ep.jsonl)/134 seen $(grep -c '"won": true' $O/$a/eval_seen_ep$ep.jsonl)/140 (zero-shot 60, 50)" >> $M; }
(ev attr_verified 3 2467981; ev attr_verified 1 2467981) & (ev placebo_verified 3 2467982; ev placebo_verified 1 2467982) & wait
echo "=== $(date '+%F %T') verified arm DONE" >> $M
