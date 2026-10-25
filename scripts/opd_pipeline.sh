#!/bin/bash
# Attr-OPD end-to-end for one student size. Usage: opd_pipeline.sh <student_model> <tag> <collect:0|1>
set -euo pipefail
cd /users/8/pan00389/multi-agent/error-attribution
P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STUDENT=$1; TAG=$2; COLLECT=$3
echo "=== $(date) pipeline student=$STUDENT tag=$TAG"
if [ "$COLLECT" = "1" ]; then
  echo "=== $(date) [1/5] collect rollouts"
  $P -u scripts/run_selfplay_debate.py --model "$STUDENT" --tag "$TAG" --benchmarks math,gpqa,mmlu_pro,scibench
fi
echo "=== $(date) [2/5] attribute all failures with Model A"
$P -u scripts/opd_attribute_all.py --tag "$TAG"
echo "=== $(date) [3/5] privileged-teacher dataset (attr/random/uniform)"
$P -u scripts/opd_build_dataset.py --tag "$TAG" --teacher_model "$STUDENT"
for ARM in attr random uniform; do
  echo "=== $(date) [4/5] train arm=$ARM"
  $P -u scripts/opd_train.py --student_model "$STUDENT" --data reports/opd_$TAG/dataset.jsonl --arm $ARM --output_dir reports/opd_$TAG/student_$ARM
done
echo "=== $(date) [5/5] evaluate base + 3 arms on TEST tasks"
$P -u scripts/opd_eval.py --student_model "$STUDENT" --name ${TAG}_base --split test
for ARM in attr random uniform; do
  $P -u scripts/opd_eval.py --student_model "$STUDENT" --adapter reports/opd_$TAG/student_$ARM/final --name ${TAG}_$ARM --split test
done
echo "=== $(date) DONE"
