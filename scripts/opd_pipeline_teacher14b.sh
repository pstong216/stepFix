#!/bin/bash
# 7B student distilled (CE, sequence-level) from the 14B privileged teacher; plus verified-only variants.
set -euo pipefail
cd /users/8/pan00389/multi-agent/error-attribution
P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=Qwen/Qwen2.5-7B-Instruct; D=reports/opd_debate/dataset_teacher14b.jsonl
for ARM in attr random uniform; do
  echo "=== $(date) train t14b arm=$ARM (ce)"
  $P -u scripts/opd_train.py --student_model $S --data $D --arm $ARM --loss ce --output_dir reports/opd_debate/student_t14b_$ARM
done
for ARM in attr uniform; do
  echo "=== $(date) train t14b arm=$ARM VERIFIED-only (ce)"
  $P -u scripts/opd_train.py --student_model $S --data $D --arm $ARM --loss ce --verified_only --output_dir reports/opd_debate/student_t14b_${ARM}_verified
done
for ARM in attr random uniform attr_verified uniform_verified; do
  echo "=== $(date) eval t14b arm=$ARM"
  $P -u scripts/opd_eval.py --student_model $S --adapter reports/opd_debate/student_t14b_$ARM/final --name debate_t14b_$ARM --split test
done
echo "=== $(date) DONE"
