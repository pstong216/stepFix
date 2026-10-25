#!/bin/bash
#SBATCH --job-name=modelA_epochs
#SBATCH --account=qianwen
#SBATCH --partition=a100-4,a100-8
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/users/8/pan00389/multi-agent/error-attribution/reports/modelA_epochs/slurm_%j.out
# Model A (7B, no_observations) retrained for 5 epochs with a checkpoint per epoch; each checkpoint evaluated on
# Who&When val, the ALFWorld transfer split, and the ALFWorld student's round-1 failures (pointer vs planner).
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
while [ "$(squeue -j 2358103 -h -o %T 2>/dev/null)" = "RUNNING" ]; do sleep 60; done   # never overlap with the interactive-hold run writing the same dir
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; O=reports/modelA_epochs; D=data/attributor_sft_no_observations
[ -d $O/final ] || $P -u scripts/train_attributor.py --model Qwen/Qwen2.5-7B-Instruct --data_dir $D --output_dir $O --epochs 5 --save_each_epoch --resume >> $O/train.log 2>&1
for ck in $(ls -d $O/checkpoint-* | sort -t- -k2 -n); do
  n=$(basename $ck)
  [ -f $ck/tokenizer_config.json ] || $P -c "from transformers import AutoTokenizer as T; T.from_pretrained('Qwen/Qwen2.5-7B-Instruct').save_pretrained('$ck')"
  for sp in val transfer_alfworld; do [ -f $O/eval_${n}_$sp.log ] || $P -u scripts/eval_attributor.py --model $ck --split $sp --data_dir $D > $O/eval_${n}_$sp.log 2>&1; done
  $P -u scripts/alfworld_attribute.py --attributor $ck --variant no_observations --release reports/alfworld/student3b_low100_train_b50.release.jsonl --out $O/alf_${n}.jsonl > $O/alf_${n}.log 2>&1
done
echo DONE
