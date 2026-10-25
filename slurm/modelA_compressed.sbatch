#!/bin/bash
#SBATCH --job-name=modelA_compress
#SBATCH --account=ee5239
#SBATCH --partition=a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/users/8/pan00389/multi-agent/error-attribution/reports/modelA_compressed/slurm_%j.out
# Model A trained on LLM-rephrased (compressed, ~30% length) step bodies; same 7B / LoRA / 3 epochs as the original, a
# checkpoint per epoch; evaluated on compressed Who&When val, compressed ALFWorld transfer, and compressed student failures.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ERROR_ATTR_COMPRESS_CACHE=$PWD/data/compress/cache.jsonl
O=reports/modelA_compressed; D=data/attributor_sft_llm_compressed
until grep -q "DONE generate" $O/compress.log 2>/dev/null; do $P -u scripts/compress_bodies.py generate --batch 24 >> $O/compress.log 2>&1 || sleep 60; done
[ -f $D/train.jsonl ] || $P -u scripts/prepare_attributor_data.py --variant llm_compressed --out_dir $D > $O/prepare.log 2>&1
[ -d $O/final ] || $P -u scripts/train_attributor.py --model Qwen/Qwen2.5-7B-Instruct --data_dir $D --output_dir $O --epochs 3 --save_each_epoch > $O/train.log 2>&1
for ck in $(ls -d $O/checkpoint-* | sort -t- -k2 -n); do
  n=$(basename $ck)
  [ -f $ck/tokenizer_config.json ] || $P -c "from transformers import AutoTokenizer as T; T.from_pretrained('Qwen/Qwen2.5-7B-Instruct').save_pretrained('$ck')"
  for sp in val transfer_alfworld; do [ -f $O/eval_${n}_$sp.log ] || $P -u scripts/eval_attributor.py --model $ck --split $sp --data_dir $D > $O/eval_${n}_$sp.log 2>&1; done
  $P -u scripts/alfworld_attribute.py --attributor $ck --variant llm_compressed --release reports/alfworld/student3b_low100_train_b50.release.jsonl --out $O/alf_${n}.jsonl > $O/alf_${n}.log 2>&1
done
echo DONE
