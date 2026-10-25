#!/bin/bash
# Compression study at matched size: 500 train rows (same ids), Model A (7B LoRA, 3 epochs) on original vs LLM-compressed
# step bodies; eval on the same 200 val rows (each in its own format) and on the ALFWorld student's failures.
H=$1; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ERROR_ATTR_COMPRESS_CACHE=$PWD/data/compress/cache.jsonl
SR="srun --jobid=$H --overlap -n1 -c4 --mem=0"; O=reports/modelA_compress500; M=$O/queue.log; mkdir -p $O
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
log "compress 1370 missing bodies"; $SR $P -u scripts/compress_bodies.py generate --only_hashes data/compress/needed_hashes.json --batch 16 >> $O/compress.log 2>&1 || { log "compress FAILED"; exit 1; }
log "prepare llm_compressed data"; $P -u scripts/prepare_attributor_data.py --variant llm_compressed --out_dir data/attributor_sft_llm_compressed >> $O/prepare.log 2>&1 || { log "prepare FAILED"; exit 1; }
$P - <<'PY' >> $M
import json
a=[json.loads(l) for l in open("data/attributor_sft_no_observations/train.jsonl")]; b={json.loads(l)["id"]:json.loads(l) for l in open("data/attributor_sft_llm_compressed/train.jsonl")}
ids=set(json.load(open("data/compress/train500_ids.json"))); r=[len(b[x["id"]]["prompt"])/len(x["prompt"]) for x in a if x["id"] in ids]
print(f"prompt length compressed/original over the 500 train rows: mean {sum(r)/len(r):.2f}")
PY
for v in no_observations llm_compressed; do
  D=data/attributor_sft_$v; A=$O/$v
  [ -d $A/final ] || { log "train $v (500 rows)"; $SR $P -u scripts/train_attributor.py --model Qwen/Qwen2.5-7B-Instruct --data_dir $D --output_dir $A --keep_ids data/compress/train500_ids.json --epochs 3 --save_each_epoch >> $O/train_$v.log 2>&1 || { log "train $v FAILED"; continue; }; }
  for ck in $A/final; do
    $SR $P -u scripts/eval_attributor.py --model $ck --split val --data_dir $D --limit 200 > $O/eval_${v}_val.log 2>&1
    $SR $P -u scripts/alfworld_attribute.py --attributor $ck --variant $v --release reports/alfworld/student3b_low100_train_b50.release.jsonl --out $O/alf_$v.jsonl > $O/alf_$v.log 2>&1
  done
  log "$v val: $(grep -E 'step  accuracy|all-3' $O/eval_${v}_val.log | tr '\n' ' ')"
done
log "COMPRESS500 DONE"
