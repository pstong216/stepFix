#!/bin/bash
cd /users/8/pan00389/multi-agent/error-attribution
srun --jobid=2898357 --overlap -n1 -c4 --mem=0 /users/8/pan00389/miniconda3/envs/error_attr/bin/python3 -u scripts/alfworld_opd_plus_v2.tmp.py train --data reports/alfworld/self_attr_rounds/selfattr_poolk40r100/r2/targets_env_pool/train.jsonl:1.0 reports/alfworld/self_attr_rounds/selfattr_poolk40r100/r2/replay.jsonl:1.0 --student Qwen/Qwen2.5-3B-Instruct --adapter $PWD/reports/alfworld/self_attr_rounds/selfattr_pool/r1/model/adapter --lr 1e-5 --epochs 1 --batch 8 --seed 0 --out reports/alfworld/tmp_nan/model > reports/alfworld/tmp_nan/train.log 2>&1
echo "EXIT $?" >> reports/alfworld/tmp_nan/train.log
