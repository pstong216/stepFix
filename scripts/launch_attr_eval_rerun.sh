#!/bin/bash
# Re-run the attr arm's lost sampled final eval from the saved adapter on whichever hold starts first; cancel the other.
cd /users/8/pan00389/multi-agent/error-attribution
J=""; while [ -z "$J" ]; do for c in 1400283 1395915; do [ "$(squeue -j $c -h -o %t 2>/dev/null)" = "R" ] && J=$c && break; done; [ -z "$J" ] && sleep 30; done
for c in 1400283 1395915; do [ "$c" != "$J" ] && scancel $c; done
sleep 20
echo "$(date) eval rerun hold $J on $(squeue -j $J -h -o %N)" | tee -a reports/attr_grpo_3b/attr/run.log
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True srun --jobid=$J --overlap -n1 -c4 --mem=0 /users/8/pan00389/miniconda3/envs/error_attr/bin/python3 -u scripts/attr_grpo.py --arm attr --eval_only --adapter reports/attr_grpo_3b/attr/final --output_dir reports/attr_grpo_3b/attr --final_samples 4 --gen_batch 64 --attn_budget 0.8e8 >> reports/attr_grpo_3b/attr/run.log 2>&1
echo "=== $(date) attr eval rerun finished (srun exit $?)" | tee -a reports/attr_grpo_3b/attr/run.log
scancel $J
