#!/bin/bash
# Location-precision gradient (x / x+2 / x-2 / none), researcher 2026-09-18.
# Usage: run_offset_experiment.sh <slurm_job_id>
set -uo pipefail
cd /users/8/pan00389/multi-agent/error-attribution
P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
J=$1; SR="srun --jobid=$J --overlap -n1 -c4 --mem=0"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== $(date) [A] Model B: magentic-one 150-set, plus2/minus2 (native extra round)"
$SR $P -u scripts/run_selfplay_feedback_chain.py --framework magentic-one --protocol debate_round --ids_file reports/selfplay_chain_magentic_150/ids.json --conditions plus2_only,minus2_only --out_dir reports/selfplay_chain_magentic_150_offsets
echo "=== $(date) [B] Model B: debate 118 true failures, plus2/minus2"
$SR $P -u scripts/run_selfplay_feedback_chain.py --framework debate --protocol debate_round --ids_file reports/selfplay_chain_debate_round_150/true_failure_ids.json --conditions plus2_only,minus2_only --out_dir reports/selfplay_chain_debate_round_150_offsets
echo "=== $(date) [C] Model B: mathchat 150-set, plus2/minus2"
$SR $P -u scripts/run_selfplay_feedback_chain.py --framework mathchat --protocol debate_round --ids_file reports/selfplay_chain_mathchat_150/ids.json --conditions plus2_only,minus2_only --out_dir reports/selfplay_chain_mathchat_150_offsets
echo "=== $(date) [D] external reader: 434 traces x (x+2, x-2)"
$SR $P -u scripts/run_feedback_study.py --prompts_path data/feedback_study/prompts_offsets.jsonl --out reports/feedback_study_results_offsets.jsonl --batch_size 4 --resume
echo "=== $(date) [E] Attr-GRPO pre-check: group-variance probe (3B and 7B teams, G=8, 60 train tasks)"
$SR $P -u scripts/grpo_group_variance_probe.py --model Qwen/Qwen2.5-3B-Instruct --n_tasks 60 --group 8 --out reports/grpo_probe_3b.json
$SR $P -u scripts/grpo_group_variance_probe.py --model Qwen/Qwen2.5-7B-Instruct --n_tasks 60 --group 8 --out reports/grpo_probe_7b.json
echo "=== $(date) [F] Attr-GRPO smoke test (3B team + 7B attributor, 2 steps, G=4, 4 tasks/step)"
$SR $P -u scripts/attr_grpo.py --arm attr --group 4 --tasks_per_step 4 --min_informative 1 --steps 2 --eval_every 2 --eval_subset 8 --output_dir /tmp/attr_grpo_smoke
echo "=== $(date) DONE"
scancel $J
