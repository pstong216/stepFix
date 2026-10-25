#!/bin/bash
# Hold $1: fine-tune Model A on ALFWorld self-play labels -> localisation on held-out oracle-r3 failures -> attribute the 319-failure set -> OPD arm attralf_nofb.
J=$1; cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=reports/alfworld/attributor_alf_chain.log; SR="srun --jobid=$J --overlap -n1 -c4 --mem=0"
echo "=== $(date) [1] fine-tune 7B no_obs attributor on $(wc -l < data/attributor_sft_alf/train.jsonl) ALFWorld rows (hold $J, $(squeue -j $J -h -o %N))" | tee -a $L
$SR $P -u scripts/finetune_attributor_alf.py --init reports/attributor_7b_no_observations/final --data data/attributor_sft_alf/train.jsonl --out reports/attributor_7b_alf --epochs 1 --lr 3e-5 >> $L 2>&1 || { echo FT_FAILED | tee -a $L; exit 1; }
echo "=== $(date) [2] localisation on held-out oracle-r3 failures (new vs old attributor)" | tee -a $L
$SR $P -u scripts/alfworld_attribute.py --attributor reports/attributor_7b_alf/final --variant no_observations --release reports/alfworld/student3b_low100_round3_oracle_train_b50.release.jsonl --out reports/alfworld/attr7balf_round3_oracle.jsonl >> $L 2>&1
$SR $P -u scripts/alfworld_attribute.py --attributor reports/attributor_7b_no_observations/final --variant no_observations --release reports/alfworld/student3b_low100_round3_oracle_train_b50.release.jsonl --out reports/alfworld/attr7bnoobs_round3_oracle.jsonl >> $L 2>&1
$P scripts/alfworld_localizability_report.py --traj reports/alfworld/student3b_low100_round3_oracle_train_b50.jsonl --progress reports/alfworld/progress_student3b_low100_round3_oracle.jsonl --attr new=reports/alfworld/attr7balf_round3_oracle.jsonl old=reports/alfworld/attr7bnoobs_round3_oracle.jsonl 2>&1 | grep -v Warning | tee -a $L
echo "=== $(date) [3] attribute the round-1 failure set with the new attributor" | tee -a $L
$SR $P -u scripts/alfworld_attribute.py --attributor reports/attributor_7b_alf/final --variant no_observations --release reports/alfworld/student3b_low100_train_b50.release.jsonl --out reports/alfworld/attr7balf_student3b_low100_train_b50.jsonl >> $L 2>&1
echo "=== $(date) [4] OPD arm attralf_nofb on the shared 319 failures" | tee -a $L
export TAG=student3b_low100 TEACHER=Qwen/Qwen2.5-3B-Instruct TEACHER_ADAPTER=reports/alfworld/student3b_sft/final ROUND=round1 IDS_FROM=reports/alfworld/opd/student3b_low100/attr/steps.jsonl
scripts/run_alfworld_opd_arm.sh $J attralf_nofb 1000 2 >> $L 2>&1
echo "=== $(date) ATTRALF_CHAIN_DONE: $(grep -E '^DONE [0-9]+ episodes' reports/alfworld/opd/student3b_low100/attralf_nofb/run.log | tail -1)" | tee -a $L
