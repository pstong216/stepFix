#!/bin/bash
# 3B Model A check (user Oct 6 11:35): same as the 7B Model A arm but the diagnosis comes from the 3B agent-text-only attributor
# (Qwen2.5-3B LoRA on Who&When Pro; ALFWorld step accuracy ~0.63 vs 0.91 for 7B).  K=40% + replay 100.
cd /users/8/pan00389/multi-agent/error-attribution; SA=reports/alfworld/self_attr_rounds
MODELA_PATH=reports/attributor_no_observations/final EVIDENCE=generic STEP=logfix STEP_POOL=1 K_FRAC=0.4 REPLAY_N=100 CHAIN_SUFFIX=_hindpoolA3b SEED=0 bash scripts/self_attr_round.sh selfattr 2 $(cat $SA/selfattr_hind/r1/ADAPTER) reports/alfworld/self_attr/r0/both/adapter any
echo "=== $(date '+%F %T') 3B Model A check (K40+replay100) DONE (selfattr_hindpoolA3b)" | tee -a $SA/queue.log
