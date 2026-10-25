#!/bin/bash
# User decision (Oct 9 ~13:00, "R3用40%的数据吧"): R3 of the 3B final-recipe chain with K = 40% of failures instead of 20%. Everything else
# is the original final recipe: from the 20% chain's R2 (561), the same R3 collection and the FULL 249-state pool (a3bfinal_k20/r3_poolcheck:
# standard 3B Model A pointer, original hindsight self-teacher, generic one-step evidence), rule env_top (no margin), K = 0.4 x 249 = 100, the
# chain's own R3 replay file (shortest-wins), same trainer, 774 eval. Tag a3bk20_r3k40. (The all-40% chain selfattr_a3bfinal is 547/572/602.)
export MODEL=Qwen/Qwen2.5-3B-Instruct GPU_MIN=10000 TRAIN_MIN=24000 EVIDENCE=generic
source scripts/lib_baselines.sh; P3=$SA/a3bfinal_k20/r3_poolcheck; S3=$SA/a3bfinal_k20/r3; D=$SA/a3bfinal_k20/r3_k40; mkdir -p $D
NF=$(grep -c . $S3/ids_fail.jsonl); K=$($P -c "print(max(1, round(0.4 * $NF)))")
[ -s $D/targets_env_top/train.jsonl ] || SE_DIR=$P3 SE_TRAJ=$S3/traj.jsonl $P scripts/proxy_arms_se3b.py --arm self --rule env_top --k $K --out $D/targets_env_top | tee -a $SA/queue.log
log "a3bk20 r3_k40: K = 0.4 x $NF = $K -> $(grep -c . $D/targets_env_top/train.jsonl) targets + the chain's R3 replay ($(grep -c . $S3/replay.jsonl))"
train1 $D $SA/a3bfinal_k20/r2/model/adapter "$D/targets_env_top/train.jsonl:1.0 $S3/replay.jsonl:1.0"; [ -s $E/eval_heldout_a3bk20_r3k40.jsonl ] || eval774 $D a3bk20_r3k40
log "a3bk20 r3_k40 DONE (a3bk20_r3k40; 20% R3 = 591 / full-pool 574; all-40% chain R3 = 602)"
