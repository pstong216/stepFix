#!/bin/bash
# Seeds for the main result (autonomous, Oct 7 23:15; user authorized paper-needed runs): the FINAL 20% chain (a3bfinal_k20) repeated
# with training seed SEED (arg 1). R1 uses the same R1 data and the same 82 corrections (Model A pointer + teacher are deterministic),
# so seeds differ in training order and replay sampling; R2/R3 recollect with their own models (sampling at T=1.0) via run_k20_round.sh.
export SEED=$1; source scripts/lib_baselines.sh; C=a3bfinal_k20_s$SEED; D=$SA/$C/r1; mkdir -p $D
[ -s $D/targets.jsonl ] || cp $SA/a3bfinal_k20/r1/targets.jsonl $D/targets.jsonl
[ -s $D/replay.jsonl ] || $P scripts/make_replay.py --traj $T1 --n 100 --seed $((1 + 1000 * SEED)) --out $D/replay.jsonl | tee -a $SA/queue.log
log "$C r1: $(grep -c . $D/targets.jsonl) corrections + 100 replay, training seed $SEED"
train1 $D $STU "$D/targets.jsonl:1.0 $D/replay.jsonl:1.0"; [ -s $E/eval_heldout_a3bk20s${SEED}_r1.jsonl ] || eval774 $D a3bk20s${SEED}_r1
for R in 2 3; do bash scripts/run_k20_round.sh $C $R $SA/$C/r$((R-1))/model/adapter a3bk20s${SEED}_r$R 1 || { log "$C r$R FAILED"; exit 1; }; done
log "$C R1->R3 DONE"
