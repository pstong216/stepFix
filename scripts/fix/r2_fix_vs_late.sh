#!/bin/bash
# Same-trajectory control for the LATE rule (Oct 5 21:55): once the final chain's R2 collection is done, run R2 again on the SAME
# trajectories with the step-back rule (STEP_FORCE=logfix) in selfattr_hindfix/r2, from the same R1 adapter.  Compare 274/774.
cd /users/8/pan00389/multi-agent/error-attribution; SA=reports/alfworld/self_attr_rounds; S=$SA/selfattr_hind/r2; N=$SA/selfattr_hindfix/r2
until tail -n 3 $S/collect.log 2>/dev/null | grep -q '^DONE' && [ -s $S/replay.jsonl ]; do sleep 120; done
mkdir -p $N/self $N/proxy $N/envlab $N/attr_labels; for f in traj.jsonl traj.release.jsonl collect.log replay.jsonl; do [ -e $N/$f ] || cp $S/$f $N/; done
echo "=== $(date '+%F %T') R2 same-trajectory control: step-back rule (selfattr_hindfix) vs late rule (selfattr_hind)" | tee -a $SA/queue.log
EVIDENCE=generic STEP=logfix STEP_FORCE=logfix TEACHER_NO_DIAG=1 K=100 CHAIN_SUFFIX=_hindfix SEED=0 bash scripts/self_attr_round.sh selfattr 2 $(cat $SA/selfattr_hind/r1/ADAPTER) reports/alfworld/self_attr/r0/both/adapter any
echo "=== $(date '+%F %T') R2 same-trajectory control DONE (selfattr_hindfix)" | tee -a $SA/queue.log
