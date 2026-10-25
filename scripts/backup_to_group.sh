#!/bin/bash
# Oct 25 2026: copy paper-relevant artifacts from scratch.global to permanent group storage.
set -u; umask 0027
DST=/projects/standard/qianwen/shared/pan00389-stepfix; mkdir -p $DST/{code,paper,models,results,misc}
SRC=/scratch.global/pan00389/error-attribution
R=$SRC/reports_repo
X='--exclude=ckpt* --exclude=ckpt_tmp --exclude=*.pt --exclude=optimizer* --exclude=checkpoint-*'
echo "== code"; rsync -a --exclude=reports --exclude=__pycache__ /users/8/pan00389/multi-agent/error-attribution/ $DST/code/
echo "== paper"; rsync -a --exclude='_bak*' --exclude='*_bak_*' --exclude='preview_*' --exclude='.codex-*' $SRC/paper/overleaf_draft/ $DST/paper/
echo "== misc"; rsync -a $SRC/efficiency $SRC/fonts $DST/misc/
echo "== models"
cd $R
for d in alfworld/student3b_low100 alfworld/student7b_low100 sciworld/student10 sciworld/student10_7b \
         attributor_no_observations/final attributor_7b_no_observations/final \
         alfworld/self_attr_rounds/a3bfinal_k20/r1/model alfworld/self_attr_rounds/a3bfinal_k20/r2/model alfworld/self_attr_rounds/a3bfinal_k20/r3/model alfworld/self_attr_rounds/a3bfinal_k20/r3_fixpool/model \
         alfworld/self_attr_rounds/a7b_final/r0/model alfworld/self_attr_rounds/a7b_final/r1/model alfworld/self_attr_rounds/a7b_final/r2/model alfworld/self_attr_rounds/a7b_final/r3*/model \
         sciworld/stepfix/w1/model sciworld/stepfix/w2k/model sciworld/stepfix/wk3/model \
         sciworld/stepfix7b/w7_1/model sciworld/stepfix7b/w7_2b3lr/model sciworld/stepfix7b/w7_3/model; do
  [ -e "$d" ] || { echo "MISSING $d"; continue; }
  rsync -a --relative $X "./$d" $DST/models/
done
echo "== results (all eval files + run logs/stats)"
rsync -a --include='*/' --include='eval*.jsonl' --include='queue.log' --include='*stats*.json' --include='pointer.jsonl' --include='targets*.jsonl' --include='train_log.jsonl' --exclude='*' --prune-empty-dirs $R/ $DST/results/
du -sh $DST/*; echo BACKUP DONE
