#!/bin/bash
# Oct 9 (user): once the dedicated TextCraft GPU job (tcded_gpu) runs, restart the TextCraft drivers so their GPU steps prefer it
# (HOLD_NAME_RE also keeps the normal holds as a fallback; the dedicated GPU usually has the most free memory, so pick() chooses it).
cd /users/8/pan00389/multi-agent/error-attribution
desc() { local p; for p in $(ps -o pid= --ppid $1); do echo $p; desc $p; done; }
until [ "$(squeue -u pan00389 -h -n tcded_gpu -t R -o %i | head -1)" != "" ]; do sleep 60; done
J=$(squeue -u pan00389 -h -n tcded_gpu -t R -o %i | head -1); echo "=== $(date '+%F %T') dedicated TextCraft GPU job $J running on $(squeue -h -j $J -o %N)" | tee -a reports/textcraft/final/queue.log
for DRV in run_textcraft_r1_v4_k40.sh run_textcraft_r2_v4.sh; do
  R=$(pgrep -f "^bash scripts/$DRV" | head -1); [ -z "$R" ] && continue
  ALL="$R $(desc $R)"; B=$(for p in $ALL; do ps -o pid=,comm= -p $p; done | awk '$2=="bash"{print $1}'); S=$(for p in $ALL; do ps -o pid=,comm= -p $p; done | awk '$2=="srun"{print $1}')
  kill $B 2>/dev/null; sleep 1; kill $S 2>/dev/null; sleep 5
  case $DRV in run_textcraft_r1_v4_k40.sh) TM=tc_r1v4k40; LOG=reports/textcraft/final/r1_v4_k40/driver.log;; run_textcraft_r2_v4.sh) TM=tc_r2v4; LOG=reports/textcraft/final/r2_v4/driver.log;; esac
  tmux kill-session -t $TM 2>/dev/null
  tmux new-session -d -s $TM -c $PWD "HOLD_NAME_RE='^(tcded_gpu|error_attr_hold_.*)\$' bash scripts/$DRV >> $LOG 2>&1"
  echo "=== $(date '+%F %T') restarted $DRV bound to the dedicated GPU (resumes from its completed stages)" | tee -a reports/textcraft/final/queue.log
done
