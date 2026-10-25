#!/bin/bash
# ScienceWorld 7B student R0 + R1 (Oct 10 2026 ~15:20, user: "先把7B的R1 science world跑了，用之前的配方").  tmux on cn1019:  bash scripts/run_sciworld_7b_r1.sh
# The 3B protocol with Qwen2.5-7B-Instruct as the student:
#  SFT   : sciworld_agent.py sft --per_task 10 --epochs 3 (what reports/sciworld/student10 actually ran: 290 trajectories, 12358 pairs, 2319 steps),
#          lr 1e-4, bs 4 x grad_accum 4, history 10, max_len 3072, seed 0, LoRA r32; checkpoint every 100 steps + auto-resume  -> student10_7b
#  R0    : greedy 848 eval (test 30/task seed 0 + dev 7/task seed 0), budget 50, --snap safe
#  R1    : = r1b (strict StepFix + relaxed validity + refill): 518 train eps (per_task 20, seed 1, T=1.0, budget 50, --snap safe) with the 7B R0;
#          Model A = 7B no_observations attributor (user change Oct 10 15:20); teacher = the 7B student itself, standard view, 8 samples,
#          --accept_env --max_extra 16, top-2 by teacher log-prob; one-step evidence; env_top K = round(0.2 x failures); replay 100 = 2 random
#          steps of each of the 50 shortest wins; train 1 ep lr 1e-5 bs 8 from the 7B LoRA; 848 eval.  + RANDOM control (K random teacher top-1).
source scripts/lib_sciworld_7b.sh; R=1; D=$SF7/r1; mkdir -p $D
s7log "7B driver start on $(hostname) (model $MODEL; Model A $MA7)"

# ---------------------------------------------------------------- SFT (single GPU, resumable)
clean_ckpt() { for c in $STU7DIR/checkpoint-*; do [ -d "$c" ] || continue
    { [ -f $c/trainer_state.json ] && [ -f $c/adapter_model.safetensors ] && [ -f $c/optimizer.pt ]; } || { s7log "SFT: removing incomplete $c"; rm -rf $c; }; done; }
# SFT GPU: the reserved job sw7b_gpu (SFT_RE overrides) with >= SFT7 MiB free and >= 1 h left, preferring the job with the most time left
pick_sft() { local skip=" $(cat $SF3/.skip_jobs 2>/dev/null | tr '\n' ' ') "
  squeue -u pan00389 -h -o "%i %j %T %L" | awk -v re="${SFT_RE:-^sw7b_gpu$}" '$2 ~ re && $3=="RUNNING"{print $1, $4}' | while read H L; do
    [ -z "$SFT_RE" ] || case "$skip" in *" $H "*) continue;; esac   # sw7b_gpu is on .skip_jobs to keep the other agents off it; it is reserved for this run
    local sec=$(echo $L | awk -F'[-:]' '{if(NF==4)print $1*86400+$2*3600+$3*60+$4; else if(NF==3)print $1*3600+$2*60+$3; else print $1*60+$2}')
    [ "$sec" -lt ${SFT_MINSEC:-3600} ] && continue   # skip jobs ending within the hour
    for N in $(scontrol show hostnames $(squeue -h -j $H -o %N) 2>/dev/null); do
      local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 -w $N nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits < /dev/null 2>/dev/null | head -n1)
      [ "${m:-0}" -ge $SFT7 ] 2>/dev/null && echo "$sec $H $N"; done; done | sort -rn | head -1 | awk '{print $2, $3}'; }
if [ ! -f $STU7/adapter_config.json ]; then
  n=0; t0=$(date +%s); s7log "SFT 7B start"
  until [ -f $STU7/adapter_config.json ]; do n=$((n + 1)); clean_ckpt; A=$SF7/sft_student10_7b.log.a$n; while [ -e $A ]; do n=$((n + 1)); A=$SF7/sft_student10_7b.log.a$n; done
    HN=""; until HN=$(pick_sft); [ -n "$HN" ]; do sleep 60; done; set -- $HN
    echo "### attempt $n $(date '+%F %T') job $1 node $2" > $A; s7log "SFT attempt $n on job $1 $2"
    srun --jobid=$1 --overlap -n1 -c4 --mem=0 -w $2 $P -u scripts/sciworld_agent.py sft --model $MODEL --per_task 10 --epochs 3 --out $STU7DIR --save_steps 100 >> $A 2>&1 \
      || { s7log "SFT attempt $n FAILED rc=$? ($A: $(tr '\r' '\n' < $A | grep -E 'Error|error' | tail -1 | cut -c1-200))"; sleep 60; }
    [ $n -ge 30 ] && { s7log "SFT GIVING UP"; exit 1; }; done
  s7log "SFT 7B DONE in $(( ($(date +%s) - t0) / 60 )) min: $(cat $SF7/sft_student10_7b.log.a* | tr '\r' '\n' | grep -o "{'train_runtime[^}]*}" | tail -1)"
fi

# ---------------------------------------------------------------- R0 848 eval
R0D=$SF7/r0
if [ ! -s $R0D/eval_dev.jsonl ] || [ "$(cat $R0D/eval_test.jsonl $R0D/eval_dev.jsonl 2>/dev/null | grep -c .)" -lt 848 ]; then
  s7log "R0 7B eval start"; eval7 $R0D $STU7 r0sw7
  s7log "R0 7B 848: $(sum848 $R0D/eval_test.jsonl $R0D/eval_dev.jsonl)"
  $P scripts/sciworld_stepfix_compare.py --a R0_3B=$SF3/r0/eval_test.jsonl,reports/sciworld/refix/eval_dev_safe.jsonl --b R0_7B=$R0D/eval_test.jsonl,$R0D/eval_dev.jsonl | tee -a $SF7/queue.log
fi

# ---------------------------------------------------------------- R1 collection (518 train eps, seed 1)
if [ ! -s $D/traj.jsonl ]; then
  s7log "R1 collection start"; collect7 $D/col $STU7 1 r1sf7 15 --per_task 20
  $P scripts/sciworld_stepfix.py prep --dir $D --inputs "$D/col/traj_s*.jsonl" --expect 518 | tee -a $SF7/queue.log
  $P -c "
L=open('$D/traj.release.jsonl').readlines()
for i in range(3): open('$D/rel_s%d.jsonl'%i,'w').writelines(L[i::3])"
fi

# ---------------------------------------------------------------- Model A (7B, no_observations)
if [ ! -s $D/pointer.jsonl ]; then
  s7log "R1 Model A: $MA7 (variant no_observations)"
  for i in 0 1 2; do NEED=$GMEM7 loop7 $D/modelA_s$i.log "^DONE" scripts/alfworld_attribute.py --attributor $MA7 --variant no_observations --release $D/rel_s$i.jsonl --out $D/modelA_pred_s$i.jsonl & sleep 40; done; wait
  $P scripts/sciworld_stepfix.py pointer --dir $D | tee -a $SF7/queue.log
fi

# ---------------------------------------------------------------- teacher (the 7B student itself, standard view) + one-step evidence
TEACHER_ARGS="$(cat $SF3/teacher_args.txt)"; s7log "R1 teacher args: [$TEACHER_ARGS] (standard view, 8 samples, top 2)"
if ! grep -q "DONE teacher (merged)" $D/teacher.log 2>/dev/null; then
  for i in $(seq 0 11); do NEED=$TEACH7 loop7 $D/teacher_s$i.log "DONE teacher" scripts/sciworld_stepfix.py teacher --model $MODEL --dir $D --adapter $STU7 --shard $i 12 --seed $R ${TEACHER_ARGS} & sleep 40; done; wait
  echo "DONE teacher (merged)" >> $D/teacher.log
fi
$P scripts/sciworld_stepfix.py select --dir $D --round $R | tee -a $SF7/queue.log
[ -s $D/replay.jsonl ] || $P scripts/sciworld_stepfix.py replay --dir $D --seed $R | tee -a $SF7/queue.log

# ---------------------------------------------------------------- train (StepFix + random control) and 848 evals
trainA() { local NM=$1; shift; NEED=$TRAIN7 loop7 $D/train_$NM.log "DONE train" scripts/sciworld_stepfix.py train --model $MODEL --data "$@" $D/replay.jsonl --adapter $STU7 --out $D/$NM --seed $R
  s7log "R1 7B $NM: trained ($(tail -1 $D/$NM/train_log.jsonl))"; }
evalA() { local NM=$1; eval7 $D/eval_$NM $D/$NM/adapter r1sw7_$NM
  s7log "R1 7B $NM 848: $(sum848 $D/eval_$NM/eval_test.jsonl $D/eval_$NM/eval_dev.jsonl)"
  $P scripts/sciworld_stepfix_compare.py --a R0_7B=$R0D/eval_test.jsonl,$R0D/eval_dev.jsonl --b R1_7B_$NM=$D/eval_$NM/eval_test.jsonl,$D/eval_$NM/eval_dev.jsonl | tee -a $SF7/queue.log; }
( trainA stepfix $D/targets_env_top.jsonl; evalA stepfix ) &
sleep 30; ( trainA random $D/targets_rand.jsonl; evalA random ) &
wait
$P scripts/sciworld_stepfix_compare.py --a R1_7B_random=$D/eval_random/eval_test.jsonl,$D/eval_random/eval_dev.jsonl --b R1_7B_stepfix=$D/eval_stepfix/eval_test.jsonl,$D/eval_stepfix/eval_dev.jsonl | tee -a $SF7/queue.log
$P scripts/sciworld_stepfix_compare.py --a r1b_3B=$SF3/r1b/eval_stepfix/eval_test.jsonl,$SF3/r1b/eval_stepfix/eval_dev.jsonl --b R1_7B_stepfix=$D/eval_stepfix/eval_test.jsonl,$D/eval_stepfix/eval_dev.jsonl | tee -a $SF7/queue.log
s7log "7B R1 DONE (Model A $MA7)"
