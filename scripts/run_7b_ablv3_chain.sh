#!/bin/bash
# Oct 10 ~03:30 (user-requested overnight, via main): the two 7B ablations re-run at R2/R3 with the NEW recipe of the main 7B chain
# (scripts/run_7b_chain_v3b.sh, a7bv3b_r2 576 / a7bv3b_r3 619), so that they are comparable with it. Copied from run_7b_chain_v3b.sh; only the
# selection step differs.
#   bash scripts/run_7b_ablv3_chain.sh rand   -> chain a7b_randv3, R1 = a7brand20_r1 (508), tags a7brandv3_r2 / a7brandv3_r3
#   bash scripts/run_7b_ablv3_chain.sh roll   -> chain a7b_rollv3, R1 = a7broll20_r1 (524), tags a7brollv3_r2 / a7brollv3_r3
# Every round: its own 1500 train games collected by its own start policy (R2 reuses the 1500-game collection the OLD-recipe ablation
# already made with the same R1 adapter, a7b_rand20/r2 or a7b_roll20/r2 -- the collection does not depend on the recipe; R3 collects anew
# with the new R2 model) -> 7B Model A (attributor_7b_no_observations) step on the failures among the first 1000 -> hindsight self-teacher
# with --full_view (adapter = the round's start policy) -> selection:
#   rand: K = round(0.2 x failures) RANDOM teacher top-1 targets (rule all_top1, a- excluded; shuffled with random.Random(100*R), as rand20
#         in scripts/run_k20_round.sh); no evidence, no margin. (One-step evidence is still computed, only for diagnostics.)
#   roll: each of the teacher's 2 candidates (a- excluded) verified by a full rollout of the round's FIXED start policy
#         (alfworld_attr_opd.py distill --verify_only, as scripts/run_7b_roll20_r23.sh); K random states among those with a winner
#         (random.Random(100*R)), target = the winner with the higher teacher prob.
# GPUs (main, Oct 10 03:25): the a7b_gpu jobs plus the error_attr_hold_* jobs.
# -> fail_weighted replay (100, seed R) -> same trainer (CKPT_EVERY=50 resumable) -> 774 eval.
MODE=$1; case $MODE in rand|roll) ;; *) echo "usage: $0 rand|roll"; exit 2;; esac
export MODEL=Qwen/Qwen2.5-7B-Instruct GPU_MIN=20000 TRAIN_MIN=36000 EVIDENCE=generic HOLD_NAME_RE='^(a7b_gpu|error_attr_hold_.*)$' CKPT_EVERY=50
source scripts/lib_baselines.sh; MA=reports/attributor_7b_no_observations/final
# Oct 10 03:35: a running 7B training (e.g. the RFT R3 training) holds only ~20 GB between its peaks, so GPU_MIN=20000 alone did NOT keep
# this driver off its card (a teacher shard landed next to it). Local override of pick(): never use a (job, node) on which ANY
# alfworld_opd_plus.py train process (launched from this node) is running. lib_baselines.sh itself is unchanged.
# Oct 10 03:50: the 7B --full_view teacher grows to ~30 GB, so two processes on one 40 GB card OOMed each other: the teacher now needs
# 32 GB free, and no other process is placed on a card that runs a full-view teacher.
_pick_raw() { local best="" bm=0 busy; busy=$(pgrep -af "srun --jobid=[0-9]+ .*(alfworld_opd_plus.py train|self_teacher_informed.py.*--full_view)" | sed -nE 's/.*--jobid=([0-9]+)( .* -w ([^ ]+))? .*/\1:\3/p')
  for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk -v re="${HOLD_NAME_RE:-^error_attr_hold_}" '$2 ~ re && $3=="RUNNING"{print $1}'); do for N in $(scontrol show hostnames $(squeue -h -j $H -o %N) 2>/dev/null); do
    echo "$busy" | grep -qxE "$H:($N)?" && continue
    [ -e $CL/${H}_$N ] && [ $(( $(date +%s) - $(stat -c %Y $CL/${H}_$N) )) -lt 90 ] && continue
    local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 -w $N nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best="$H $N"; }; done; done; [ "$bm" -ge ${1:-12000} ] 2>/dev/null && echo $best; }
# Oct 10 04:05: two drivers polling at the same minute both picked the same empty card (OOM). Picks are now serialized with flock, and a
# card just handed out is skipped for 90 s (until its process has loaded and shows up in nvidia-smi).
CL=$SA/.ablv3_claims; mkdir -p $CL
# Oct 10 04:40: priority (task 1 = rand before task 2 = roll). While any rand process waits for a GPU it touches a flag; the roll driver
# does not take a card while that flag is fresh (< 90 s), so rand's trainings / evals are not starved by roll's rollouts.
pick() { local PF=/dev/shm/ablv3_rand_waiting; if [ $MODE = rand ]; then touch $PF; elif [ -e $PF ] && [ $(( $(date +%s) - $(stat -c %Y $PF) )) -lt 90 ]; then return; fi
  ( flock 9; local r; r=$(_pick_raw "$@"); [ -n "$r" ] && touch $CL/${r// /_}; echo $r ) 9>/dev/shm/ablv3_pick.lock; }
if [ $MODE = rand ]; then C=$SA/a7b_randv3; OLD=$SA/a7b_rand20; else C=$SA/a7b_rollv3; OLD=$SA/a7b_roll20; fi
vround() { local R=$1 D=$C/r$1 SRC=$2 START=$3 TAG=a7b${MODE}v3_r$1; mkdir -p $D/self $D/proxy; local T=$SRC/traj.jsonl
  [ -f $D/model/adapter/adapter_config.json ] && [ -s $E/eval_heldout_$TAG.jsonl ] && return
  log "a7b_${MODE}v3 r$R: start on $(hostname) (from $START, games from $SRC)"
  if [ "$SRC" = "$D" ]; then collect4 $D $START a7b_${MODE}v3_r$R; fi
  $P -c "import json; eps=[json.loads(l) for l in open('$T')][:1000]; open('$D/ids_fail.jsonl','w').writelines(json.dumps({'id':e['id']})+'\n' for e in eps if not e['won'])"
  log "a7b_${MODE}v3 r$R: $(grep -c . $T) games, $(grep -c '"won": true' $T) won; $(grep -c . $D/ids_fail.jsonl) failures in the first 1000 (from $START)"
  until grep -q DONE $D/modelA.log 2>/dev/null; do $(S 22000) $P -u scripts/alfworld_attribute.py --attributor $MA --variant no_observations --release ${T%.jsonl}.release.jsonl --out $D/modelA_pred.jsonl >> $D/modelA.log 2>&1 && echo DONE >> $D/modelA.log || sleep 60; done
  [ -s $D/pointer.raw.jsonl ] || $P - <<PY | tee -a $SA/queue.log
import json
D="$D"; ids={json.loads(l)["id"] for l in open(f"{D}/ids_fail.jsonl")}; eps={json.loads(l)["id"]:json.loads(l) for l in open("$T")}
pred={r["id"]:r for r in map(json.loads, open(f"{D}/modelA_pred.jsonl"))}; rows=[]
for i in ids:
    if i not in pred or i not in eps: continue
    t=max(0, min(int(pred[i]["t_hat"] if pred[i]["t_hat"] is not None else 0), len(eps[i]["steps"])-1)); rows.append({"id":i,"t_hat":t,"type":None})
open(f"{D}/pointer.jsonl","w").writelines(json.dumps({"id":r["id"],"steps":[r["t_hat"]],"mode":None})+"\n" for r in rows)
open(f"{D}/pointer.raw.jsonl","w").writelines(json.dumps(r)+"\n" for r in rows)
open(f"{D}/diag_modelA.raw.jsonl","w").writelines(json.dumps({"id":r["id"],"t_hat":pred[r["id"]]["t_hat"],"type":"Model-A mode "+str((pred[r["id"]]["pred"] or {}).get("error_mode"))})+"\n" for r in rows)
ts=[r["t_hat"] for r in rows]; print("=== a7b_${MODE}v3 r$R 7B Model A pointer: %d states, mean t %.2f, t0 %.2f" % (len(rows), sum(ts)/len(ts), sum(t==0 for t in ts)/len(ts)))
PY
  cp $D/pointer.jsonl $D/self/steps.jsonl
  if ! grep -q "DONE informed_teacher" $D/teacher.log 2>/dev/null; then
    for i in 0 1 2; do ( until grep -q "DONE informed_teacher" $D/teacher_s$i.log 2>/dev/null; do $(S 32000) $P -u scripts/self_teacher_informed.py --traj $T --steps $D/self/steps.jsonl --diag $D/diag_modelA.raw.jsonl --adapter $START --model $MODEL --out $D/self/targets_s$i.jsonl --shard $i 3 --full_view >> $D/teacher_s$i.log 2>&1 || sleep 30; done ) & sleep 60; done; wait
    cat $D/self/targets_s*.jsonl > $D/self/targets.jsonl; echo "DONE informed_teacher (merged 3 shards)" >> $D/teacher.log
  fi
  $P -c "
import json; S={(json.loads(l)['id'],json.loads(l)['steps'][0]) for l in open('$D/self/steps.jsonl')}; R=[(json.loads(l)['id'],json.loads(l)['t']) for l in open('$D/self/targets.jsonl')]
print('=== $TAG teacher coverage: states', len(S), 'covered', len(set(R)&S), 'rows', len(R)); assert set(R)==S and len(R)==len(S), 'TEACHER COVERAGE INCOMPLETE'" | tee -a $SA/queue.log || exit 1
  local NF=$(grep -c . $D/ids_fail.jsonl); local K=$($P -c "print(max(1, round(0.2 * $NF)))")
  if [ $MODE = rand ]; then
    if ! grep -q "DONE onestep" $D/onestep.log 2>/dev/null; then   # diagnostics only (the random rule does not use it)
      for i in $(seq 0 11); do ( until grep -q "DONE onestep" $D/onestep_s$i.log 2>/dev/null; do SE_DIR=$D SE_TRAJ=$T $P -u scripts/proxy_eval_se3b.py onestep --arms self --limit 300 --shard $i 12 >> $D/onestep_s$i.log 2>&1; sleep 3; done ) & done; wait
      cat $D/proxy/onestep_s*.jsonl >> $D/proxy/onestep.jsonl 2>/dev/null; rm -f $D/proxy/onestep_s*.jsonl
      until SE_DIR=$D SE_TRAJ=$T $P -u scripts/proxy_eval_se3b.py onestep --arms self --limit 5000 > $D/onestep_final.log 2>&1 && grep -q "DONE onestep" $D/onestep_final.log; do sleep 3; done
      echo "DONE onestep" >> $D/onestep.log
    fi
    [ -s $D/targets_all_top1/train.jsonl ] || SE_DIR=$D SE_TRAJ=$T $P scripts/proxy_arms_se3b.py --arm self --rule all_top1 --out $D/targets_all_top1 | tee -a $SA/queue.log
    [ -s $D/targets_env_top_margin/train.jsonl ] || SE_DIR=$D SE_TRAJ=$T $P scripts/proxy_arms_se3b.py --arm self --rule env_top_margin --k $K --out $D/targets_env_top_margin | tee -a $SA/queue.log   # diagnostics only: overlap with the main rule
    mkdir -p $D/targets_rand20
    [ -s $D/targets_rand20/train.jsonl ] || $P -c "import random; L=open('$D/targets_all_top1/train.jsonl').readlines(); random.Random(100 * $R).shuffle(L); open('$D/targets_rand20/train.jsonl','w').writelines(L[:$K])"
    log "a7b_randv3 r$R: random 20% = $(grep -c . $D/targets_rand20/train.jsonl) of $(grep -c . $D/targets_all_top1/train.jsonl) teacher top-1 targets (K = 0.2 x $NF)"
    SEL=$D/targets_rand20/train.jsonl
  else
    for i in 0 1 2 3; do ( until grep -q "DONE distill" $D/rollout_s$i.log 2>/dev/null; do $(S $GPU_MIN) $P -u scripts/alfworld_attr_opd.py distill --traj $T --targets $D/self/targets.jsonl --student $MODEL --adapter $START --out $D/rollout_s$i --verify_k 2 --exclude_a_minus --max_examples_per_run 250 --max_continue_steps 0 --verify_only --shard $i 4 >> $D/rollout_s$i.log 2>&1 || sleep 3; done ) & sleep 60; done; wait
    cat $D/rollout_s{0,1,2,3}/examples.jsonl > $D/rollout_examples.jsonl; mkdir -p $D/targets_rollout
    [ -s $D/targets_rollout/train.jsonl ] || $P - <<PY | tee -a $SA/queue.log
import json, random
D="$D"; tg={(json.loads(l)['id'],json.loads(l)['t']):json.loads(l) for l in open(f"{D}/self/targets.jsonl")}
nf=sum(1 for _ in open(f"{D}/ids_fail.jsonl")); K=max(1, round(0.2*nf)); rows=[]; ncand=0; nwin=0; seen=set()
for r in map(json.loads, open(f"{D}/rollout_examples.jsonl")):
    k=(r['id'],r['t'])
    if k in seen: continue
    seen.add(k); T=tg[k]; cmds=T['admissible']; alts=r.get('alternatives') or {}; ncand+=len(alts)
    wins=[c for c,w in alts.items() if w and c in cmds]; nwin+=len(wins)
    if not wins: continue
    c=max(wins, key=lambda c: T['teacher_logprobs'][c])
    rows.append({'kind':'kl','src':'$T','id':r['id'],'t':r['t'],'cmds':cmds,'target':[1.0 if x==c else 0.0 for x in cmds]})
random.Random($R*100).shuffle(rows); keep=rows[:K]
open(f"{D}/targets_rollout/train.jsonl","w").writelines(json.dumps(x)+"\n" for x in keep)
print(f"=== rollout (a7b_rollv3 r$R): {len(tg)} states, {len(seen)} rolled out, {ncand} candidates, {nwin} won; {len(rows)} states with a winner; K = 0.2 x {nf} = {K}; kept {len(keep)}")
PY
    SEL=$D/targets_rollout/train.jsonl
  fi
  [ -s $D/replay.jsonl ] || $P scripts/make_replay.py --traj $T --n 100 --seed $R --mode fail_weighted --out $D/replay.jsonl | tee -a $SA/queue.log
  train1 $D $START "$SEL:1.0 $D/replay.jsonl:1.0"; [ -s $E/eval_heldout_$TAG.jsonl ] || eval774 $D $TAG
  [ -f $D/model/adapter/adapter_config.json ] || { log "a7b_${MODE}v3 r$R ended WITHOUT a model"; exit 1; }
  log "a7b_${MODE}v3 r$R DONE ($TAG)"; }
vround 2 $OLD/r2 $OLD/r1/model/adapter
vround 3 $C/r3 $C/r2/model/adapter
log "a7b_${MODE}v3 R2->R3 DONE (new recipe: 7B Model A + full-view teacher + fail_weighted replay; selection = $MODE)"
