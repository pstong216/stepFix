#!/bin/bash
# One round of the SELF-ATTRIBUTION chain, v3: the ATTRIBUTOR IS IN THE LOOP and is updated generation by generation, from this
# round's own data only (lineage protocol, as Who&When / AgenTracer build their data from the agent's own successes):
#   round k: student S_{k-1} replays the 1500 train games
#            -> counterfactual failures: inject a typed error into S_{k-1}'s OWN wins, let S_{k-1} continue (1000 attempts)
#            -> env-evidence labels (step + type) on S_{k-1}'s failures (also used to measure the attributor's agreement)
#            -> attributor A_k = A_{k-1} continued on these labels (20% of the synthetic GAMES held out -> per-round accuracy)
#            -> A_k reads S_{k-1}'s failed logs: step t* + type  -> teacher (hindsight + A_k's diagnosis + own analysis)
#            -> two-layer filter (future decides the state, env one-step picks the action), top 86 + 86 replay -> S_k.
#   self_attr_round_v3.sh <chain> <round> <start adapter> <prev attributor adapter | M5 estimator> <hold job name>
# chains: selfattr (full) | selfattr_plain (A_k steps + plain teacher + env_top, no replay) | m5_informed (frozen M5 steps + informed
#         teacher + two-layer + replay) | envattr (env-evidence steps + informed teacher + two-layer + replay).
# Round 1 reuses the original b50 collection and the round-0 attributor.
set -u; CH=$1; R=$2; START=$3; PREV=$4; HOLD=$5
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
B=reports/alfworld/self_attr_rounds; D=$B/$CH${CHAIN_SUFFIX:-}/r$R; mkdir -p $D/self $D/proxy $D/envlab $D/attr_labels; M=$B/queue.log
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
hold() { if [ "$HOLD" = "any" ]; then local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge 10000 ] 2>/dev/null && echo $best; else squeue -u pan00389 -h -o "%i %j %T" | awk -v n="$HOLD" '$2==n && $3=="RUNNING"{print $1}' | sort -n | tail -n1; fi; }
SEED=${SEED:-0}
S() { local H; until H=$(hold); [ -n "$H" ]; do sleep 120; done; echo "srun --jobid=$H --overlap -n1 -c4 --mem=0"; }
guard() { until [ "$($(S) nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1)" -ge "$1" ] 2>/dev/null; do sleep 60; done; }
log "chain $CH${CHAIN_SUFFIX:-} round $R (v3, attributor in the loop): start (from $START, attributor/estimator $PREV)"
# 1. trajectories
if [ "$R" = "1" ]; then T=reports/alfworld/student3b_low100_train_b50.jsonl
else T=$D/traj.jsonl
  until tail -n 3 $D/collect.log 2>/dev/null | grep -q "^DONE"; do
    guard 10000; $(S) $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $START --split train --n_games 1500 --budget 50 --temperature 1.0 --out $T --tag ${CH}_r$R --max_episodes_per_run 450 >> $D/collect.log 2>&1 || sleep 60
  done
fi
NW=$(grep -c '"won": true' $T); NF=$(grep -c '"won": false' $T); log "chain $CH${CHAIN_SUFFIX:-} r$R: $(wc -l < $T) games, $NW won, $NF failed"
$P - <<PY
import json
eps=[json.loads(l) for l in open("$T")][:1000]
open("$D/ids_fail.jsonl","w").writelines(json.dumps({"id":e["id"]})+"\n" for e in eps if not e["won"])
PY
envlabels() {   # env-evidence attribution of this round's failures: GPU scoring of top-2 alternatives at the first 8 steps, CPU one-step (4 shards), label
  until grep -q "DONE score" $D/envlab/score.log 2>/dev/null; do guard 10000; $(S) $P -u scripts/attr_labels_env.py score --traj $T --ids $D/ids_fail.jsonl --out_dir $D/envlab --k 8 --adapter $START >> $D/envlab/score.log 2>&1 || sleep 30; done
  for i in 0 1 2 3; do ( until grep -q "DONE onestep" $D/envlab/onestep_s$i.log 2>/dev/null; do $P -u scripts/attr_labels_env.py onestep --traj $T --ids $D/ids_fail.jsonl --out_dir $D/envlab --k 8 --shard $i 4 >> $D/envlab/onestep_s$i.log 2>&1; sleep 3; done ) & done; wait
  [ -s $D/envlab/labels.jsonl ] || $P scripts/attr_labels_env.py label --traj $T --ids $D/ids_fail.jsonl --out_dir $D/envlab --k 8 2>&1 | tee -a $M; }
# budget: K_FRAC (e.g. 0.4) sets K = round(K_FRAC * number of failures this round); otherwise K (default 100)   [Oct 6 03:50]
if [ -n "${K_FRAC:-}" ]; then NFAIL=$(grep -c . $D/ids_fail.jsonl); K=$($P -c "print(max(1, round($K_FRAC * $NFAIL)))"); export K; log "chain $CH${CHAIN_SUFFIX:-} r$R: K = $K_FRAC x $NFAIL failures = $K"; fi
# 2.+3. step selection
case $CH in
  m5_informed)
    guard 10000; [ -s $D/m5_scores.jsonl ] || $(S) $P -u scripts/alfworld_progress_model.py score --traj $T --adapter $PREV --ids_from $D/ids_fail.jsonl --out $D/m5_scores.jsonl > $D/m5.log 2>&1
    $P scripts/alfworld_progress_model.py pointer --scores $D/m5_scores.jsonl --out $D/m5_pointer.jsonl --delta 1.0 >> $D/m5.log 2>&1
    $P - <<PY
import json
rows=[json.loads(l) for l in open("$D/m5_pointer.jsonl")]
open("$D/self/steps.jsonl","w").writelines(json.dumps({"id":r["id"],"steps":[r["t_hat"]],"mode":None})+"\n" for r in rows)
open("$D/diag.raw.jsonl","w").writelines(json.dumps({"id":r["id"],"t_hat":r["t_hat"],"type":None})+"\n" for r in rows)
PY
    ATTR=$PREV ;;
  envattr)
    envlabels; cp $D/envlab/steps.jsonl $D/self/steps.jsonl
    $P - <<PY
import json
rows=[json.loads(l) for l in open("$D/envlab/labels.jsonl")]
open("$D/diag.raw.jsonl","w").writelines(json.dumps({"id":r["id"],"t_hat":r["t_star"],"type":r.get("type")})+"\n" for r in rows)
PY
    ATTR=$PREV ;;
  *)   # selfattr / selfattr_plain (user decision Oct 4 22:50): step selection = M5 progress estimator, refreshed each round on the
       # round's own wins (no error injection, no learned log-reading attributor for now).  Round 1 keeps the A0 pointer already made.
    if [ -s $D/pointer.raw.jsonl ] || { [ "$R" = "1" ] && [ "${STEP:-}" != "logfix" ]; }; then ATTR=$PREV   # resume, or round 1 of the old (A0) chains; STEP=logfix (clean chains, Oct 5) uses the step-back log rule at every round
    else   # user decision (Oct 5 01:40): step = LOG-EVIDENCE RULE -- first step whose own action produced non-positive verifiable
           # evidence (new information / progress / null transition, read from the log).  Zero cost; no M5 refresh, no injection.
      $P - <<PYX
import json, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from compare_step_selectors import log_rule, log_fix, log_late, log_late2
import os
ctl = "$B/STEP_LATE_FROM_R2"   # control file (Oct 5 21:45): if present, rounds >= 2 use the LATE log rule; its content = window (default 3)
use_late = int("$R") >= 2 and os.path.exists(ctl) and "${STEP_FORCE:-}" != "logfix"   # STEP_FORCE=logfix: same-trajectory control
mode = open(ctl).read().strip() if use_late else ""   # control file content: "plateau:2" (generic change-point rule, default) or "take" (ALFWorld-specific: first move after the first take)
win = int(mode.split(":")[1]) if mode.startswith("plateau:") else 2
sel = (lambda e: log_late2(e, win)) if mode == "take" else (lambda e: log_late(e, win))
from compare_step_selectors import late_generic, first_nonpos_generic, taken_scores
if os.environ.get("EVIDENCE") == "generic":   # vocabulary-free selectors on the generic evidence (Oct 5 23:20)
    sel = lambda e: late_generic(taken_scores(e), win)
    log_fix = lambda e: first_nonpos_generic(taken_scores(e))
eps = {json.loads(l)["id"]: json.loads(l) for l in open("$T")}; ids = [json.loads(l)["id"] for l in open("$D/ids_fail.jsonl")]
rows = [{"id": i, "t_hat": (sel(eps[i]) if use_late else log_fix(eps[i])), "type": None} for i in ids if i in eps]   # step-back rule; LATE rule for R>=2 when the control file exists
print("selector:", ("late:" + mode) if use_late else "log_fix")
if os.environ.get("STEP_POOL") == "1":   # ENVIRONMENT ADJUDICATION (final method, Oct 6): propose BOTH the first-step and the stall-entry step;
    # teacher + one-step scan run on both, the filter (rule env_pool) keeps per failure the step with the largest verified improvement
    rows2 = []
    for i in ids:
        if i not in eps: continue
        sc = taken_scores(eps[i]); tf = first_nonpos_generic(sc); tl = late_generic(sc, 2)
        for t in sorted({tf, tl}): rows2.append({"id": i, "t_hat": t, "type": None})
    rows = rows2; print("pool proposals:", len(rows), "states for", len(set(r["id"] for r in rows)), "failures")
open("$D/pointer.jsonl", "w").writelines(json.dumps({"id": r["id"], "steps": [r["t_hat"]], "mode": None}) + "\n" for r in rows)
open("$D/pointer.raw.jsonl", "w").writelines(json.dumps(r) + "\n" for r in rows)
print("log-rule pointer:", len(rows), "states")
PYX
      ATTR=$([ "$R" -ge 2 ] && [ -f $B/STEP_LATE_FROM_R2 ] && echo logrule_late || echo logrule_fix)
    fi
    guard 10000; [ -s $D/pointer.raw.jsonl ] || $(S) $P -u scripts/alfworld_self_attributor.py pointer --traj $T --ids $D/ids_fail.jsonl --adapter $ATTR --out $D/pointer.jsonl > $D/pointer.log 2>&1
    cp $D/pointer.jsonl $D/self/steps.jsonl; cp $D/pointer.raw.jsonl $D/diag.raw.jsonl
    if [ "$R" != "1" ]; then $P - <<PY | tee -a $M
import json, collections
U={"wrong_location":"wrong_target","wrong_object":"wrong_target","misplace":"wrong_value","random":"wrong_op","misorder":"misorder","skip":"skip"}
held={json.loads(l)["id"]:json.loads(l) for l in open("$D/attr_labels/synth.heldout.jsonl")}; H=[json.loads(l) for l in open("$D/attr_held.raw.jsonl")]
env={json.loads(l)["id"]:json.loads(l) for l in open("$D/envlab/labels.jsonl")}; R=[json.loads(l) for l in open("$D/pointer.raw.jsonl")]
tt=[held[r["id"]]["t_star"] for r in H]; best_const=max(sum(abs(t-c)<=1 for t in tt) for c in range(0,30))/max(1,len(tt))
print(f"=== chain $CH${CHAIN_SUFFIX:-} r$R: attributor A$R held-out ({len(H)}): step acc {sum(r['t_hat']==held[r['id']]['t_star'] for r in H)/max(1,len(H)):.2f}, +-1 {sum(abs(r['t_hat']-held[r['id']]['t_star'])<=1 for r in H)/max(1,len(H)):.2f} (best constant {best_const:.2f}), type acc {sum((r.get('type') or '')==U.get(held[r['id']].get('type'),held[r['id']].get('type')) for r in H)/max(1,len(H)):.2f} | on real failures: +-1 agreement with env-evidence {sum(abs(r['t_hat']-env[r['id']]['t_star'])<=1 for r in R if r['id'] in env)/max(1,sum(r['id'] in env for r in R)):.2f}, type agreement {sum((r.get('type') or '')==env[r['id']].get('type') for r in R if r['id'] in env)/max(1,sum(r['id'] in env for r in R)):.2f}")
PY
    fi ;;
esac
PSTAT=$($P - <<PY
import json, statistics as st, collections
R=[json.loads(l) for l in open("$D/diag.raw.jsonl")]; ts=[r["t_hat"] for r in R]; ty=collections.Counter(r.get("type") for r in R)
print(f"{len(ts)} states, mean t {st.mean(ts):.1f}, t0 {sum(t==0 for t in ts)/len(ts):.2f}, t>=5 {sum(t>=5 for t in ts)/len(ts):.2f}, types {dict(ty.most_common(4))}")
PY
)
log "chain $CH${CHAIN_SUFFIX:-} r$R: pointer $PSTAT"
# 4. teacher candidates.  Diagnosis for the informed teacher = Model A (external 7B attributor, user decision Oct 4) when a release
#    file exists; run its inference on this round's trajectories if not done yet.
if [ "$CH" != "selfattr_plain" ] && [ "${TEACHER_NO_DIAG:-0}" != "1" ] && [ -s ${T%.jsonl}.release.jsonl ] && [ ! -s $D/diag_modelA.raw.jsonl ]; then   # TEACHER_NO_DIAG=1 (final recipe, Oct 5 16:30): no Model A
  until grep -q "DONE" $D/modelA.log 2>/dev/null; do guard 18000; $(S) $P -u scripts/alfworld_attribute.py --attributor ${MODELA_PATH:-reports/attributor_7b_no_observations/final} --variant no_observations --release ${T%.jsonl}.release.jsonl --out $D/modelA_pred.jsonl >> $D/modelA.log 2>&1 && echo DONE >> $D/modelA.log || sleep 60; done
  $P -c "import json; rows=[json.loads(l) for l in open('$D/modelA_pred.jsonl')]; open('$D/diag_modelA.raw.jsonl','w').writelines(json.dumps({'id':r['id'],'t_hat':r['t_hat'],'type':'Model-A mode '+str(r['pred'].get('error_mode'))})+'\n' for r in rows); print('Model A diagnoses:', len(rows))"
fi
case $CH in selfattr_plain) TOPT="--no_hind --no_diag --no_analysis" ;; *) TOPT="${TEACHER_NO_DIAG:+--no_diag}" ;; esac   # final teacher = hindsight + own analysis
DIAG=$D/diag.raw.jsonl; [ -s $D/diag_modelA.raw.jsonl ] && DIAG=$D/diag_modelA.raw.jsonl   # user decision (Oct 4): teacher = hindsight + Model A diagnosis + own analysis when available
if [ "${PAR:-1}" -gt 1 ] && ! grep -q "DONE informed_teacher" $D/teacher.log 2>/dev/null; then   # PAR shards on separate GPUs (Oct 6 14:45)
  for i in $(seq 0 $((PAR-1))); do ( until grep -q "DONE informed_teacher" $D/teacher_s$i.log 2>/dev/null; do guard 12000; $(S) $P -u scripts/self_teacher_informed.py --traj $T --steps $D/self/steps.jsonl --diag $DIAG --adapter $START --out $D/self/targets_s$i.jsonl --shard $i $PAR $TOPT >> $D/teacher_s$i.log 2>&1 || sleep 30; done ) & sleep 20; done; wait
  cat $D/self/targets_s*.jsonl >> $D/self/targets.jsonl; echo "DONE informed_teacher (merged $PAR shards)" >> $D/teacher.log
fi
until grep -q "DONE informed_teacher" $D/teacher.log 2>/dev/null; do
  guard 12000; $(S) $P -u scripts/self_teacher_informed.py --traj $T --steps $D/self/steps.jsonl --diag $DIAG --adapter $START --out $D/self/targets.jsonl $TOPT >> $D/teacher.log 2>&1 || sleep 30; done
# 5. one-step evidence for the teacher's candidates (CPU) and the filter
case $CH in selfattr_rollout|selfattr_placebo|selfattr_rollsub) ;; *) if [ "${PAR:-1}" -gt 1 ] && ! grep -q "DONE onestep" $D/onestep.log 2>/dev/null; then NS=$((PAR+1)); for i in $(seq 0 $((NS-1))); do ( until grep -q "DONE onestep" $D/onestep_s$i.log 2>/dev/null; do SE_DIR=$D SE_TRAJ=$T $P -u scripts/proxy_eval_se3b.py onestep --arms self --limit 300 --shard $i $NS >> $D/onestep_s$i.log 2>&1; sleep 3; done ) & done; wait; cat $D/proxy/onestep_s*.jsonl >> $D/proxy/onestep.jsonl; rm -f $D/proxy/onestep_s*.jsonl; echo "DONE onestep (merged $NS shards)" >> $D/onestep.log; fi
     until grep -q "DONE onestep" $D/onestep.log 2>/dev/null; do SE_DIR=$D SE_TRAJ=$T $P -u scripts/proxy_eval_se3b.py onestep --arms self --limit 300 >> $D/onestep.log 2>&1; sleep 3; done ;; esac
case $CH in
  selfattr_rollout)   # control: same steps, teacher, replay; filter = rollout verification (student continuation <=20 steps, keep winners)
    RULE=rollout; until grep -q "DONE distill" $D/distill.log 2>/dev/null; do guard 24000; $(S) $P -u scripts/alfworld_attr_opd.py distill --traj $T --targets $D/self/targets.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter $START --out $D/model_rollout --verify_k 2 --exclude_a_minus --max_examples_per_run 250 --max_continue_steps ${ROLL_H:-0} >> $D/distill.log 2>&1 || sleep 30; done
    mkdir -p $D/targets_rollout; $P -c "
import json; tg={(json.loads(l)['id'],json.loads(l)['t']):json.loads(l) for l in open('$D/self/targets.jsonl')}; n=0
with open('$D/targets_rollout/train.jsonl','w') as f:
    for r in map(json.loads, open('$D/model_rollout/examples.jsonl')):
        T=tg[(r['id'],r['t'])]; cmds=T['admissible']; wins=[c for c,w in r['alternatives'].items() if w and c in cmds]
        if not wins: continue
        f.write(json.dumps({'kind':'kl','src':'$T','id':r['id'],'t':r['t'],'cmds':cmds,'target':[1.0/len(wins) if c in wins else 0.0 for c in cmds]})+'\n'); n+=1
print('=== proxy set self/rollout: %d verified targets' % n)" | tee -a $M ;;
  selfattr_rollsub)   # control (user Oct 5 17:20): rollout verification, then a RANDOM subset of K winners (equal budget to env top-K).
    RULE=rollsub; mkdir -p $D/targets_rollsub
    if [ "$R" = "1" ]; then SRCV=$B/selfattr_rollout_hind/r1/targets_rollout/train.jsonl; until [ -s $SRCV ]; do sleep 300; done   # R1: same candidates -> reuse the rollout control's verification
    else until grep -q "DONE distill" $D/distill.log 2>/dev/null; do guard 24000; $(S) $P -u scripts/alfworld_attr_opd.py distill --traj $T --targets $D/self/targets.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter $START --out $D/model_rollout --verify_k 2 --exclude_a_minus --max_examples_per_run 250 --max_continue_steps ${ROLL_H:-0} >> $D/distill.log 2>&1 || sleep 30; done
      SRCV=$D/targets_rollout_all.jsonl; $P -c "
import json; tg={(json.loads(l)['id'],json.loads(l)['t']):json.loads(l) for l in open('$D/self/targets.jsonl')}
with open('$SRCV','w') as f:
    for r in map(json.loads, open('$D/model_rollout/examples.jsonl')):
        T=tg[(r['id'],r['t'])]; cmds=T['admissible']; wins=[c for c,w in r['alternatives'].items() if w and c in cmds]
        if wins: f.write(json.dumps({'kind':'kl','src':'$T','id':r['id'],'t':r['t'],'cmds':cmds,'target':[1.0/len(wins) if c in wins else 0.0 for c in cmds]})+'\n')"; fi
    $P -c "
import json, random; rng=random.Random($R*100+$SEED); rows=[l for l in open('$SRCV')]; rng.shuffle(rows); open('$D/targets_rollsub/train.jsonl','w').writelines(rows[:${K:-100}])
print('=== proxy set self/rollsub: %d of %d rollout winners (random subset, K=${K:-100})' % (min(${K:-100},len(rows)), len(rows)))" | tee -a $M ;;
  selfattr_eqv)   # control (user Oct 5 17:20): env one-step ranking with K = number of rollout winners in the SAME round of the rollout control chain.
    RULE=env_top; until [ -s $B/selfattr_rollout_hind/r$R/targets_rollout/train.jsonl ]; do sleep 300; done; NV=$(wc -l < $B/selfattr_rollout_hind/r$R/targets_rollout/train.jsonl)
    SE_DIR=$D SE_TRAJ=$T $P scripts/proxy_arms_se3b.py --arm self --rule $RULE --k $NV --out $D/targets_$RULE | tee -a $M; log "chain $CH${CHAIN_SUFFIX:-} r$R: env top-$NV (equal to rollout winners this round)" ;;
  selfattr_placebo)   # control: same steps, teacher, replay; filter = 86 random states, uniformly random action != a-
    RULE=placebo; mkdir -p $D/targets_placebo; $P -c "
import json, random; rng=random.Random($R*100+$SEED); tg=[json.loads(l) for l in open('$D/self/targets.jsonl')]; rng.shuffle(tg)
with open('$D/targets_placebo/train.jsonl','w') as f:
    for T in tg[:${K:-100}]:
        cmds=T['admissible']; c=rng.choice([x for x in cmds if x!=T['a_minus']])
        f.write(json.dumps({'kind':'kl','src':'$T','id':T['id'],'t':T['t'],'cmds':cmds,'target':[1.0 if x==c else 0.0 for x in cmds]})+'\n')
print('=== proxy set self/placebo: ${K:-100} random states, random action')" | tee -a $M ;;
  *) RULE=env_top; [ "${STEP_POOL:-0}" = "1" ] && RULE=env_pool; SE_DIR=$D SE_TRAJ=$T $P scripts/proxy_arms_se3b.py --arm self --rule $RULE --k ${K:-100} --out $D/targets_$RULE | tee -a $M ;;
esac
DATA="$D/targets_$RULE/train.jsonl:1.0"
if [ "$CH" != "selfattr_plain" ]; then $P scripts/make_replay.py --traj $T --n ${REPLAY_N:-${K:-100}} --seed $R --out $D/replay.jsonl | tee -a $M; DATA="$DATA $D/replay.jsonl:1.0"; fi
# 6. train, 7. eval
until [ -f $D/model/adapter/adapter_config.json ]; do guard 24000; $(S) $P -u scripts/alfworld_opd_plus.py train --data $DATA --student Qwen/Qwen2.5-3B-Instruct --adapter $PWD/$START --lr 1e-5 --epochs 1 --batch 8 --seed $SEED --out $D/model > $D/train.log 2>&1 || { log "chain $CH${CHAIN_SUFFIX:-} r$R: TRAIN FAILED (retrying in 3 min)"; sleep 180; }; done
ADAPTER=$D/model/adapter
for sp in valid_unseen valid_seen; do ( until [ "$(grep -c . $D/eval_${sp#valid_}.jsonl 2>/dev/null)" -ge $([ $sp = valid_unseen ] && echo 134 || echo 140) ]; do
  guard 10000; $(S) $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $ADAPTER --split $sp --budget 50 --temperature 0 --out $D/eval_${sp#valid_}.jsonl --tag ${CH}_r$R >> $D/eval_${sp#valid_}.log 2>&1 || sleep 30; done ) & sleep 20; done; wait   # unseen and seen in parallel (Oct 6 14:45)
echo "$ADAPTER" > $D/ADAPTER; echo "$ATTR" > $D/ATTR
log "CHAIN $CH${CHAIN_SUFFIX:-} R$R: train-game wins $NW/$((NW+NF)) | pointer $PSTAT | $(wc -l < $D/targets_$RULE/train.jsonl) targets ($RULE) | unseen $(grep -c '"won": true' $D/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen.jsonl)/140 (student 56/134, 80/140; env chain R1 167, R2 178, R3 184)"
