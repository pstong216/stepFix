#!/bin/bash
# Rebuild the ablation arms from the CLEAN round-1 candidates (step-back log rule steps, informed teacher = hindsight + Model A +
# own analysis), so every finalized experiment shares one candidate set.  Waits for the clean seed-0 R1 one-step scan, then:
#   filters:  env_top 86 / k103 / k205 / k308, placebo, hind_top (teacher confidence), fut_top (hindsight audit), all_top1 (no filter),
#             env_filter_all (veto only)
#   B-set:    after the rollout-control clean R1 verification: V = rollout-verified, B = env-top-86 states rejected by rollout,
#             R = same number of random rollout-rejected states with the teacher's top candidate; arms V, B, V+B, V+R
#   teachers: hindsight-only (no diagnosis) and plain (no hindsight, no diagnosis) teachers on the same steps -> env_top 86
# Queue: reports/alfworld/rigorous/queue_clean.txt (arms prefixed clean_), run by run_rigorous_arms_v3.sh.
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3; SA=reports/alfworld/self_attr_rounds; C=$SA/selfattr_clean/r1
T=reports/alfworld/student3b_low100_train_b50.jsonl; R=reports/alfworld/rigorous; Q=$R/queue_clean.txt; M=$SA/queue.log; L=$R/data/clean_build.log
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
until grep -q "DONE onestep" $C/onestep.log 2>/dev/null; do sleep 120; done
mkdir -p $R/data; : > $Q
for spec in "env_top 86 clean_env_top86" "env_top 103 clean_env_top_k103_25pct" "env_top 205 clean_env_top_k205_50pct" "env_top 308 clean_env_top_k308_75pct" "placebo 86 clean_placebo86" "hind_top 86 clean_teacherconf86" "fut_top 86 clean_future86" "all_top1 410 clean_all410" "env_filter_all 410 clean_veto_only"; do
  set -- $spec; rule=$1; k=$2; arm=$3; out=$C/arm_$arm
  [ -s $out/train.jsonl ] || SE_DIR=$C SE_TRAJ=$T $P scripts/proxy_arms_se3b.py --arm self --rule $rule --k $k --out $out >> $L 2>&1
  [ -s $out/train.jsonl ] && printf "%s\t%s:1.0\n" "$arm" "$out/train.jsonl" >> $Q || log "clean arm $arm: BUILD FAILED (rule $rule)"
done
log "clean filter arms built: $(cut -f1 $Q | tr '\n' ' ')"
tmux new-session -d -s rig_clean1 "RQ=$Q SEEDS='1' bash scripts/run_rigorous_arms_v3.sh"; sleep 90; tmux new-session -d -s rig_clean2 "RQ=$Q SEEDS='1' bash scripts/run_rigorous_arms_v3.sh"
# teacher-input ablations on the same steps (GPU teacher passes), then env_top 86
pick() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge 14000 ] 2>/dev/null && echo $best; }
for spec in "nodiag --no_diag clean_teacher_hind_only" "plain --no_diag_--no_hind clean_teacher_plain"; do
  set -- $spec; name=$1; flags=${2//_/ }; arm=$3; D=$C/teacher_$name; mkdir -p $D/self $D/proxy; cp $C/self/steps.jsonl $D/self/
  until grep -q "DONE informed_teacher" $D/teacher.log 2>/dev/null; do H=""; until H=$(pick); [ -n "$H" ]; do sleep 60; done
    srun --jobid=$H --overlap -n1 -c4 --mem=0 $P -u scripts/self_teacher_informed.py --traj $T --steps $D/self/steps.jsonl --diag $C/diag_modelA.raw.jsonl --adapter reports/alfworld/student3b_low100/final $flags --out $D/self/targets.jsonl >> $D/teacher.log 2>&1 || sleep 60; done
  until grep -q "DONE onestep" $D/onestep.log 2>/dev/null; do SE_DIR=$D SE_TRAJ=$T $P -u scripts/proxy_eval_se3b.py onestep --arms self --limit 300 >> $D/onestep.log 2>&1; sleep 3; done
  SE_DIR=$D SE_TRAJ=$T $P scripts/proxy_arms_se3b.py --arm self --rule env_top --k 86 --out $D/arm >> $L 2>&1
  [ -s $D/arm/train.jsonl ] && printf "%s\t%s:1.0\n" "$arm" "$D/arm/train.jsonl" >> $Q && log "clean teacher arm $arm queued" || log "clean teacher arm $arm: BUILD FAILED"
done
# B-set from the rollout-control clean R1 (needs its verification pass)
RO=$SA/selfattr_rollout_clean/r1
until grep -q "DONE distill" $RO/distill.log 2>/dev/null && [ -s $RO/targets_rollout/train.jsonl ]; do sleep 180; done
$P - <<PY | tee -a $M
import json, random
C="$C"; RO="$RO"; T="$T"; rng=random.Random(0)
tg={(r["id"],r["t"]):r for r in map(json.loads,open(f"{C}/self/targets.jsonl"))}
V=[json.loads(l) for l in open(f"{RO}/targets_rollout/train.jsonl")]; ver={(r["id"],r["t"]) for r in V}
env=[json.loads(l) for l in open(f"{C}/arm_clean_env_top86/train.jsonl")]
B=[r for r in env if (r["id"],r["t"]) not in ver]
rejected=[k for k in tg if k not in ver and k not in {(r["id"],r["t"]) for r in B}]
rng.shuffle(rejected); Rrows=[]
for (i,t) in rejected[:len(B)]:
    r=tg[(i,t)]; cmds=r["admissible"]; lp=r["teacher_logprobs"]; cand=max((c for c in cmds if c!=r["a_minus"]), key=lambda c: lp.get(c,-1e9))
    Rrows.append({"kind":"kl","src":T,"id":i,"t":t,"cmds":cmds,"target":[1.0 if c==cand else 0.0 for c in cmds],"cand":cand})
open(f"{C}/B_env_kept_rollout_rejected.jsonl","w").writelines(json.dumps(r)+"\n" for r in B)
open(f"{C}/R_random_rollout_rejected.jsonl","w").writelines(json.dumps(r)+"\n" for r in Rrows)
print(f"=== clean B-set: candidates {len(tg)} | V rollout-verified {len(V)} | env-top86 rejected by rollout (B) {len(B)} | random rejected (R) {len(Rrows)}")
PY
{ printf "clean_V_verified\t%s:1.0\n" "$RO/targets_rollout/train.jsonl"; printf "clean_B_alone\t%s:1.0\n" "$C/B_env_kept_rollout_rejected.jsonl"
  printf "clean_V_plus_B\t%s:1.0 %s:1.0\n" "$RO/targets_rollout/train.jsonl" "$C/B_env_kept_rollout_rejected.jsonl"; printf "clean_V_plus_R\t%s:1.0 %s:1.0\n" "$RO/targets_rollout/train.jsonl" "$C/R_random_rollout_rejected.jsonl"; } >> $Q
log "clean B-set arms queued (V, B, V+B, V+R)"
