#!/bin/bash
# User-requested (Oct 8 ~22:35): 7B "rollout verification 20% + the same 100 single-step replay", R1 only. Same R0 (7B SFT), same R1 states,
# candidates and replay file as the 7B final-recipe R1 (a7b_final/r1, 549); only the filter differs: each candidate is verified by a full
# rollout of the FIXED 7B R0 policy (alfworld_attr_opd.py distill --verify_only), then K = 0.2 x failures random states among those with a
# winning candidate (target = the winner with the higher teacher prob), as in scripts/run_rollout20_chain.sh. 774 tag a7broll20_r1.
export MODEL=Qwen/Qwen2.5-7B-Instruct GPU_MIN=20000 TRAIN_MIN=36000
source scripts/lib_baselines.sh; F=$SA/a7b_final/r1; C=a7b_roll20; D=$SA/$C/r1; mkdir -p $D/self; T=$F/traj.jsonl; AD=$SA/a7b_final/r0/model/adapter; R=1
for f in ids_fail.jsonl pointer.jsonl pointer.raw.jsonl diag_modelA.raw.jsonl self/steps.jsonl self/targets.jsonl replay.jsonl; do [ -s $D/$f ] || cp $F/$f $D/$f; done
log "$C r1: start (states / candidates / replay = a7b_final/r1; rollout verification with the 7B R0 policy)"
for i in 0 1 2 3; do ( until grep -q "DONE distill" $D/rollout_s$i.log 2>/dev/null; do $(S $GPU_MIN) $P -u scripts/alfworld_attr_opd.py distill --traj $T --targets $D/self/targets.jsonl --student $MODEL --adapter $AD --out $D/rollout_s$i --verify_k 2 --exclude_a_minus --max_examples_per_run 250 --max_continue_steps 0 --verify_only --shard $i 4 >> $D/rollout_s$i.log 2>&1 || sleep 3; done ) & sleep 60; done; wait
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
print(f"=== rollout (a7b_roll20 r1): {len(tg)} states, {ncand} candidates rolled out, {nwin} won; {len(rows)} states with a winner; K = 0.2 x {nf} = {K}; kept {len(keep)}")
PY
train1 $D $AD "$D/targets_rollout/train.jsonl:1.0 $D/replay.jsonl:1.0"; [ -s $E/eval_heldout_a7broll20_r1.jsonl ] || eval774 $D a7broll20_r1
log "$C r1 DONE (a7broll20_r1; 7B final-recipe R1 = 549)"
