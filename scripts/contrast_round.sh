#!/bin/bash
# One round of the CONTRAST chain: self-imitation at the divergence step.  The round's student replays the 1500 training games;
# every failure is paired with a won trajectory of the SAME game from ANY collection so far (earlier rounds, other chains, this
# round); the first step where the two action sequences diverge is the error step (identical prefix => identical state), and the
# win's action there is the training target (one-hot).  No environment replay, no teacher, no rollout.
#   contrast_round.sh <round> <start adapter>          (GPU hold = whichever has most free memory; CPU pairing on this node)
set -u; R=$1; START=$2
cd /users/8/pan00389/multi-agent/error-attribution; P=/users/8/pan00389/miniconda3/envs/error_attr/bin/python3
B=reports/alfworld/self_attr_rounds; D=$B/contrast/r$R; mkdir -p $D; M=$B/queue.log; RB=reports/alfworld/self_evolve3b_rounds
export ERROR_ATTR_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "=== $(date '+%F %T') $*" | tee -a $M; }
pick() { local best="" bm=0; for H in $(squeue -u pan00389 -h -o "%i %j %T" | awk '$2 ~ /^error_attr_hold_/ && $3=="RUNNING"{print $1}'); do local m=$(srun --jobid=$H --overlap -n1 -c1 --mem=0 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1); [ "${m:-0}" -gt "$bm" ] 2>/dev/null && { bm=$m; best=$H; }; done; [ "$bm" -ge "${1:-10000}" ] 2>/dev/null && echo $best; }
S() { local H=""; until H=$(pick $1); [ -n "$H" ]; do sleep 60; done; echo "srun --jobid=$H --overlap -n1 -c4 --mem=0"; }
log "chain contrast round $R: start (from $START)"
T=$D/traj.jsonl
until tail -n 3 $D/collect.log 2>/dev/null | grep -q "^DONE"; do
  $(S 10000) $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $START --split train --n_games 1500 --budget 50 --temperature 1.0 --out $T --tag contrast_r$R --max_episodes_per_run 450 >> $D/collect.log 2>&1 || sleep 60
done
NW=$(grep -c '"won": true' $T); NF=$(grep -c '"won": false' $T); log "chain contrast r$R: $(wc -l < $T) games, $NW won, $NF failed"
# pairing: this round's failures vs wins from every collection so far
ALL="$T reports/alfworld/student3b_low100_train_b50.jsonl $(ls $RB/*/r*/traj.jsonl $B/*/r*/traj.jsonl 2>/dev/null | grep -v "^$T$" | tr '\n' ' ')"
$P scripts/divergence_labels.py --trajs $ALL --out_dir $D/div | tee -a $M
$P - <<PY | tee -a $M
import json
src="$T"; eps={json.loads(l)["id"]:json.loads(l) for l in open(src)}
pairs=[json.loads(l) for l in open("$D/div/dpo_pairs.jsonl")]; n=0
with open("$D/train.jsonl","w") as f:
    for r in pairs:
        if r["src"]!=src or r["id"] not in eps: continue
        e=eps[r["id"]]; t=r["t"]; cmds=list(e["steps"][t]["admissible"])
        if r["chosen"] not in cmds: continue
        f.write(json.dumps({"kind":"kl","src":src,"id":r["id"],"t":t,"cmds":cmds,"target":[1.0 if c==r["chosen"] else 0.0 for c in cmds],"cand":r["chosen"]})+"\n"); n+=1
print(f"=== chain contrast r$R: {n} divergence pairs from this round's failures (one-hot on the win's action)")
PY
until [ -f $D/model/adapter/adapter_config.json ]; do $(S 16000) $P -u scripts/alfworld_opd_plus.py train --data $D/train.jsonl --student Qwen/Qwen2.5-3B-Instruct --adapter $PWD/$START --lr 1e-5 --epochs 1 --batch 8 --out $D/model > $D/train.log 2>&1 || { log "chain contrast r$R: TRAIN FAILED"; sleep 120; }; done
ADAPTER=$D/model/adapter
for sp in valid_unseen valid_seen; do until [ "$(grep -c . $D/eval_${sp#valid_}.jsonl 2>/dev/null)" -ge $([ $sp = valid_unseen ] && echo 134 || echo 140) ]; do
  $(S 10000) $P -u scripts/alfworld_collect.py --model Qwen/Qwen2.5-3B-Instruct --adapter $ADAPTER --split $sp --budget 50 --temperature 0 --out $D/eval_${sp#valid_}.jsonl --tag contrast_r$R >> $D/eval.log 2>&1 || sleep 30; done; done
echo "$ADAPTER" > $D/ADAPTER
log "CHAIN contrast R$R: train-game wins $NW/$((NW+NF)) | $(wc -l < $D/train.jsonl) pairs | unseen $(grep -c '"won": true' $D/eval_unseen.jsonl)/134 seen $(grep -c '"won": true' $D/eval_seen.jsonl)/140 (student 56/134, 80/140; contrast R1 = sft_all 187; env chain R3 184)"
