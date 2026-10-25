# 状态记录：2026-10-07 凌晨（集群 05:00–17:00 维护前）

写于 01:40。会话所在节点 acn216（cpu_tmux）约 02:10 到期，所有还在跑的实验已挪到能撑到 ~04:40–04:52 的 GPU 节点上。
结果都会追加到 `reports/alfworld/self_attr_rounds/queue.log`（以 `BASELINE <tag>:` / `CHAIN ...` / `... DONE` 开头的行）。
774 局 = valid_unseen 134 + valid_seen 140 + 训练集留出 500 局（game 1500–1999）。配对检验：`scripts/paired_compare.py name=dir ... --vs name`。

## 一、今晚已完成的结果（774 局）

### 主结果：定稿链（3B Model A 选步 + 诊断，hindsight 自我老师，一步环境检查，40% 纠正 + 100 回放）
| | R0 学生 | R1 | R2 | R3 |
|---|---|---|---|---|
| 774 | 416 | 547 | 572 | **602** |
| 274 | 136 | 196 | 205 | 219 |
| vs 上一轮（只赢/只输，p） | — | 165/34, 8e-22 | 50/25, 0.005 | 56/26, 0.001 |
R3 vs 主线 R3（环境裁决，无 Model A）563：67/28, p=8e-5；vs 旧 7B 链 R3 574：59/31, p=0.004。目录 `selfattr_a3bfinal/r{1,2,3}`。

### 基线（R1，同一 SFT 学生，同一评测）
| 基线 | 774 | vs 定稿 R1 547 |
|---|---|---|
| RFT 等量（264 条自己胜局步，`rft_matched/r1`） | 481 | 105/39, p=4e-8 |
| rollout 验证（同步同候选，固定策略全程 rollout，157 纠正 + 100 回放，`selfattr_rollsub_a3b/r1`） | 508 | 71/32, p=0.0002；验证成本 32,559 环境步（每步一次模型调用）vs 一步检查 820 步、0 次调用 |

### 回放 / 预算消融（R1）
| 纠正 | 无回放 | +100 回放 |
|---|---|---|
| 0 | 416 | 481（264 条回放） |
| ~20%（82–86 条） | 524（`diag_c_top86`） | 537（`a3bfinal_k20/r1`，vs 547 p=0.31） |
| 40%（164 条） | 440 / 412（种子 1）/ 451（旧训练器） | 547 |
诊断结论：440 不是设置错误、不是训练器（旧 top-86 纠正 + 新训练器 = 518 ≥ 旧 500/499/505）、也不是意外；是 40% 收进了排名靠后的弱纠正（新信息少、学生自身概率低），无回放时把模型拉回学生水平。加回放后 K 不敏感。

### 一步证据的排序
等级排序（目标条件：完成 > 满足 > 定位 > 只有新信息 > 无变化 > 不可逆）与现有加权分数在 R1/R2/R3 选出的纠正完全相同（164/164、110/110、102/102）。ALFWorld 上 ⊥（不可逆）从不触发；WebShop 上 ⊥ = 购买且奖励 < 1。

## 二、02:10 交接时仍在跑的实验
| 实验 | 控制脚本 / 节点（tmux） | 输出 | 774 tag | 预计 |
|---|---|---|---|---|
| 20% 链 R2（K=20% + 100 回放，从 537 出发） | `scripts/run_a3b_k20_r2.sh` @ aga29（`k20r2`，节点到 ~04:38） | `a3bfinal_k20/r2` | `a3bk20_r2` | ~03:15（对比 40% 链 R2 = 572） |
| R2 用 A0（原主链 R2，只把重训归因器换成 A0） | `scripts/run_r2_A0.sh` @ aga13（`r2A0`，到 ~04:52） | `selfattr_r2A0/r2` | `r2A0` | ~02:00（对比原 R2 531、日志规则 R2 464/500） |
| RFT 等量 R2（210 条胜局步） | `scripts/run_rft_matched.sh` @ aga13（`bl_rft`） | `rft_matched/r2` | `rftm_r2` | ~03:00 |
| rollout 基线 R2 | `scripts/run_rollout_a3b_chain.sh` @ aga29（`bl_roll`） | `selfattr_rollsub_a3b/r2` | `rollA3b_r2` | ~03:30–04:00 |
| ETO 精确复现（503 对，DPO β0.5 lr1e-6 b32 3ep，种子 1） | `scripts/run_eto_fair_v2.sh` @ aga29（`bl_eto`） | `reports/alfworld/rigorous/eto_fair/seed1` | `eto_fair_s1` | ~04:00（紧，可能跑不完） |
GPU 作业：aga21 到 ~02:40，agc01/agd02 到 ~03:25，aga12 到 ~04:17，aga29 到 ~04:38，aga13 到 ~04:52；2971486（5 h）在排队，维护后会启动。

## 三、维护后怎么续
所有脚本都能从中断处续跑（已完成的收集 / 老师 / 一步检查 / 训练 / 评测都按文件判断跳过）。在任意有作业的节点上：
```
cd /users/8/pan00389/multi-agent/error-attribution
tail -n 40 reports/alfworld/self_attr_rounds/queue.log          # 看哪些已出结果
tmux new -d -s k20r2   'bash scripts/run_a3b_k20_r2.sh'          # 没完成的就重新挂上（同名脚本）
tmux new -d -s r2A0    'bash scripts/run_r2_A0.sh'
tmux new -d -s bl_rft  'bash scripts/run_rft_matched.sh'
tmux new -d -s bl_roll 'bash scripts/run_rollout_a3b_chain.sh'
tmux new -d -s bl_eto  'bash scripts/run_eto_fair_v2.sh'
```
需要先有 GPU 作业：`sbatch slurm/hold_gpu_a100-4.sbatch`（或 `hold_gpu_igpu_any.sbatch`）。ETO 第 410 局的规划器会卡死，已在 `eto_fair/expert_s11.jsonl` 标记为无示范，不会再重试。

## 四、待决定 / 待做
- 20% 链 R2 出来后决定定稿用 20% 还是 40%（R1：537 vs 547，不显著）。
- 消融 P0 待批准：A3（同配方去 Model A）、A1/A2（Model A 2×2）、A6（随机选步），各 R1+R2。
- WebShop：代码准备（Model A 输入转换、老师采样 + 诊断、分级 Φ：属性/选项/价格比例；打开不合格商品 = 无进展），等批准后跑；评测建议 test 500 + dev 200。
- AppWorld：未安装、未测试，需批准做可行性检查。
- 论文计划文档：`docs/paper_plan_final_method.md`（需加入今晚的新结果和目标条件 / 进度势函数写法）。

## 五、01:48 新结果：R2 用 A0（不更新归因器）
| 原主链 R2（同一批 R2 轨迹、同老师、同筛选、同回放、同旧训练器） | 选步 | 第 0 步占比 / 平均步 | 774 |
|---|---|---|---|
| 原 R2 | R2 时重训的归因器（A0 + 本轮 201 合成 + 352 环境标签） | 15% / 2.7 | **531** |
| **R2 用 A0** | A0（不更新） | 63% / 1.0 | **486** |
| R1（起点） | A0 | 50% / 1.3 | 475 |
R2 用 A0 vs 原 R2：34/79，p=3e-5（显著更差）；vs R1：84/73，p=0.43（不涨）。结论：原链 R2 的增长来自更新后的归因器把步往后指；A0 不更新则 R2 不涨。
（A0 = SFT 学生上 LoRA，458 条：139 合成注入错误 + 319 条 SFT 学生真实失败的环境证据标签。）

## 六、02:07 新结果：RFT 等量 R2 = 513/774
```
rftm_r2                vs rftm_r1               : rftm_r2 wins only  61, rftm_r1 wins only  29, common games 774, p=0.000973
rftm_r2                vs a3bfinal_r2           : rftm_r2 wins only  42, a3bfinal_r2 wins only 101, common games 774, p=8.74e-07
```
(RFT 等量：R1 481 → R2 513；定稿 40%+回放：R1 547 → R2 572)

## 七、维护后检查（10-07 19:15）
- 20% 链 R2 = 561（R1 537 → R2 561，52/28 p=0.01）；vs 40% 链 R2 572：38/49 p=0.28（无显著差别）。R2 K = 0.2 × 271 = 54。
- rollout 基线 R2 = 485（R1 508 → 485，下降 p=0.03）；R2 只有 94 个状态有 rollout 赢家（K=129）；vs 定稿 R2 572：23/110 p=8e-15；vs RFT R2 513：67/95 p=0.03。
- RFT 等量 R2 = 513（vs 定稿 R2 572：42/101 p=9e-7）。
- ETO 没跑完：参考概率 503/503 已完成；DPO 训练极慢（每对约 43 s，503 对 × 3 epoch = 1509 步 ≈ 18 小时单卡），节点关机前只到 80/1509。需要先决定怎么加速（见对话）。
- 集群：只剩 2971486（igpu @ agc09，到 ~23:07）在跑；cpu_tmux 2995731 排队中；没有控制脚本在跑。

## 八、10-07 晚（用户生病休息，授权自主推进论文所需实验）
**定稿 = 20% 纠正 + 100 回放。** 每个在跑的实验都有一个监督子代理（只能重启自己的驱动；坏 GPU / 代码 bug 停下报告）。
| 实验 | 驱动（节点 / tmux） | 输出 | 774 tag |
|---|---|---|---|
| 主链 R3（20%） | agd07 / k20r3 | a3bfinal_k20/r3 | a3bk20_r3 |
| 20% 只纠正 R2 | agd07 / k20nr | a3bk20_norep/r2 | a3bk20nr_r2（R1 = 530） |
| 不筛选基线 R1→R2 | agd07 / nofilter | a3b_nofilter/r{1,2} | a3bnf_r{1,2} |
| HINT-SD 基线 R1→R2 | agd07 / hintsd | a3b_hintsd/r{1,2} | a3bhsd_r{1,2} |
| 7B 链 R1→R3 | agd07 / a7b | a7b_final/r{0..3} | a7b_r{0..3}（R0 = 444） |
| A3：去掉 Model A（环境裁决选步，其余同定稿） R1→R3 | cn1019 / noA20 | selfattr_noA20/r{1,2,3} | noA20_r{1,2,3} |
| 主链种子 1、2（R1→R3） | cn1019 / seed1, seed2 | a3bfinal_k20_s{1,2} | a3bk20s{1,2}_r{1,2,3} |
| WebShop 定稿（环境裁决选步）R1→R2 | cn1019 / ws_final | reports/webshop/final_adj/r{1,2} | dev 200（学生 86 / 0.598） |
| SERL 移植（子代理，只冒烟测试） | — | serl_work | — |

**WebShop 的决定与依据：**
- Model A 在 WebShop 上退化：95% 指第 0 步；不带观察和带观察两种格式结果完全相同。
- 证据：45% 的原始首次搜索结果页已含目标商品，225 / 407 的失败买了部分正确的东西。
- 因此 WebShop 选步改用环境裁决（与 ALFWorld 的 STEP_POOL 同一规则）：587 个候选步，平均第 2.84 步，第 0 步 3%。
- 同时修了两个打分 bug：首页横幅识别、选项点击被判无变化。结果页"定位"改为分级（标题命中的属性比例 + 价格）。
- 论文里如实写"归因器在 WebShop 上不迁移，回退到环境裁决"。

**基础设施：**
- 坏 GPU 作业已替换：2997883 被分到两个节点；2997885 的 aga36 GPU 报 CUDA_ERROR_UNKNOWN。
- WebShop 服务器：agd07:38300。
- 4 天 CPU 作业 cn1019 已启动，新的驱动放在那里。
- 10-07 23:10 用户决定：**不需要 ETO 基线，也不需要 A3（定稿去掉 Model A）**。A3 链已停（cn1019 tmux noA20 已关，其监督代理已停），ETO 不再跑。
- 10-08 00:40 用户决定：不跑多种子，先把每个实验的一个种子做完。种子 1 / 2 已在 R1 后停止（557 / 552）。
- 10-08 02:05 发现并修复 WebShop 发布格式 bug（步号写成字符串，Model A 收到空转录）。修复后 Model A 在 WebShop 上 2% 指第 0 步、平均第 5.3 步（多指购买 / 选项）。之前 WebShop Model A 的 R1（65/200）作废；重跑中：reports/webshop/final_modelA（cn1019 tmux wsAf）。
- 10-08 13:20：所有 ALFWorld 基线 / 消融的 R3 完成（见 paper_writing_guide.md §9）。WebShop R1→R3 完成，逐轮变差（dev 85 → 51 → 40；700 局 R1 95 vs 学生 145 —— 作废，见 10-08 15:15 条）。7B 链 R1 在 04:10 停掉后未重启，13:15 重启（加了"每轮必须有模型"的检查和 bash -x 日志）。
- 10-08 13:30 新增消融（用户要求）：回放 + 随机 20%（cn1019 tmux rand20，a3b_rand20，tag a3brand20_r1..3）。
- 10-08 13:30 新增（用户要求）：rollout 验证 + 20% + 100 回放（cn1019 tmux roll20，selfattr_rollsub_a3b20，tag roll20_r1..3）。
- 10-08 14:15：回放 + 随机 20% 的 R1 = 505（定稿 537，p=0.006）。WebShop 新提示只在推理时使用，反而让学生和 R1 都大幅变差（145 → 64，95 → 45 / 700）。
- 10-08 14:30：rollout 20% + 回放 R1 = 483（定稿 537，p=2e-6）。7B R1 = 549（R0 444，p=3e-14）。

## Oct 8 ~20:30 — WebShop failure analysis (offline, hidden goal used for analysis only)
Train collection R1 (student, 600 sessions, 407 failures):
- bought a wrong product 249 (61%): target was listed but another product bought 159, target never listed 90. Only 20/249 wrong products pass the visible checks (options / price / attribute phrase), so the mistake is visible on the product page.
- no purchase, hit the 15-step limit 130 (32%); bought the target with wrong options 28 (7%, mean reward 0.83).
- failures with reward > 0: 225/407 (mean 0.41) -> partial purchases raise mean reward but never success.
Model A pointer vs failure type: on the 249 wrong-product failures it points at `buy` 167 times; exactly at the click that opened the wrong product only 10/249. Kept R1 corrections (81): buy->option 17, search->search 38, option->option 18, product->product 1, nothing ->buy.
(INVALID, see 15:15 entry) 700-session eval, student -> R1: no purchase 278 -> 446, wrong product 254 -> 145, success 145 -> 95. Valid evidence for the same mechanism: chain dev 200 (separate worker per shard), no purchase 33 -> 60, success 86 -> 85, option clicks 3.4 -> 4.4.
Side note: R1 train collection covers sessions 700-1299 and overlaps the SFT demo sessions 700-810 (win 51/111 there vs 142/489 elsewhere). Dev/test evaluation is unaffected.

## Oct 8 15:15 — WebShop eval700 runs (v1 / v2 / v3 prompt) are INVALID
All shards of one model ran concurrently under ONE server worker id (`--tag ev_${name}_b${B}`, `evp2_$name`, `evp3_$name`); webshop_server.py keeps one env per worker id, so shards reset/stepped each other's sessions. Instruction switching mid-episode: v1 893/1400 sessions, v2 1151/1400, v3 (partial) 886/1041. Clean: chain dev evals (per-shard tags) 0/200, train collection 0/600.
Numbers that must not be used: student 145/700 vs R1 95/700 (p=6e-5), v2 64/45, any v3 numbers. Fix = per-shard tag (e.g. evp3_${name}_${split}${off}) and fresh output dirs; rerun needs user approval. The v3 driver (tmux ws_evp3) is still running and producing contaminated records.

## Oct 8 15:57 — clean WebShop 700 eval (reports/webshop/eval700c, per-shard worker ids, 0/2800 contaminated)
student_v1 305/700 (43.6%, reward 0.581, no purchase 113, 8.2 steps); r1_v1 297 (0.520, no purchase 222, 10.0 steps), vs student 31/39 p=0.40.
student_v3 292, r1_v3 282; v3 vs v1: student 12/25 p=0.047, R1 21/36 p=0.063. Dev-200 slices: 86 / 83 / 83 / 77.
## Oct 8 15:47 — teacher-shard resume bug fixed in scripts/self_teacher_informed.py (backup _backup_oct8.py)
Restarted shards re-sliced the shrunken todo list -> missing states in the selection pool (training targets themselves deduplicated; one-step evidence coverage complete everywhere).
Affected pools: main 20% R3 171/249 (result 591), no-replay R2 303/306, rand20 R2 211/301, roll20 R2 226/322 (running), 7B R2 172/214 (training). Repairs await user approval.

## Oct 8 16:40 — what is running (all drivers in tmux on cn1019; each has a supervising subagent)
| tmux | script | what | next result |
|---|---|---|---|
| a7b | scripts/run_7b_chain.sh | 7B final recipe; R2 repaired (214/214 states), R2 trained 16:23, 774 eval running | a7b_r2, then R3 (a7b_r3) |
| roll20 | scripts/run_rollout20_chain.sh | rollout 20% + replay; R2 repaired, teacher refill -> rollout verification (~1 h) | roll20_r2, roll20_r3 |
| rand20 | scripts/run_rand20_chain.sh | replay + random 20%; R2 eval | a3brand20_r2, r3 |
| r3check | scripts/check_r3_pool.sh | main R3 pool check (control reproduced original 50/50); refill 78 states -> evidence -> env_top K=50 -> retrain if different | "R3 POOL CHECK" line; a3bk20_r3fix |
| rsplit | scripts/run_rollout_split.sh | rollout-split ablation, R1: pass82 / rej82 (+100 replay) | rsplit_pass82, rsplit_rej82 |
| — | SERL job 3072993 (qianwen a100-4, aga26) | started 15:48; checkpoints 10/22/34 -> 774 eval | serl_s10/22/34 |
| — | online HINT-SD (option B) | being implemented by a subagent (scripts/hintsd_online.py); full run after review | hsdon_c1/c2/c3 |
Other fixes today: lib_baselines.sh train1 waits up to 2 min for the adapter (NFS delay caused duplicate trainings; backup _backup_oct8.sh). Holds: qianwen account added (slurm/hold_gpu_a100-4_qw.sbatch, higher fairshare).

## Oct 8 16:53 — main R3 pool check DONE: selection changed (31/50 overlap), full-pool R3 retrained = 574 (orig 591; 27/44, p=0.057). Both numbers kept (user 17:00); which one goes in the paper is undecided.
R3fix vs R2 48/35 p=0.19; vs R1 65/28 p=2e-4; vs student 192/34 p=6e-28; vs RFT R3 95/45 p=3e-5; vs rollout-40% R3 111/34 p=1e-10. Teacher analysis text is not bit-reproducible (GPU drift; 3/6 control states changed text, 1/6 changed top-2).

## Oct 8 17:25 — new runs + WebShop goal-seed fix
- Online HINT-SD (option B) launched: tmux `hsdon`, scripts/run_hintsd_online.sh (fidelity table in the implementing agent's report; deviations: our SFT LoRA start (7 projections), warm-up 5% as in the paper text, HF scorer instead of vLLM). Evals hsdon_c1..c3.
- WebShop: the WebShop code samples each goal's price cap with the global `random` before its own seed(233) shuffle -> different caps per env (worker id) and per server restart. Fixed in scripts/webshop_server.py (GOAL_SEED=0 before each env; backup _backup_oct8.py). New seeded server: tmux `wsserver`, http://aga34:38300 (hold 3080458). All earlier WebShop numbers (incl. eval700c) used random caps per worker -> unbiased but noisier; the old agd07 server ends ~19:15.
- WebShop prompt-v2 chain relaunched from R0 (user): tmux `ws_p2`, scripts/run_webshop_p2.sh -> reports/webshop/final_p2 (seeded server; Model A selector unchanged).
- Why Model A points at buy (R1 data): 181 of 277 failures that end with a purchase are attributed to that final buy; the buy page shows "Your score 0.857" (the failure becomes visible there), while the real mistake (opening the wrong product) is earlier. 47% of all pointers are the last step. Not a step-limit effect: those agents bought within budget; the 132 budget-exhausted failures are attributed to option/search steps.
## Oct 8 17:24 — rollout-split ablation DONE: pass82 548 (vs roll20 R1 483: 102/37 p=3e-8; vs ours R1 537: 42/31 p=0.24), rej82 498 (vs student 121/39 p=6e-11; vs roll20 R1 61/46 p=0.18; vs pass82 49/99 p=5e-5).
## Oct 8 18:10 — rand20 DONE: 505 / 458 / 462 (R3 vs main R3fix 28/140 p=4e-19). Model A score-free diagnostic: 393/407 pointers unchanged (buy 181->175; exact wrong-product hit 10->14/249) -> leakage is not the cause. webshop_final.py release now drops the score by default (--keep_score for old format); final_p2 R1-R3 will use it.
## Oct 8 18:20 — user decision: main-table R3 = 591 (footnote † discloses the 171/249 pool and the full-pool rerun 574; appendix reports 574 and its paired tests).
## Oct 8 18:53 — 7B chain DONE: R0 444, R1 549, R2 576 (vs R1 66/39 p=0.01), R3 540 (vs R2 43/79 p=0.001; held-out 381->351).
## Oct 8 20:28 — roll20 DONE: 483 / 390 / 435 (R2/R3 coverage 322/322, 373/373; verification ~26.6k / ~33k env steps per round).

## Oct 8 21:05 — replay/RFT clarification and new runs (user-approved)
- Replay and the old "RFT matched" were STEP-level samples from the SHORTEST wins (make_replay.py: 50 shortest wins x 2 steps; RFT matched 264/210/202 steps). Shortest wins are almost all simple pick-and-place/look tasks -> heat/clean/cool steps were never rehearsed (7B R3 losses concentrated there).
- make_replay.py now has --mode shortest (old, default, reproduces old files exactly) | random_full (random wins, full trajectories, 100 steps) | all (every step of every win = standard RFT data). Backup: make_replay_backup_oct8.py.
- Launched: tmux `rftfull` = scripts/run_rft_full.sh (STANDARD RFT: SFT on all steps of all wins each round; R1 10,853 steps from 779 wins; tags rftfull_r1..3). tmux `a7brr` = scripts/run_7b_r3_randreplay.sh (7B R3 from R2 with the same 45 corrections, replay = random_full; tag a7b_r3rr).
## Oct 8 20:45 — status
- New (user): tmux `sft100e` = scripts/run_sft100eps_chain.sh: final recipe with replay = full-trajectory SFT on 100 RANDOM wins (~1.4k steps) + top-20% corrections (R1 reuses a3bfinal_k20 r1 corrections). run_k20_round.sh now takes REPLAY_ARGS (default unchanged; backup _backup_oct8b.sh).
- Online HINT-SD: step 238/541; ckpt1 (1323 episodes) = 367/774 (55/71/241); training rollout success 0.58 -> 0.43.
- SERL: a100 job 3072993 was cancelled 17:55; now running as 3087233 (ee5239 interactive-gpu, agd06); 6/34 steps, ~27 min/step -> ETA ~09:00 Oct 9.
- WebShop v2 prompt chain: R0 305/700 (no purchase 92), R1 152/700 (no purchase 478) — same mechanism (buy->option 16, search->search 29, option->option 21, product->product 3; Model A buy 171). R2 running.
## Oct 8 20:47 — user decisions
- WebShop v2 chain STOPPED after R1 (R0 305 -> R1 152/700). WebShop to be written up as a limitation with the diagnostics; second dataset = TextCraft (feasibility running). Seeded server on aga34 (tmux wsserver) left running for now.
- Standard RFT and "100 random wins full SFT + 20% corrections" run ONE round only (watchers tmux stop_rft / stop_s100 = scripts/stop_after_r1.sh stop each driver right after its R1 774 line). Compare rftfull_r1 vs sft100e_r1 vs ours R1 (537).
## Oct 8 20:46 — WebShop v2 chain final: R0 305/700 (dev 92, no purchase 92, 7.97 steps) -> R1 152/700 (dev 44, no purchase 478, 13.16 steps); R1 vs R0 14/167 p<1e-4; kept corrections option 37 / search 29 / nav 8 / product 3 / buy 0; R2 collect partial (126/574 won) then stopped by user.
## Oct 8 ~21:10 — TextCraft feasibility DONE (go) and R1 approved (R1 only)
Feasibility: AgentGym textcraft ported to deterministic pure python (scripts/textcraft_env.py / _agent.py / _evidence.py); 544 goals; SFT student (100 oracle demos) 60.8% on test (d1 96%, d2 63%, d3 23%, d4 0%); failures mostly too-few/missing ingredients (local, reversible); evidence AUC 0.90 (good vs bad candidates); Model A (standard input) points at step 1 in 68% -> ~14/50 true corrections.
R1 protocol (user-approved): TEST = 162 goals x 5 seeds = 810 tasks (goal-disjoint from TRAIN 378 goals x 4 seeds = 1512); standard Model A input; env_top 20% + 100 default replay; same trainer; per-task McNemar + goal-level sign test. Outputs reports/textcraft/final/{r0,r1}.
## Oct 8 21:28 — sft100only_r1 420 (= student), sft100e_r1 503 (corrections +83 over the same SFT; below ours 537, p=0.004). Online HINT-SD diagnostics requested (format-shift hypothesis: extra feedback turn makes the teacher diffuse; teacher lp of taken action -2.88 vs student -1.32).
## Oct 8 21:42 — 7B R3 with random full-trajectory replay = 566 (vs orig R3 540: 46/20 p=0.002; vs R2 576: 48/58 p=0.38); heat 61->76, clean 99 unchanged.
## Oct 8 22:35 — 7B R1 ablations launched (user): tmux a7brand20 (scripts/run_7b_rand20_r1.sh, random 65/323, tag a7brand20_r1) and a7broll20 (scripts/run_7b_roll20_r1.sh, rollout verification with 7B R0, tag a7broll20_r1). Both reuse a7b_final/r1 states/candidates and its exact 100-step replay file (shortest-wins rule, same as 7B R1 = 549) -> one-component comparisons vs 549.
GiGPO: jobs 3112270 (qianwen a100-4, est. start 00:26) / 3112271 (ee5239), smoke job 3112183; settings in serl_work/run_gigpo_ours.sh; evals gigpo_s10/22/34.
## Oct 8 22:34 — GiGPO smoke passed (resume, step groups, LoRA sync/export checks OK; 61 GB host RAM); full run started in job 3112357 (qianwen, a100-8 partition, agb02 A100-40GB). Pending twins 3112270/3112271 to be cancelled once steps advance.
## Oct 8 22:54 — standard RFT R1 = 431 (vs student 75/60 p=0.23; ours R1 537 vs it 154/48 p=4e-14; step-matched RFT 481 vs it 100/50 p=6e-5). 7B random-full replay R2 = 541 (vs old R2 576: 14/49 p=1e-5) -> random_full replay is NOT better; awaiting user decision on the 7B chain.
## Oct 8 22:58 — SERL step 10 = 447 (70/86/291; vs student 77/46 p=0.007; ours R1 vs it 135/45 p=1e-11); SERL training success 0.50 -> 0.63 (steps 1-10). Supervisors asked to evaluate every checkpoint (SERL 14/18/26/30, GiGPO 4/8/...) for interim readouts.
## Oct 8 23:05 — SERL note: the first run (3072993) was cancelled at step 4 by its supervisor because peft 0.21.2 returned an empty LoRA state dict under FSDP1 (vLLM rollouts never received updated LoRA weights; saved adapter empty). Fixed with a key-based LoRA collector (serl_work/port/CHANGES_SUPERVISOR.md); restarted as 3087233 (L40S) at 17:57; verified 504 tensors per sync. GiGPO uses the same fixed tree (its smoke test checks 504-tensor syncs). Appendix must mention this port fix.

## Oct 9 00:25 — two method improvements under test (user decision: test both together)
1. Margin rule (rule env_top_margin in scripts/proxy_arms_se3b.py, backup _backup_oct9.py): a correction must STRICTLY beat the taken action's one-step evidence. In past ALFWorld selections 80-98% already did (3B R1/R2/R3 96/80/96%, 7B 98/88/80%); TextCraft only 30/76.
2. Failure-weighted replay (make_replay.py --mode fail_weighted): 100 steps split over task types by the agent's OWN training failures, complete trajectories of each type's shortest wins (7B R3: two_obj 26, heat 25, clean 22, cool 17, simple 7, look 2; contains heat/clean/cool actions, old replay had none).
Runs: tmux a7bv2 = scripts/run_7b_r3_v2.sh (7B R3 from the original R2, both changes; tag a7b_r3v2). TextCraft R1_v2 (both changes, reuse R1 collection/Model A/teacher/evidence) requested from the TextCraft agent -> reports/textcraft/final/r1_v2.
Stopped by user: a7b_rr chain during R3 (random_full replay; R2 = 541 vs old 576, 14/49 p=1e-5) and the replay-only run a7b_r3fw.
TextCraft R1 (old recipe): 439/810 vs R0 454 (48/63, p=0.18) — 76/76 corrections were "get", only 30 true corrections; replay from trivial depth-1 wins (mean length 2.0).
## Oct 9 00:45 — SERL step 14 = 478 (vs student 116/54 p=2e-6; vs ours R2 561: 39/122 p=4e-11). TextCraft R1_v3 requested (see docs/status_2026-10-09_textcraft_changes.md).
## Oct 9 00:50 — 7B R1 ablations: rand20 508 (ours 7B R1 549 vs it 84/43 p=3e-4), roll20 524 (58/33 p=0.01). Extended to R2/R3 (user): tmux a7brand23 (scripts/run_7b_rand20_r23.sh), a7broll23 (scripts/run_7b_roll20_r23.sh), same protocol as a7b_final (shortest-wins replay, 3B Model A, 7B teacher). GiGPO s4 = 460 (512 episodes; vs student 69/25 p=6e-6; ours R1 vs it 128/51 p=8e-9). All 6 holds busy; submitted holds 3125923 (qianwen a100-8), 3125924 (qianwen a100-4), 3125925 (ee5239 a100-4).
## Oct 9 00:55 — standard RFT relaunched (user) to run R2 and R3 (tmux rftfull, scripts/run_rft_full.sh; R1 = 431 reused).

## Oct 9 01:20 — 7B R3 work under the user's overnight autonomy note (memory: feedback_autonomy_oct9_night)
- a7b_r3v2 (from original R2; margin rule + fail_weighted replay) = 576 (95/107/374): = R2 (48/48, p=1), +36 over R3 540 (54/18, p=3e-5); heat 61->79, clean 99->108.
- Why R3 plateaus: of the 7B R2 model's 226 training failures, 124 (55%) fail AFTER picking/placing (wrong place, skipped heat/clean, second object), but the 3B Model A points before the first pick in 195/226 (t0 74%), so all 45 corrections are step-0 "go to".
- New code: scripts/full_view.py + self_teacher_informed.py --full_view (complete prefix, all future steps, full observations; default off; backup _backup_oct9.py). scripts/run_7b_r3_v3.sh (Model A variant/weights via env, full-view teacher, env_top_margin, fail_weighted replay; cached one-step results reused).
- Launched: a7bv3b = 7B Model A (attributor_7b_no_observations, its training format; mean t 11.8, t0 0.55 vs 3B 5.6/0.74) -> tag a7b_r3v3b; a7bv3a = 3B Model A with the FULL trajectory (variant full) -> tag a7b_r3v3a.
- Gold check of Model A input formats on the ALFWorld transfer set (n=311): 3B no-obs prompts = step 64.3%; 3B full prompts and 7B runs pending (reports/alfworld/modelA_fullinfo_check/summary.txt).
- 01:24 v3a STOPPED: 3B Model A with the full trajectory points at step 0 in 218/226 (96%; standard input 167/226), at/after the first pick in 1/226 — full observations make the 3B Model A MORE step-0-biased, so it cannot reach the late failures. v3b (7B Model A) continues: t0 124/226, 86/226 at/after the first pick.
## Oct 9 01:48 — online HINT-SD DONE: c1 367, c2 341, c3 344 (student 416); 7.6 GPU-h, 4324 episodes, 5,701 trained steps.
## Oct 9 02:05 — SERL step 18 = 505 (2304 episodes; vs student 132/43 p=1e-11; ours R2 561 vs it 111/55 p=2e-5); curve 426/452/447/478/505. GiGPO step 8 = 475 (1024 episodes; ours R1 vs it 113/51 p=2e-6); curve 460/475.
- 02:30 v3b selection analysis (7B pointer; 226 states): 89 pass the margin; top-45 cut-off env score 2.4. t=0: 124 states, 72 pass, 37 selected (taken "go to" mean 0.80; 26 states reach the max 3.0 = "go to" revealing the goal object). At/after first pick-up: 86 states (taken: take 54, go to 16, cool 8, move 7, heat 1; mean taken score -0.86), only 14 pass, 6 selected — the full-view teacher's top-2 alternatives there are mostly "go to" (151/172). Two bottlenecks for late failures: (1) the teacher rarely proposes non-navigation fixes at late states; (2) the generic one-step score gives a goal-revealing "go to" (2 loc + new info = 3.0) more than completing an operation (2 sat), so the top-K fills with step-0 navigation. Decision: do NOT change the evidence weights overnight (core of all main results); write up options for the user.
## Oct 9 02:40 — GiGPO step 10 (1280 episodes, R1 budget) = 475 (68/93/314): vs student 106/47 p=2e-6; ours R1 537 vs it 111/49 p=1e-6; vs SERL s10 447: 70/42 p=0.01. Curve 460/475/475.
## Oct 9 02:45 — TextCraft R1_v3 = 282/810 (collapse: craft->get corrections taught 'stop crafting'); R1_v2 = 459 (= R0). See docs/status_2026-10-09_textcraft_changes.md.
- 03:00 **a7b_r3v3b = 602** (96/112/394), best 7B run: vs R2 576 64/38 p=0.013; vs R3 540 69/7 p=6e-14; vs r3v2 576 44/18 p=0.001. Gains vs R2: heat +10, cool +9, two +7, clean -5. Recipe = 7B Model A (no-obs) + full-view teacher + margin + fail_weighted replay, trained from the original R2. CAVEAT: the 7B chain now mixes recipes (R1/R2 original, R3 v3b); one seed; two factors changed at once vs r3v2.
- 03:05 launched the 2x2 ablation (pointer x teacher view), both with margin + fail_weighted: a7bv3c = 7B pointer + standard teacher (tag a7b_r3v3c), a7bv3d = 3B pointer + full-view teacher (tag a7b_r3v3d). run_7b_r3_v3.sh now takes TEACHER_FLAGS.
## Oct 9 03:10 — 7B rand20 R2 = 505 (vs its R1 508: 56/59 p=0.85; vs a7b_r2 576: 37/108 p=3e-9). R3 started. 7B roll20 R2 in eval (teacher 258/258, rollout coverage OK).
## Oct 9 03:40 — 7B roll20 R2 = 569 (85/107/377): vs a7b_r2 576 38/45 p=0.51 (n.s.); vs its R1 524 70/25 p=4e-6; vs rand20 R2 505 92/28 p=4e-9. Rollout verification ~15.7k+ env steps (one shard partially unlogged). R3 of both 7B ablation chains collecting.
## Oct 9 03:50 — SERL step 22 (2816 episodes, R2 budget) = 532 (84/105/343): vs student 151/35 p=2e-18; ours R2 561 vs it 90/61 p=0.022. Curve 426/452/447/478/505/532, +~7/step; step 34 due ~07:15. Compute-accounting analysis requested (docs/compute_accounting_oct9.md).
## Oct 9 03:55 — standard RFT R2 = 428 (59/90/279): vs R1 431 51/54 p=0.85; ours R2 561 vs it 180/47 p=2e-19; 12,467 SFT steps from 942 wins; R3 collecting.
## Oct 9 04:00 — v3c selection (7B pointer + standard teacher): 7/45 not step-0, 6/45 after the first pick-up (v3b 8 and 6); identical targets v3c-v3b 31/45 -> selection follows the pointer, not the teacher view. Infra: lib_baselines.sh pick()/S now measure and pin each node of a hold (-w node), because the 2-node igpu hold 3072782 caused OOMs when the step landed on the busier node (backup _backup_oct9.sh; affects newly started drivers only).
## Oct 9 04:10 — GiGPO step 14 (1792 episodes) = 514 (81/102/331): vs student 139/41 p=1e-13; ours R2 561 vs it 95/48 p=1e-4; vs SERL s14 478: 79/43 p=0.001. Curve 460/475/475/514.

## Oct 9 08:10 — overnight results (all supervisors died ~04:15 from an API session limit; the drivers kept running and finished; GiGPO re-supervised)
ALFWorld 3B, budget-matched (R1 ~1300 / R2 ~2800 / R3 ~4300 episodes):
| method | R1 | R2 | R3 |
|---|---|---|---|
| ours | 537 | 561 | 591 (full-pool 574) |
| SERL | 447 (s10) | 532 (s22) | **596** (s34) |
| GiGPO | 475 (s10) | 538 (s22) | s30 582; s34 pending (~10:30) |
| online HINT-SD | 367 | 341 | 344 |
| standard RFT | 431 | 428 | 446 |
Paired: R3 SERL s34 vs ours 591: 63/58 p=0.72 (tie); vs 574: 82/60 p=0.08. R2: ours vs SERL 90/61 p=0.022; ours vs GiGPO s22 83/60 p=0.065. R3 ours vs RFT 187/42 p=5e-23. SERL curve 426/452/447/478/505/532/540/554/596; GiGPO 460/475/475/514/559/538/562/582.
7B R3 2x2 (all with margin + fail_weighted replay, from original R2 576): 3B ptr + std teacher (v2) 576; 3B ptr + full-view teacher (v3d) **596** (vs v2 30/10 p=0.002); 7B ptr + std teacher (v3c) 594 (vs v2 51/33 p=0.06); 7B ptr + full-view teacher (v3b) **602** (vs v2 44/18 p=0.001; vs v3c 37/29 p=0.39; vs v3d 30/24 p=0.50). => the full-view teacher alone (keeping the 3B Model A) recovers most of the gain.
7B ablations: rand20 508/505/397; roll20 524/569/558 (v3b vs roll20 R3 94/50 p=3e-4). 7B ours original 549/576/540.
Compute (docs/compute_accounting_oct9.md): GPU-hours to R3 — ours ~11 proc-wall / ~7 fair-share, SERL ~13.4, GiGPO ~10.3 (est.), HINT-SD 7.8/4.5; ours is NOT cheaper than GiGPO in GPU-hours (87% of ours is unbatched HF collection); backprop tokens ours 3.5M vs SERL/GiGPO ~80M (23x fewer), RFT 15.5M by R2.
## Oct 9 09:05 — GiGPO s34 = 603 (96/112/395): vs ours R3 591 79/67 p=0.36; vs full-pool 574 97/68 p=0.029; vs SERL s34 596 65/58 p=0.59; 10.14 GPU-h (1 A100). At the R3 budget both online-RL baselines match or exceed our 3B R3; we lead at R1 (p<=1e-6) and R2 (vs SERL p=0.022, vs GiGPO p=0.065).

## Oct 9 ~11:40 — new runs (user)
- 3B R3 with the improved recipe (standard 3B Model A + full-view teacher + env_top_margin + fail_weighted replay), from R2 561 on the R3 collection (full 249-state pool): tmux a3bv3, scripts/run_3b_r3_v3.sh, tag a3bk20_r3v3. (First launch lost the EVIDENCE=generic export through a bad sed; stopped within minutes, fixed, relaunched.)
- 7B baselines: online HINT-SD (tmux h7on, scripts/run_7b_hintsd_online.sh, 1125 tasks x 4 = 4500 episodes, ckpts 1500/3000/4500, tags h7on_c1..3); standard RFT (tmux rft7b, scripts/run_7b_rft_full.sh, tags a7brft_r1..3); SERL-7B and GiGPO-7B being set up by an agent (2-GPU jobs, 35 steps = 4480 episodes, evals at steps 12/23/35).
## Oct 9 12:10 — SERL-7B / GiGPO-7B set up (merged 7B SFT model serl_work/models/qwen25_7b_sft100_merged; 2-GPU FSDP + vLLM TP2; 35 steps; evals at 4/8/12/16/20/23/28/32/35; tags serl7b_sN / gigpo7b_sN); jobs 3157851-3157856 PENDING (cluster full); tmux s7b_wd (watchdog) and s7b_eval (eval scheduler); supervised together with 3B R3 v3 / 7B HINT-SD / 7B RFT.
## Oct 9 12:15 — 3B R3 with the improved recipe (a3bk20_r3v3) = 566 (100/103/363): vs R2 561 48/43 p=0.68; vs full-pool R3 574 27/35 p=0.37; vs old R3 591 32/57 p=0.011; vs SERL s34 596 56/86 p=0.015; vs GiGPO s34 603 69/106 p=0.006. 41/50 targets identical to full-pool R3 (all 'go to', 37 at t0). The 7B gain (v3d 596 vs v2 576) did NOT reproduce on 3B.
## Oct 9 12:45 — the all-40% chain (selfattr_a3bfinal; same recipe, K=40%; pools complete 274/274, 256/256) = 547 / 572 / 602 (R3 vs full-pool 20% R3 574: 61/33 p=0.005; vs 591: 53/42 p=0.31) — on par with GiGPO 603 / SERL 596 at the R3 budget and ahead at R1/R2. User: 'R3用40%的数据吧' -> launched R3 with K=40% from the 20% chain's R2 on the full pool (tmux a3bk40, scripts/run_3b_r3_k40.sh, tag a3bk20_r3k40). Pending user decision: adopt the all-40% chain as the main result.
## Oct 9 13:15 — TextCraft R1 final recipe (v4) = 470/810 (vs R0 454: 66/50 p=0.16 n.s.; vs old R1 p=0.0013).
## Oct 9 13:25 — 3B R3 with K=40% from the 20% chain's R2 (a3bk20_r3k40) = 560 (95/104/361): = R2 561 (40/41); vs full-pool 574 31/45 p=0.14; vs all-40% chain R3 602 27/69 p=2e-5; vs SERL 596 61/97 p=0.005; vs GiGPO 603 67/110 p=0.002. All R3 variants from the 20% R2 (561): old 591 (pool 171/249), full pool 574, v3 566, K40 560 -> only the buggy-pool run beat R2. The all-40% chain (547/572/602) took a different path (its own R2 = 572). R3 outcomes look dominated by the starting R2 model plus run-to-run noise; seeds needed.
## Oct 9 13:50 — 7B standard RFT R1 (a7brft_r1) = 475 (65/93/317): vs 7B R0 444 86/55 p=0.011; 7B chain R1 549 vs it 129/55 p=5e-8. 15,145 SFT rows from 1,006 wins. R2 collecting.
## Oct 9 14:00 — TextCraft (user): R1_v4_k40 (K=152, reuse v4 pool) and R2_v4 (final recipe 20%, from R1_v4 = 470) launched in parallel by the TextCraft agent.
## Oct 9 15:45 — launched 7B chain from the original 7B R1 with 7B Model A + full-view teacher + margin + fail_weighted for R2/R3 (tmux a7bv3bchain, scripts/run_7b_chain_v3b.sh, tags a7bv3b_r2/r3). Dedicated TextCraft GPU requested (job 3175414 tcded_gpu, qianwen preempt-gpu); watcher tmux tc_bind restarts the TextCraft drivers with HOLD_NAME_RE once it runs; lib_baselines.sh pick() honors HOLD_NAME_RE (backup _backup_oct9b.sh).
## Oct 9 15:50 — 7B online HINT-SD c1 (1504 episodes) = 429 (62/83/284): vs 7B R0 444 54/69 p=0.21; 7B chain R1 549 vs it 157/37 p=9e-19; vs 7B RFT R1 475 58/104 p=4e-4. 2.46 GPU-h to ckpt1.
## Oct 9 ~16:20 — user approved a NEW TextCraft pipeline (reports/textcraft/plan): plan-then-act student (plan at t=0 with total base-ingredient counts + craft order, SFT on expert plans+actions, new R0), Model A with a hint and the full task statement (variant chosen on training failures), teacher that first lists total requirements and what went wrong (analysis budget ~256 tokens), then the final recipe (one-step F, env_top_margin, K=20%, fail_weighted replay). SFT -> R0 -> R1.
## Oct 9 ~16:35 — TextCraft plan pipeline switched to option B (user): the student writes a short plan EVERY step (inventory / total requirements / missing / next) then scores actions; SFT on expert per-step plans + actions; Model A sees the plans as the agent's reasoning (matches its training format); corrections = teacher's corrected plan (token SFT) + one-step-verified action conditioned on it; replay rows include the student's own plan.
## Oct 9 ~16:50 — TextCraft R2_v4 (old final recipe, round 2) = 417/810: worse than R1_v4 470 (44/97 p=1e-5) and R0 454 (50/87 p=0.002). Get-loop drift compounds: get share 0.30 -> 0.52 -> 0.68; failures ending in 4x same get 3 -> 60 -> 106. Supports the per-step plan redesign.
## Oct 9 ~16:55 — user: stop the old TextCraft recipe; R1_v4_k40 stopped mid-training; dedicated GPU (tcded_gpu) left for the plan pipeline.
## Oct 9 17:05 — user: 3B main result = the 20% chain; 7B SERL/GiGPO dropped (jobs 3157851-3157856 cancelled, tmux s7b_wd/s7b_eval stopped).
## Oct 9 17:15 — user: all compute to TextCraft. PAUSED (resume by relaunching the same scripts; completed stages are skipped, HINT-SD resumes from state/): scripts/run_7b_chain_v3b.sh (R2 stage), scripts/run_7b_rft_full.sh (R2 training), scripts/run_7b_hintsd_online.sh (step ~380/563). Submitted 3 more preempt-gpu tcded_gpu jobs for TextCraft.
## Oct 9 17:20 — paused 7B runs: where each stands (relaunch the same script to resume)
- 7B RFT (scripts/run_7b_rft_full.sh): R1 475 done; R2 collection done (1500 games, 1088 won, 16,613 rows); R2 training killed at row 13,424 with no adapter -> relaunch retrains R2 (~1.6-3 h), then eval + R3.
- 7B online HINT-SD (scripts/run_7b_hintsd_online.sh): state at step 380 (3,040 episodes); ckpt1 = 429; ckpt2 (3,000 episodes) saved, eval partial (resumes); ckpt3 needs ~183 steps (~2.5-3 h).
- 7B v3b chain (scripts/run_7b_chain_v3b.sh): R2 trained; eval partial (unseen 96/134, seen 107/140, held-out shards 0 and 3 done; shards 1-2 pending) — on 524 evaluated games 390 wins vs a7b_r2 386; R3 not started.
Holds running: 3072782 (to ~19:20), 3125923 (~20:50), 3125924 (~21:20); TextCraft also has tcded_gpu 3175414, 3181977, 3181978 (+3181979 pending).

## Oct 9 18:30 — paper tables updated (user: "3B R3 写 590+ 的数字，解释我自己写")
- Overleaf draft (local clone, not committed or pushed) updated:
  - tables/main_results.tex: R3 = 591.
  - tables/baselines.tex: all baselines through R3.
  - New tables: tables/ablations.tex, tables/results_7b.tex (596 / 602), tables/verification_cost.tex.
  - tables/cost.tex: measured numbers.
  - Appendix tables: tab:paired regenerated from raw logs; tab:r1controls.
- Compiles: 13 pages, main text ends on p.8.
- All numbers recomputed from raw eval files: docs/paper_numbers_verified_oct9.md.
- Measured SERL / GiGPO compute to s34: 12.85 / 10.14 GPU-h, 75.8M / 70.6M backprop tokens. The token ratio is therefore 20–22x, not the 23x estimate.

## Oct 9 17:55 — TextCraft plan2 R0/R1 timing
- At 17:52, 14 processes share 8 GPUs (6 R0 eval shards + 8 R1 collection shards). That gives 17.6 s/task, against 9.1 s/task with a GPU to itself, plus a few OOM restarts.
- ETA: R0 eval ~18:20, R1 collection ~18:35, R1 result ~20:00–20:30.
- Submitted 3 more tcded_gpu jobs (3185163-5) for the later stages, and as insurance against the expiry of igpu hold 3072782 (~19:20).

## Oct 9 19:15 — 7B runs resumed (user: "如果有空的gpu 可以把之前7B的继续跑")
- Free at 19:10: the two A100 holds (agb01 until ~20:50, aga26 until ~21:17). TextCraft used 2 of its 6 dedicated GPUs (Model A stage); the igpu hold expires ~19:18.
- Requested 4 preempt-gpu jobs named a7b_gpu (3191409-3191412, slurm/a7b_dedicated_gpu.sbatch), reserved for 7B. TextCraft's HOLD_NAME_RE does not match them.
- Relaunched the same scripts in tmux, so completed stages are skipped:
  - a7bv3bchain (run_7b_chain_v3b.sh, HOLD_NAME_RE `^(a7b_gpu|error_attr_hold_a100.*)$`; R2 held-out shards 1-2, then R3)
  - rft7b (run_7b_rft_full.sh, `^a7b_gpu$`; R2 retrain, eval, R3)
  - h7on (run_7b_hintsd_online.sh, `^a7b_gpu$`; c2 eval, training to c3)
- A background supervisor agent reports each checkpoint. An attempt to scancel the expiring igpu hold was blocked by the auto-mode classifier; not needed, since it expires by itself.

## Oct 9 night — resumed 7B runs
Paired tests are McNemar on the pooled 774 games (unseen 134 + seen 140 + held-out 500, keyed by game_file), reported as only-A / only-B, two-sided binomial p.
- 19:21 — the first a7b_gpu job (3191409) started, and all 8 waiting RFT and HINT-SD processes picked it at the same moment. They ran out of memory and RFT R2 training logged one TRAIN FAILED. The retry loops sorted it out by 19:27 with no intervention. 3191410 and 3191411 started at 19:23; 3191412 was still pending.
- 19:31 — **a7bv3b_r2 = 576** (96/107/373). It ties a7b_r2 576: 33/33, p=1.0. It is +8 on the 274 (203 vs 195) and -8 on held-out (373 vs 381). vs a7b_r1 549: 80/53, p=0.024 (a7b_r2 vs R1 was 66/39, p=0.011). Below both R3 variants: vs a7b_r3v3d 596, 38/58, p=0.052; vs a7b_r3v3b 602, 32/58, p=0.008. R3 collection started.
- 20:09 — **h7on_c2 = 408** (52/80/276), online HINT-SD at 3,000 episodes. Worse than its own c1 429: 19/40, p=0.0086. Worse than the SFT start a7b_r0 444: 50/86, p=0.0026. vs a7b_r2 576: 18/186, p=2e-36. vs a7brft_r1 475: 54/121, p=4e-7. Same degradation pattern as the 3B online HINT-SD (367/341/344).
- 20:03-20:08 — RFT R2 training and HINT-SD training both picked the newly emptied agc04 (same pick() race). HINT-SD ran out of memory at 20:08. It resumes from state at step 390/563 on the next a7b_gpu with at least 36 GB free. RFT R2 training (restarted from scratch at 20:03) has agc04 to itself.

## Oct 9 22:40 — TextCraft stopped, ScienceWorld StepFix started (user decisions)
TextCraft final results, 810 TEST tasks:
- Plan-every-step R0 = 446; R1 = 357 (vs R0 71/160, p=5e-9, significantly worse).
- Plan3 teacher test: the corrected plan's Need total is right in 17/150 states (11%), below the 50% GO bar, so plan3 R1 was never run.
- Plan3 Model A hint test: the no-obs variant's step-0 share goes from 0.26 to 0.38; the full variant is unchanged.
- The plan3 SFT was preempted at 73%; restarted, then killed at 22:30 (user: all GPUs to ScienceWorld).
- TextCraft is to be written up as an applicability-boundary case.

ScienceWorld:
- Snapping bug fixed (sciworld_agent.py --snap safe; the old fuzzy cutoff=0 mapping snapped invalid text into a fatal 'focus on X').
- student10 on dev 197: 68/197 after the fix vs 11/197 before (paired 59/2, p=2e-15); mean score 44.2 vs -42.8.

ScienceWorld StepFix settings (user-confirmed):
- All 30 tasks, budget 50, R0 = student10.
- Eval set = test (up to 30 per task, 651) + dev 197 = 848.
- 600 train episodes per round; K = 20% of failures; Model A selects the step.
- Generative teacher: 8 samples, safe snap, top-2 by teacher log-prob.
- Generic one-step evidence using the native score; env_top selection.
- Token-CE corrections + 100 replay steps.
- Random-selection control at R1.
- GPUs: all tcded_gpu jobs, ~5 student processes per GPU (6.7 GB each, GPU mostly idle with one).
- Built and run by a background agent; outputs in reports/sciworld/stepfix/.
- 20:16 — a7b_gpu 3191410 was preempted (preempt-gpu) and requeued, killing v3b R3 collection shard s1125, which resumed on agb01. The A100 holds expired at 20:50 and 21:17; the v3b teacher shard s2 killed on agb01 resumed on aga26. 3191412 started at 20:43. Since 21:32 a SciWorld re-eval has shared the a7b_gpu jobs (restarted by main with a 22 GB free-memory minimum).
- 22:23 — **a7bv3b_r3 = 619** (97/115/407), the best 7B result so far. R3 collection: 1,086/1,500 won, 272 failures, 54 margin targets plus 100 fail-weighted replay steps. vs a7bv3b_r2 576: 65/22, p=4e-6. vs a7b_r2 576: 73/30, p=3e-5. vs a7b_r3v3d 596: 43/20, p=0.005. vs a7b_r3v3b 602: 29/12, p=0.012. vs original a7b_r3 540: 89/10, p=6e-17. Chain R1 549 -> R2 576 -> R3 619. The v3b driver finished normally.

## Oct 10 01:20 — spare GPUs: resumed 3B main-chain seeds 1 and 2 (user: "等有多的gpu再补")
- 7B HINT-SD finished: c3 = 417 (curve 429/408/417); 7B RFT R2 = 424. One a7b GPU was idle.
- Relaunched scripts/run_k20_seed_chain.sh 1 and 2 (tmux seed1/seed2) with HOLD_NAME_RE ^a7b_gpu$, GPU_MIN 12000, TRAIN_MIN 26000. They share the a7b GPUs with the 7B RFT R3 and never touch the ScienceWorld tcded GPUs.
- R1 = 557 / 552 already done; R2 collection resumes, then R2 and R3. Same recipe as the main chain (env_top, shortest replay; the teacher-shard bug is fixed).
- ScienceWorld main R1 (relaxed validity + refill) = 299/848, vs R0 267 (64/32, p=0.0014) and vs the random control 269 (64/34, p=0.0032); random vs R0 37/35, p=0.91. Strict-rule R1 = 286 (p=0.037).

## Oct 10 03:20 — overnight plan (user asleep: "确保没有gpu在空转 … 补上还没做的实验（不需要跑multiseed） … 明天把全部实验完成")
ScienceWorld:
- old-recipe chain = 267 → 299 → 308 → 287 (R3 dropped vs R2, p=0.019).
- Running:
  - refined chain r2f → r3f (3B Model A, full-view teacher, margin, fail-weighted replay, fail-proportional correction quota);
  - long-budget eval of 7 long task types (R0, r1b, r2f) right after r2f;
  - ablations: random R2/R3, no-replay R1-R3;
  - RFT baselines: rftm and rftfull R1-R3, separate agent; their GPU threshold was raised to 16 GB.
- Lowest-priority fillers if slots stay free: unfiltered ablation, rollout-verification R1.

7B:
- RFT R3 training restarted 02:55 with resumable checkpoints (new optional CKPT_EVERY env in alfworld_opd_plus.py train, default off; backup alfworld_opd_plus_backup_oct10.py).
- New agent: 7B random-selection and rollout-20% ablations with the NEW recipe (R2/R3 from their own R1s), so they are comparable to the main 7B chain 549/576/619.

GPUs:
- Submitted a7b_gpu 3232901/3232902 (preempt) and 3232903 (interactive), plus tcded interactive replacements 3232904/3232905; the current interactive jobs end ~07:20.
- Overnight GPU-idle watchdog running (scratchpad gpu_idle_watch.sh; alerts on <2 GB used for 15 min or on a job-set change).

## Oct 10 03:30– — 7B ablations re-run with the NEW recipe (R2/R3), background agent
- Driver: scripts/run_7b_ablv3_chain.sh rand|roll (new; copied from run_7b_chain_v3b.sh; tmux abl7b_rand / abl7b_roll). Chains a7b_randv3 / a7b_rollv3, tags a7brandv3_r* / a7brollv3_r*.
  - Recipe: 7B Model A, full-view teacher, fail_weighted replay 100, same trainer (CKPT_EVERY=50), 774 eval.
  - R2 reuses the old ablation's R2 collection (same R1 adapter); R3 collects anew.
  - rand: K = round(0.2·NF) random teacher top-1, Random(100·R). roll: rollout verification with the round's start policy, K random among winners.
- GPU anomalies (all fixed inside the new driver only):
  - (1) The RFT R3 training sits at ~20 GB between its peaks, so GPU_MIN 20000 placed a teacher shard next to it. It was killed within ~2 min; RFT was unaffected. pick() now skips any card that runs a training.
  - (2) The 7B full-view teacher reaches ~30 GB, so two teachers per 40 GB card OOMed. The teacher now needs 32 GB, and the card is exclusive while it runs.
  - (3) Two drivers grabbed the same card in the same minute → picks are serialized (flock) with a 90 s claim.
  - (4) rand has priority over roll for cards.
  - Note: two 7B eval/rollout processes (~20–22 GB each) on one A100-40GB still OOM now and then; the retries resume per game/target.
- 05:18 — **a7brandv3_r2 = 506** (82/90/334). 50 random targets of 251 + 100 replay.
  - vs main a7bv3b_r2 576: 41/111, p=1.2e-8.
  - vs its R1 a7brand20_r1 508: 62/64, p=0.93.
  - (vs old-recipe a7brand20_r2 505: 45/44, p=1.)
- 08:14 — **a7brollv3_r2 = 525** (81/98/346). Rollout check: 258 states, 204 candidates, 137 won, 102 states with a winner; K=52 random + 100 replay.
  - vs main a7bv3b_r2 576: 30/81, p=1.4e-6.
  - vs its R1 a7broll20_r1 524: 62/61, p=1.
  - vs old-recipe a7broll20_r2 569: 30/74, p=1.9e-5 (the new recipe is clearly worse for the rollout baseline).
  - vs a7brandv3_r2 506: 64/45, p=0.084.
- rand-v3 R3 collection (R2 model): 986/1500 won, 329 failures in the first 1000 (R1-policy collection: 1131 won). K=66 random + 100 replay.
- 08:59 — **a7brandv3_r3 = 220** (39/37/144): collapse, below R0 444. Chain: R1 508 → R2 506 → R3 220.
  - vs main a7bv3b_r3 619: 1/400, p=1.6e-118.
  - vs its R2 506: 10/296, p=2.7e-74.
  - vs old-recipe a7brand20_r3 397: 40/217, p=1.3e-30.
  - Checked as genuine:
    - eval files complete and unique (134/140/500);
    - training normal (166 examples, 0 non-finite, the R3 adapter evaluated).
  - Mechanism: "examine" loops.
    - Failed eval episodes have 12.9k examine actions at R3 vs 1.5k at R2; 554 failures hit the 50-step budget at R3 vs 268 at R2.
    - Random teacher top-1 targets contain ~15–20% "examine X" corrections (R2 10/50, R3 9/66). env_top_margin never selects one (R3 diagnostic set: 66/66 "go").
    - Examine rate in the training collections: rand20-R1 policy 6.1%, randv3-R2 policy 10.3%, main v3b R3 0.2%.
    - So the evidence filter screens out no-information corrections, and under random selection they compound over rounds.
- 11:23 — **a7brollv3_r3 = 537** (85/100/352). Chain: R1 524 → R2 525 → R3 537.
  - R3 collection (R2 model): 938/1500 won, 363 failures; 88% "go" steps.
  - Rollout check: 363 states, 206 candidates, 145 won, 103 states with a winner; K=73 + 100 replay.
  - vs main a7bv3b_r3 619: 27/109, p=7e-13.
  - vs its R2 525: 58/46, p=0.28.
  - vs old-recipe a7broll20_r3 558: 50/71, p=0.069.
  - vs a7brandv3_r3 220: 327/10, p=3e-83.
- Summary, 7B with the same new recipe (R1/R2/R3):
  - evidence+margin (main) 549/576/619;
  - rollout-verified random-K 524/525/537;
  - random-K 508/506/220.
  - Both ablation drivers finished normally (11:23).

## Oct 10 11:40– — 7B single-round fillers on the main chain's R2 data (main approved; background agent)
- Driver: scripts/run_7b_v3b_r2_fillers.sh rand|norep|nofilt (new). Outputs in a7b_v3b_fill/<mode>; tags a7bv3b_r2_<mode>.
  - Shared with the main chain's R2: the same 214 7B Model A states, the same full-view teacher candidates, the same fail_weighted replay.
  - Start adapter a7b_final/r1 (549); same trainer (CKPT_EVERY=50) and 774 eval.
  - K = round(0.2·214) = 43, the main chain's K. main had asked for 52, which is the roll-v3 R2 K.
- 12:52 — **(A) matched-state random, a7bv3b_r2_rand = 513** (82/92/339). 43 random teacher top-1 targets (Random(200)) + replay.
  - vs a7bv3b_r2 576: 40/103, p=1.4e-7.
  - vs a7b_r1 549: 20/56, p=4.4e-5 (random selection is worse than not training).
- 12:59 — **(C) no replay, a7bv3b_r2_norep = 396** (69/68/259). The main chain's 43 env_top_margin targets only, all of them "go" corrections.
  - vs a7bv3b_r2 576: 6/186, p=2.1e-47.
  - vs a7b_r1 549: 38/191, p=1.0e-25.
  - Training normal (43 examples, 0 non-finite); eval files complete and unique.
  - Failures that hit the 50-step budget: 378 vs 198 for the main R2 (wandering "go" loops). Replay is essential; this matches 3B no-replay (530/502/275).
- 13:35 — **(B) no filter, a7bv3b_r2_nofilt = 284** (50/50/184). All 214 teacher top-1 targets + replay (315 examples, 0 non-finite).
  - vs a7bv3b_r2 576: 5/297, p=5e-81.
  - vs a7b_r1 549: 8/273, p=4.6e-70.
  - vs (A) 513: 16/245, p=8e-54.
  - Failures that hit the 50-step budget: 490; 349 end in 2-action loops (98.7% "go").
- Summary on the same R2 states/candidates/replay from R1 549:
  - evidence+margin 576;
  - random 43: 513;
  - margin targets without replay: 396;
  - all 214 unfiltered: 284.
  - All three filler drivers finished normally.
