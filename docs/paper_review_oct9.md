# 论文草稿检查（2026-10-09 17:40）

草稿位置：`/scratch.global/pan00389/error-attribution/paper/overleaf_draft`（Overleaf git，本地未改、未 push）。
数字来源：`docs/paper_writing_guide.md` §9、`docs/compute_accounting_oct9.md`、`reports/alfworld/self_attr_rounds/queue.log`。

## 一、必须改

1. **删除所有起草备注**（审稿人会读成"结果未核实"）：
   - `\pending`：tables/baselines.tex:14–16，appendix.tex:58–61
   - experiments.tex:12 整段 Pending comparisons（含 ETO，已决定不做）
   - appendix.tex:106 整节 Pending Experiments
   - limitations.tex:8 "Aggregate results supplied … require reconciliation"
   - appendix.tex:64 "supplied rounded summaries"
   - appendix.tex:29 / :37 "remain to be documented / must be recovered"
   - appendix.tex:96 "Planned controls"（承诺不做的实验）
2. **数字过时**：摘要、introduction.tex:13、结论只写到 416→537→561。主表 R3 列：
   - 774：591†
   - 验证 274：210（unseen 96 + seen 114）
   - 留出 500：381
   - 纠正 50，回放 100，前 1000 局失败 249
   - R3 vs R2 55/25，p=0.001；vs R0 206/31，p=7e-33
3. **主张随 SERL/GiGPO 改。** 配对数均为"我们只赢/对方只赢"：
   - SERL：R1 447（135/45，p=1e-11），R2 532（90/61，p=0.022），R3 596（58/63，p=0.72）
   - GiGPO：R1 475（111/49，p=1e-6），R2 538（83/60，p=0.065），R3 603（67/79，p=0.36）
   - 可以说：
     - 一轮（1,323 局）到 537；在线 RL 按检查点需要 1.7–2.5 倍局数（GiGPO 2,304 局 559；SERL 3,328 局 540）。
     - 三轮预算持平。
     - 反向传播 token 少 23 倍（3.5M vs 约 80M）。
   - 不能说：
     - 全面优于在线 RL：574 vs GiGPO 68/97，p=0.029。
     - 更省 GPU 小时：11.1 vs GiGPO 10.1。
4. **方法与实现不一致**：
   - method.tex:17 "recorded reasoning"：ALFWorld 学生没有推理文本，Model A 看任务和动作序列（去掉观察）。
   - method.tex:49 "Each environment must implement its own condition extraction and observation parser" 和 tables/conditions.tex 的 ALFWorld 一列，与实现不符。所有 ALFWorld 结果都用 `EVIDENCE=generic`（scripts/generic_evidence.py），不解析任务：
     - loc = 新出现的词出现在任务描述里；
     - sat = 环境分数上升 / 成功，或环境确认 "You <verb> the X"（X 是任务中的物体，且不重复）；
     - nul = 观察不变 / 回到旧观察 / "Nothing happens" / 没有新词且无进展；
     - vio = 环境分数下降或未成功结束。
   - method.tex:59 的序数排序检查只在 40% 链上做过，主线是 20% 链。
   - 已核对、一致的部分：
     - S 公式、S>0 保留规则、平局规则（proxy_arms_se3b.py env_top）；
     - ρ=0.2，K=82/54/50；回放 50 局最短胜局 × 2；LoRA 设置；
     - 纠正用可行命令上的交叉熵、回放用动作 token NLL（alfworld_opd_plus.py:155–162）；
     - McNemar 公式。
5. **必须披露**：
   - † 591 的候选池只覆盖 171/249 个状态。全池重做 574：
     - 574 只赢 / 591 只赢 = 27/44，p=0.057；
     - 574 vs 学生 192/34，p=6e-28；vs RFT 95/45，p=3e-5；vs rollout-40% 111/34，p=1e-10；
     - 574 vs 随机 20% 140/28，p=4e-19；vs SERL 60/82，p=0.08；vs GiGPO 68/97，p=0.029；
     - R2→R3（574）48/35，p=0.19；R1→R3 65/28，p=2e-4。
   - 老师贪心解码跨 GPU 不逐位复现。
   - SERL/GiGPO 训练时生成自由文本动作，评测与我们同一可行命令贪心评测。SERL 移植修过 LoRA→vLLM 同步 bug。硬件不同（L40S / A100）。三个在线基线都只用 0–1499 号训练局（脚本强制检查，已核对）。
   - 全部单种子。
   - WebShop 已评测且无提升：R0 305/700 → R1 297/700（p=0.40）；dev 86→85→51→40。limitations.tex:4 和 conditions 表的"未评测"不再准确。

## 二、表和图

- 基线表（三轮）：
  - 匹配 RFT 481/513/524；标准 RFT 431/428/446
  - rollout-40% 508/485/497；rollout 20%+回放 483/390/435
  - 在线 HINT-SD 367/341/344；按轮 HINT-SD 381/358/337
  - SERL / GiGPO 见上
- 消融：无回放 530/502/275；随机 20% 505/458/462；不筛选 396/350/304
- R1 机制分析：
  - rollout 拆分：pass82 = 548 vs 随机通过者 483（p=3e-8）；rej82 = 498
  - SFT 对照：只 SFT 1437 步 = 420；SFT + 82 纠正 = 503
- 7B（R0 444）：
  - 我们 549/576，R3 原配方 540；改进 R3 596（3B Model A）/ 602（7B Model A）
  - 随机 20% 508/505/397；rollout 20% 524/569/558
  - R3 配方与 R1/R2 不同；无在线 RL 基线
- 成本（实测）：
  - 到 R3 的 GPU 小时：我们 11.1（共享折算 7.2），SERL 13.4，GiGPO 10.1，HINT-SD 7.8
  - 验证新增 1,704 个候选环境步（占收集 1.6%），含前缀重放 10,120
  - rollout 验证三轮约 9.2 万环境步、8.7 GPU 小时
- 学习曲线：加 R3，SERL / GiGPO 各 9 个检查点，以及 HINT-SD
- 附录：补 40% 那四行的配对数；位置统计换成 20% 链

## 三、写作

- 每个结论只留一句限定语。
- 相关工作：
  - 删 related_work.tex:10 "pending"；
  - 加 GiGPO；
  - 加 CSO（ACL Findings 2026）作为最接近的工作，并写明区别；
  - HINT-SD 只在 BFCL/AppWorld 上评测。
- 审稿风险：第 0 步集中，且没有固定第一步对照。用 20% 链步位分布和 rollout 拆分实验回应。
- appendix.tex:34 剩余 930 条：写清过滤原因或删掉。
