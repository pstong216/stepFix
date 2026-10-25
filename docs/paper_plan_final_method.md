# 论文计划（定稿方法）：动机、文章结构、实验设计

> 状态：2026-10-06 21:30 定稿。方法已确定（见 §3）；R3 正在运行，结果出来后更新 §6。
> 所有数字均为 ALFWorld、Qwen2.5-3B 学生、单种子，除非另外注明。774 = valid_unseen 134 + valid_seen 140 + 训练集留出 500 局（game 1500–1999，从未用于 SFT / 收集 / 训练）。
> 比较一律只比更新后的学生（不和 oracle 比）。

---

## 0. 一句话与摘要草稿

**一句话**：小模型 agent 自我进化时，失败轨迹里信息最多却最难用；我们用一个冻结的、跨领域的 3B 错误归因器找出失败的关键一步，让 agent 自己带着事后信息给出改法，再用**一步**环境反馈筛选，把失败变成步级训练信号——不用更强老师、不用 rollout、不用新标注，全程一个 3B 骨干。

**暂定标题**（二选一）：
- *Fix the Step, Not the Trajectory: Efficient Self-Evolving LLM Agents via Failure Attribution*
- *Learning Where You Failed: Self-Evolving Small Agents with a Transferable Failure Attributor*

**摘要草稿**：
Self-evolving LLM agents improve from their own experience, but small agents mostly learn from their successes: turning a failed
trajectory into a training signal requires knowing *which* step went wrong and *what* should have been done instead, which existing
methods obtain with expensive Monte-Carlo rollouts, expert demonstrations, or a stronger teacher model. We propose a round-based
self-evolution procedure that converts failures into step-level corrections cheaply. A frozen 3B failure attributor, trained once on
public multi-agent failure logs from other domains, locates the decisive step of each failure; the agent itself, given hindsight of
the failed episode and the attributor's diagnosis, proposes alternative actions at that step; and a single environment step, scored
by a task-agnostic evidence template, decides which corrections to keep. The agent is trained on a small budget of verified corrections
plus replay of its own wins. On ALFWorld, a 3B agent improves from 416 to 547 to 572 of 774 games over two rounds (each round significant,
paired McNemar), reaching in two rounds what other variants need three for, while using ~200 training examples per round, no rollouts,
and no model larger than the agent.

---

## 1. 动机（Motivation）

### 1.1 背景
- **Self-evolving agent**：agent 用自己在环境中产生的经验迭代改进（收集 → 产生训练信号 → 训练 → 再收集），不依赖新的人工示范。
- 对**小模型**（3B）尤其重要：小模型是部署成本最低的形态，但起点弱、失败多。

### 1.2 问题：失败轨迹最有信息，却最难变成训练信号
一条失败轨迹要变成监督，需要回答两个问题：
1. **哪一步错了**（credit assignment / 错误归因）；
2. **那一步应该怎么做**（纠正动作），而且这个纠正要可信。

现有路线各有代价：

| 路线 | 代表方法 | 怎么得到步级信号 | 代价 / 问题 |
|---|---|---|---|
| 只学成功 | RFT / STaR 类（rejection-sampling fine-tuning） | 不用失败 | 浪费失败；我们的数据里多轮 RFT 只到 436–443/774 |
| 专家对比 | ETO | 失败轨迹 vs **专家轨迹** 做偏好学习 | 需要目标环境的专家示范，不是纯自进化 |
| MC rollout 步级价值 | IPR / Math-Shepherd 式 | 从每个候选状态 rollout 到结束估计价值 | 每个候选最多 ~50 次 LLM 调用；我们的数据显示 rollout 通过的纠正反而更差（§6.2） |
| 更强老师 | 蒸馏 / 大模型纠错 | 7B/14B/API 模型给纠正 | 不自给自足，不高效 |

### 1.3 我们的关键观察（均有数据支撑，见 §6）
1. **一步就够**：失败往往有一个关键步；只在那个状态给一个 one-hot 纠正目标，~264 条训练样本即可让 774 局从 416 提到 547（R1）。
2. **一步环境反馈比完整 rollout 更好的筛子**：同样数量的纠正，按一步环境证据选的比按 rollout 成功选的好（固定 100 条：508 vs 472，p=0.004；等量 175 条：539 vs 455，p=6e-11）。rollout 成功受学生后续动作的噪声影响，而一步证据直接衡量"这个动作本身是否推进任务"。
3. **小的、跨领域的归因器可迁移**：3B Model A 只在 Who&When Pro（多智能体系统失败日志，**ALFWorld 部分整体留出**）上训练、冻结不更新；它在 ALFWorld 轨迹上的步级准确率只有约 64%，但作为选步器 + 诊断来源，得到目前最好的链（547 → 572）。环境一步检查会过滤掉它指错的地方。

### 1.4 贡献
1. 一个**把失败变成步级纠正**的自进化流程：冻结归因器选步 → 事后自我老师给候选 → 一步环境证据筛选 → 预算化训练（one-hot KL + 自身成功回放）。
2. 一个**任务无关的一步证据模板**（信息 / 相关 / 进展 / 倒退 / 无信息五项，只读环境原生信号），在 ALFWorld 上与任务专用规则 97% 符号一致。
3. 证据：多轮显著增长（R0→R1→R2，774 局配对检验均显著）；与 rollout 验证、RFT、ETO 等对比；消融说明各组件作用。
4. **效率**：全程一个 3B 骨干（策略 LoRA + 归因 LoRA），验证只需每个候选一步环境交互、零额外 LLM 调用；每轮 ~200 条训练样本。

### 1.5 定位：这还算 self-evolving 吗？
- 严格说法：**带一个冻结的跨领域评判器的 self-evolving**。这在文献中很常见（WebRL 训练了单独的结果奖励模型，DigiRL / PAE 用单独的 VLM 评估器，它们都称为 self-evolving / autonomous improvement）——**引用需核对**。
- 我们用的外部信息比上述方法更少、更不贴近目标域：

| | 目标环境人工示范 | 比学生更大的模型 | 额外训练的模块 | 纠正动作来源 | 验证 |
|---|---|---|---|---|---|
| 本文 | 仅 SFT 起点（100 条，与所有基线相同） | 无 | 3B 归因器：一次性在跨领域公开数据上训练，冻结，**未见过 ALFWorld** | 学生自己（带事后信息） | 一步环境交互 |
| RFT | 同上 | 无 | 无 | 学生自己的成功轨迹 | 结果奖励 |
| ETO | 专家轨迹 | 无 | 无 | 专家 | 结果奖励 + 专家对比 |
| rollout 验证 | 同上 | 无 | 无 | 学生 | MC rollout |

- 推荐表述：**"a task-agnostic failure-attribution prior: trained once on public multi-agent failure logs, frozen, plugged into any environment"**。
- 加强论证的两个实验：(a) **完全自给版**：用学生自己找错步替代 Model A（消融 A4）；(b) **第二个环境**（AppWorld）用同一个冻结 Model A（§5.9）。

---

## 2. 文章结构

| 节 | 内容 | 主要图表 |
|---|---|---|
| 1 Introduction | §1 的动机、观察、贡献 | Fig.1：方法总览（一轮的四步） + 774 局 R0→R1→R2→R3 曲线 |
| 2 Related Work | self-evolving agents（RFT/STaR, ETO, IPR, WebRL, DigiRL, PAE, Self-Rewarding）；failure attribution（Who&When）；步级奖励 / PRM / MC 标注（Lightman et al., Math-Shepherd）；hindsight 与 context distillation；高效训练 | — |
| 3 Problem Setup | 环境、轮次、预算、评测协议（§3.1） | — |
| 4 Method | 4.1 一轮流程（算法框）；4.2 冻结归因器选步；4.3 事后自我老师；4.4 一步环境证据模板；4.5 预算化选择与训练；4.6 成本分析 | Alg.1；Table：证据模板五项定义 |
| 5 Experiments | 5.1 设置；5.2 多轮主结果；5.3 与基线比较；5.4 消融；5.5 效率；5.6 分析；5.7 第二环境 | Table 1 主结果；Table 2 基线；Table 3 消融；Table 4 成本；Fig.2 选步位置分布；Fig.3 质量-成本散点 |
| 6 Discussion & Limitations | 选步偏向第 0 步、单环境/单骨干、SFT 起点、WebShop 边界案例 | — |
| Appendix | 证据模板细节与例子；提示词全文；超参；每轮统计；完整配对检验表；WebShop / ScienceWorld 结果 | — |

---

## 3. 方法（定稿，精确描述）

### 3.1 设置与记号
- 环境：ALFWorld（文本版），每局步数上限 50；动作从当前**可执行命令集**（admissible commands）中选。
- 学生 π_θ：Qwen2.5-3B-Instruct + LoRA（r=32, α=64, dropout 0.05, 全部线性投影），R0 = 在 100 条专家轨迹上 SFT 的起点（`reports/alfworld/student3b_low100/final`）。每轮在同一个 LoRA 上继续训练。
- 轮次 r = 1, 2, 3, …；第 r 轮从上一轮模型 π_{r-1} 出发。

### 3.2 一轮流程（Algorithm 1）
```
输入: 上一轮学生 π_{r-1}，冻结归因器 A（3B Model A）
1. 收集: π_{r-1} 在训练局 0–1499 上各玩一局（温度 1.0，上限 50 步）
         （R1 例外：复用 SFT 学生已有的收集，1323 局）
2. 失败集合 F = 前 1000 局中的失败局
3. 选步: 对每个 τ ∈ F，A 输出关键错误步 t*(τ) 和错误类型
4. 事后老师: π_{r-1} 在状态 s_{t*} 上，带着 [整条失败轨迹的后续(hindsight) + A 的诊断 + 自己写的分析]，
           对所有可执行命令打分；候选 = 除原动作 a⁻ 外得分最高的 2 个命令
5. 一步证据: 对每个候选 c，重放前缀到 s_{t*}，执行 c 一步，按通用模板打分 score(c)
6. 选择: 每个状态保留 score 最高且 score>0、非"无信息"的候选；所有状态按 (score, 老师概率) 排序，
        取前 K = round(0.4 × |F|) 个
7. 训练: K 条 one-hot KL 纠正 + 100 条自身成功回放，1 个 epoch，得到 π_r
8. 评测: 774 局，贪心
```

### 3.3 选步：冻结 3B 归因器（Model A）
- 模型：`reports/attributor_no_observations/final`，Qwen2.5-3B-Instruct + LoRA（r=16, α=32），3 epoch，lr 1e-4。
- 训练数据：Who&When Pro 官方文本版（6,257 条，8 个框架），**ALFWorld 的 311 条整体留出**；其余 7 个框架 90/10 分层切分，实际训练 5,016 条（≤8,192 token）。
- 输入格式："no_observations"——删去环境/工具返回的 `[observation]` 块，保留 agent 的思考、动作、输出。这一格式使 ALFWorld 留出集步级准确率从 53.4% 升到 64.3%（7B 同格式版本为 91.3%）。
- 输出：关键错误步编号 t̂ 和 17 类错误模式之一；t* = clip(t̂, 0, |τ|−1)。
- 输入的轨迹是学生的失败轨迹（release 格式）。Model A **从不更新**。

### 3.4 事后自我老师（hindsight self-teacher）
- 老师 = **当前学生 π_{r-1} 自己**（同一个 LoRA）。
- 两次调用：(1) 生成一段分析（ANALYSIS_Q 提问），(2) 在附加 [诊断 + 分析] 和 hindsight（失败局在 t* 之后发生了什么）的上下文中，对 s_{t*} 的所有可执行命令打分（log-prob）。
- 候选：排除原动作 a⁻ 后得分最高的 2 个命令。
- 诊断文本格式：`Diagnosis (...): the first decisive error was at step {t̂}, type: Model-A mode {error_mode}`。
  - **实现注意**：提示中描述归因器的旧措辞是 "an error attributor trained on this agent's own trajectories"，与 3B Model A 的真实来源不符。论文中照实描述；若重跑全部主实验，可统一改为中性措辞（需在所有臂上一致）。

### 3.5 一步环境证据（通用模板，所有环境共用）
对候选 c：重放失败轨迹前缀到 s_{t*}，执行 c 一步，得到新观察 o'。内容单元 = 带编号的物体实例（如 "soapbar 4"）或去停用词后的词。

| 项 | 定义 |
|---|---|
| n_new | o' 中此前（初始观察 + 所有已见观察）没出现过的内容单元个数 |
| relevance | 某个新单元的首词出现在任务描述中 |
| progress | 环境原生分数上升，或成功结束，或环境确认了状态改变（"You \<verb\> the X"）且 X 在任务描述中，且不是同一动词作用于同一实例的重复（防止拿起-放下循环） |
| regress | 原生分数下降，或未成功结束 |
| no_info | 观察不变 / 与之前某个观察相同（回访、撤销）/ "Nothing happens" / 无新单元且无进展 |

**score = 2·relevance + 2·progress − 4·regress + min(n_new, 5)/5 − 2·no_info**

- 每个环境只改 `observe`：怎么去掉固定样板（如 WebShop 每页重复指令）、原生分数是什么（ALFWorld 无 → 0）。
- 与 ALFWorld 专用证据规则符号一致 97%；R1 top-100 选择 96/100 相同，R2 97/100 相同。
- 一步执行在 CPU 上进行，不调用任何 LLM。

### 3.6 预算化选择
- K = round(0.4 × |F|)，|F| = 本轮前 1000 局中的失败数（R1：410 → K=164；R2：274 → K=110）。随学生变强，纠正数自动减少。
- 回放固定 100 条：本轮自己赢的局中最短的 50 局、每局 2 步（跳过 look / inventory / help），SFT 目标 = 当时采取的动作。

### 3.7 训练
- 纠正样本损失：在可执行命令集上归一化的学生分布 q_θ 与 one-hot 目标 p 的 KL（等价于对目标命令的交叉熵）：L = Σ_c p(c)[log p(c) − log q_θ(c|s)]。
- 回放样本损失：动作文本的 token 级 NLL。
- AdamW，lr 1e-5，batch 8，1 epoch，权重均为 1.0；梯度检查点；**逐样本反传 + fp32 梯度累积 + 非有限梯度跳过**（旧的 bf16 整批累积会因单个坏样本丢掉整批）。

### 3.8 每轮成本（R1/R2 实测）
| 项 | R1 | R2 |
|---|---|---|
| 收集 | 复用 1323 局 | 1500 局 × ≤50 步（3 GPU 并行约 75 min） |
| Model A 调用 | 每个失败 1 次生成 | 424 次 |
| 老师调用 | 410 状态 × 2 | 274 状态 × 2 |
| 一步执行 | 820 个候选 × (前缀重放 + 1 步)，CPU | 约 548 个候选 |
| 训练样本 | 164 + 100 = 264 | 110 + 100 = 210 |
| 训练时间 | ≈21 min（单 GPU） | ≈15–20 min |
| 评测 | 774 局 | 774 局 |

---

## 4. 研究问题与对应证据

| RQ | 问题 | 对应实验 | 状态 |
|---|---|---|---|
| RQ1 | 方法能否多轮持续显著提升？ | 主链 R0→R3（§5.4） | R1、R2 完成；R3 进行中；3 种子待做 |
| RQ2 | 比现有自进化 / 步级方法好吗？ | 基线 B1–B5（§5.5） | 部分有旧数字，需按定稿协议重跑 |
| RQ3 | 每个组件贡献多少？Model A 是选步有用还是信息有用？ | 消融 A1–A6, C1–C2, D1, E1（§5.6） | 待批准 |
| RQ4 | 效率如何？ | 成本表（§5.7） | 部分实测，需补 rollout 臂的计数 |
| RQ5 | 能否迁移到其他环境？ | AppWorld（§5.9） | 方法定稿后开始 |

---

## 5. 实验设计

### 5.1 环境与数据划分（ALFWorld）
| 用途 | 局 | 说明 |
|---|---|---|
| SFT 起点 | 100 条专家轨迹 | 所有方法共用同一个 R0 |
| 每轮收集 | 训练局 0–1499 | 失败集取前 1000 局；回放取全部 1500 局中的成功 |
| 测试 | valid_unseen 134 + valid_seen 140 | "274" |
| 留出测试 | 训练局 1500–1999（500） | 从未用于 SFT/收集/训练；与 274 合为 "774" |

### 5.2 模型
- 学生：Qwen2.5-3B-Instruct + LoRA（所有方法、所有基线相同）。
- 归因器：3B Model A（主方法）；7B Model A 只用于尺寸消融。
- 老师：学生自己（主方法）；更大老师只出现在效率对比基线 B5。

### 5.3 评测协议与统计
- 主指标：**774 局成功数**（贪心解码，上限 50 步）；同时报告 274 和 500 两个子集。
- 显著性：**配对精确 McNemar 检验**（不一致对上的双侧二项检验），报告"A 赢 B 输 / B 赢 A 输"的局数和 p 值。
- 功效：274 局只能检出约 ≥25 局的差距；所以结论以 774 为准。
- 种子：主方法和关键对照（A3、C1、B1、B2）跑 3 个种子（种子影响收集采样、回放抽样、训练顺序），报告均值 ± 标准差 和逐种子检验；其余消融单种子。
- 多重比较：消融表内用 Holm 校正，正文同时给出原始 p 值。
- 轮次：主链至少到 R3；消融和基线至少 R1+R2（自进化的核心主张是跨轮增长）。

### 5.4 主实验：多轮链（RQ1）
- **主链**：定稿方法（§3），R0 → R1 → R2 → R3，3 个种子。
- 报告：每轮 774 / 274 / 500 成功数；相邻轮配对检验；每轮训练局胜率、|F|、K、选步位置分布。
- 已完成（种子 0）：774：416 → **547 → 572**（R1 vs R0：165/34，p=8e-22；R2 vs R1：50/25，p=0.005）；R3 进行中。

### 5.5 基线（RQ2）——全部从同一 R0 出发、同样每轮 1500 训练局、同样 774 评测、至少 R1+R2

| # | 基线 | 精确定义 | 公平性说明 | 现有数字 |
|---|---|---|---|---|
| B0 | SFT 学生 | R0 | — | 416 |
| B1 | RFT（STaR 式） | 每轮用学生自己成功轨迹做 SFT（步数匹配或全量两种） | 最纯的自进化基线 | 旧：436 / 443，需按定稿协议重跑多轮 |
| B2 | ETO（按论文设置精确复现） | SFT → 贪心探索 → 专家 vs 失败轨迹配对 → DPO（β=0.5, lr 1e-6, batch 32, 3 epoch） | 使用目标环境专家轨迹（比我们多的监督），需在文中注明；专家来自训练局 0–999 的规划器示范 | 旧版不公平（胜轨迹来自后期链），复现脚本 `scripts/run_eto_fair.sh` 已写好、暂停 |
| B3 | Rollout 步级验证（IPR / MC 式） | 与主方法同选步、同老师候选；用 rollout 到结束是否成功来选纠正；两个版本：等量 K、以及全部通过者 | 完整 rollout（不截断 20 步） | R1（旧选步）：全部通过 455、随机 100 条 472 |
| B4 | 不加筛选的自我纠正 | 同选步、同老师，取老师 top-1，不用一步证据，随机 K | 隔离"筛选"的作用 | 旧：all410 398 |
| B5 | 更强老师 | 同选步，候选由 7B/14B 老师给出，其他不变 | 效率对比：我们的 3B 自我老师能否接近大老师 | 旧配方下有 14B 老师结果，需重跑 |

### 5.6 消融（RQ3）——每次只改一个组件，其余与定稿完全相同，R1+R2，774 评测

**P0（必须）**
| # | 改动 | 回答的问题 | 现有相关证据 |
|---|---|---|---|
| A1 | Model A 选步，老师**不给**诊断（hindsight + 自己分析） | 选步本身的作用 | — |
| A2 | 环境裁决选步（无 Model A），老师**给** Model A 诊断 | 诊断信息本身的作用 | R2 单轮（同一首步规则 R1、新训练器）：7B 诊断 536 vs 无 Model A 533（p=0.84），提示信息单独作用小 |
| A3 | 完全无 Model A（环境裁决选步 + hindsight 老师），同 40% 预算、同新训练器 | 隔离 Model A；消除训练器/预算混杂 | 主线（R1 固定 100、R1/R2 旧训练器）：524 → 529 → 563 |
| A4 | 学生自己找错步（紧凑视图 "step: action → obs"，`scripts/self_point_step.py`）替代 Model A | 完全自给版本；支撑 self-evolving 论述 | R2 平均指向第 8.7 步，未训练 |
| C1 | 一步证据 → rollout 验证（= B3，等量 K） | 一步 vs rollout | 见 B3 |
| C2 | 无筛选（老师候选随机取 K） | 筛选的作用 | 见 B4 |

**P1（应该）**
| # | 改动 | 回答的问题 |
|---|---|---|
| A5 | 7B Model A 替代 3B | 归因器尺寸（"小就够"） |
| A6 | 随机步 / 首个非正步规则 | 选步的下界 |
| D1 | 去掉回放 | 回放防遗忘的作用 |
| E1 | 固定 K=100 vs 40% | 预算规则 |
| T1 | 老师去掉 hindsight（只给诊断 + 分析） | 事后信息的作用（早期 3B 选步链用了无 hindsight/无诊断/无分析的老师 + top-86 + 无回放：492 → 538） |

### 5.7 效率分析（RQ4）
对每个方法记录：
| 指标 | 定义 |
|---|---|
| 环境步数 | 收集 + 验证（一步：Σ(t*+1)；rollout：Σ 每个候选 rollout 的步数） |
| LLM 调用 | 归因 + 老师 + rollout 中的策略调用 |
| GPU 小时 | 每轮墙钟 × GPU 数 |
| 训练样本数 | 每轮 |
| 最大模型尺寸 | 流程中出现的最大模型 |
| 774 成功数 | 每轮 |
图：x = 每轮 GPU 小时（或 LLM 调用数），y = 774 成功数，每个方法一条轨迹（R0→R1→R2）。

### 5.8 分析
1. **选步位置分布**：每轮 t* 的分布（R1：t0 占 66%；R2：t0 占 70%，平均 5.2）；被选入训练的纠正中 t0 占 79%（R1）/ 77%（R2）。与主线对比（主线 R3 t0 39% 时上升）。
2. **Model A 准确率 vs 收益**：Model A 在 ALFWorld 上只有约 64% 步级准确率，但 3B 链优于 7B 诊断链——分析一步筛选如何过滤错误指向（被选中的纠正里 Model A 指向与环境证据一致的比例）。
3. **纠正类型**：被选纠正的动作类别（go to / take / open / put …）及其在下一轮是否被学生采用。
4. **一步证据 vs rollout 的分歧案例**：rollout 判失败但一步证据判有进展的纠正，在训练后的效果。
5. **每轮失败模式变化**：Model A 错误模式分布随轮次的变化。

### 5.9 第二个环境（RQ5）
- **AppWorld**（方法定稿后开始）：同一冻结 3B Model A、同一通用证据模板（只改 `observe`）、同一预算规则；学生可能需要 7B coder 起点；约 90 个训练任务——需按任务数调整收集量。
- **WebShop**（附录，边界案例）：通用模板偏向"下一页"（9/200）；按指令+观察做任务实例化的 v2 证据 87/200、平均奖励 0.645（学生 86 / 0.598；rollout 56）——成功数不变、奖励上升。定稿后用 v2 实例化 + 定稿配方 + rollout 对比重跑，报告成功率与平均奖励。
- **ScienceWorld**（附录）：第一轮负结果（学生 8/197；没有任何臂提升成功数）。

### 5.10 执行顺序与算力（4 块 GPU 预估，均需逐项批准）
| 优先级 | 实验 | 估计墙钟 |
|---|---|---|
| 进行中 | 主链 R3（种子 0） | ≈2 h |
| P0-a | A3（无 Model A 干净对照）R1+R2 | ≈3 h |
| P0-b | A1、A2（Model A 2×2）R1+R2 | ≈5 h |
| P0-c | C1 / B3（rollout，等量）R1+R2 | ≈4 h（rollout 更慢） |
| P0-d | B1 RFT、B2 ETO R1+R2 | ≈5 h |
| P0-e | 主链种子 1、2（R1–R3） | ≈8 h |
| P1 | A4、A5、A6、D1、E1、T1、C2 | ≈12 h |
| 之后 | AppWorld；WebShop 重跑 | — |

---

## 6. 当前结果（种子 0）

### 6.1 多轮链（774 局；括号内为 274 局）
| 链 | 配置 | R1 | R2 | R3 |
|---|---|---|---|---|
| **定稿 + 3B Model A** | 3B Model A 选步 + 诊断；hindsight 自我老师；一步证据 top 40%；回放 100；新训练器 | **547 (196)** | **572 (205)** | 进行中 |
| 主线 | 环境裁决选步，无 Model A；R1 固定 100 + 回放 100、R2 40% + 100（均旧训练器）；R3 40% + 100（新训练器） | 524 (181) | 529 (186) | 563 (194) |
| 旧 7B 链（10-04 配方） | M5 进度估计器选步 + 7B Model A 诊断；top-86 + 回放 86；旧训练器 | 475 (166) | 531 (189) | 574 (210) |
| 早期 3B 选步链 | 3B Model A 选步；老师无 hindsight/诊断/分析；top-86；无回放 | 492 (168) | 538 (192) | — |
| 首步规则 R1 → 环境裁决 | R1 首个非正步规则（K=100）；R2 环境裁决 40% + 100（旧训练器）；R3 同（新训练器） | 508 (178) | 536 (189) | (158，274 局，显著下降) |
R0（SFT 学生）= 416 (136)。

**定稿链配对检验（774）**：
| 比较 | 只赢 / 只输 | p |
|---|---|---|
| R1 vs R0 | 165 / 34 | 8e-22 |
| R2 vs R1 | 50 / 25 | 0.005 |
| R1 vs 主线 R1 | 78 / 55 | 0.056 |
| R1 vs 首步规则 R1 | 91 / 52 | 0.001 |
| R1 vs 环境裁决 R1（同为 164 条纠正，但回放 164、旧训练器，505） | 82 / 40 | 0.0002 |
| R2 vs 主线 R2 | 83 / 40 | 0.0001 |
| R2 vs 旧 7B 链 R2 | 88 / 47 | 0.0005 |
| R2 vs 早期 3B 选步链 R2 | 66 / 32 | 0.0008 |
| R2 vs 主线 R3 | 60 / 51 | 0.45 |
| R2 vs 旧 7B 链 R3 | 58 / 60 | 0.93 |

### 6.2 筛选方式（R1，早期选步、无 Model A；774）
| 筛选 | 纠正数 | 774 |
|---|---|---|
| 一步证据 top-175 | 175 | 539 |
| 一步证据 top-100（环境裁决选步） | 100 | 524 |
| 一步证据 top-100（首步规则） | 100 | 508 |
| rollout 通过者中随机 100 | 100 | 472 |
| rollout 全部通过者 | 175 | 455 |
固定 100：508 vs 472，p=0.004；等量 175：539 vs 455，p=6e-11。旧配方（top-86 时期）还有：老师置信度 top 393、无筛选全部 410 条 398、随机动作 placebo 314。

### 6.3 选步（R1/R2，无 Model A，274 / 774）
- R1：环境裁决 181 (524)、首步 178 (508)、随机 171 (506)、晚选 152。
- R2（40%，同一首步规则 R1，新训练器）：环境裁决 189 (533)、晚选 163（vs 环境裁决 p=0.0002，显著更差）。
- 训练器本身：同一 R1、同一份 R2 训练数据，旧训练器 536 vs 新训练器 533——新训练器只是避免了 NaN 丢批，本身不带来提升。
- 所有非正步的一步分数都是 −2，累积证据类方法（PFF / CUSUM）退化，不可用。

### 6.4 基线（旧协议，需重跑）
- RFT：436 / 443；ETO（旧，不公平）：492 / 538 / 516；对比式 one-hot：532。

---

## 7. 风险与待解决问题
1. **R1 混杂**：定稿链 R1 同时改了选步（Model A）、预算（40%）、训练器（新）。已有证据表明训练器本身不带来提升（同数据 536 vs 533，§6.3），但干净结论仍需 A3 消融。
2. **选步偏向第 0 步**：R2 被选纠正 77% 在 t0。主线在 t0 比例降到 39% 的那轮才上升；若定稿链 R3 停滞，候选方案是 R3 起叠加晚选规则（需另行批准）。
3. **单种子**：所有链目前只有种子 0；"首步规则 R1 → 环境裁决" 链在 R3 显著下降（189 → 158，274 局），说明链与链之间方差不小——主结果必须 3 种子。
4. **ETO 公平复现**尚未完成。
5. **通用性**：目前只在 ALFWorld 上有正结果；WebShop 为边界案例、ScienceWorld 为负结果。AppWorld 是关键。
6. **老师提示词中归因器的旧措辞**（§3.4）需要在论文里照实写或统一修正。
7. **引用核对**：§1–2 中提到的 WebRL / DigiRL / PAE / IPR / ETO / STaR 等的具体设定需在写作时逐一核对原文。

---

## 附录 A：复现路径
| 组件 | 路径 |
|---|---|
| 一轮脚本 | `scripts/self_attr_round.sh`（`EVIDENCE=generic K_FRAC=0.4 REPLAY_N=100 MODELA_PATH=... PAR=3`） |
| 定稿链驱动 | `scripts/run_a3b_final_chain.sh`（R1+R2），`scripts/run_a3b_final_r3.sh`（R3） |
| Model A 推理 | `scripts/alfworld_attribute.py --attributor reports/attributor_no_observations/final --variant no_observations` |
| 事后老师 | `scripts/self_teacher_informed.py` |
| 一步执行 / 筛选 | `scripts/proxy_eval_se3b.py onestep`，`scripts/proxy_arms_se3b.py --rule env_top` |
| 通用证据模板 | `scripts/generic_evidence.py` |
| 回放 | `scripts/make_replay.py` |
| 训练 | `scripts/alfworld_opd_plus.py train` |
| 774 评测 | 274：每轮 `eval_unseen/seen.jsonl`；500：`reports/alfworld/ext/eval_heldout_<tag>.jsonl` |
| 配对检验 | `scripts/paired_compare.py` |
| 结果目录 | `reports/alfworld/self_attr_rounds/selfattr_a3bfinal/r{1,2,3}` |
