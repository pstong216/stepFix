# 报告：错误归因器的训练，以及它对多智能体系统自纠正的作用

日期：2026-09-09。数据、脚本、逐条决策见 `STATUS.md`、`TODO.md`、`DECISIONS.md`（#12–28）。
本报告按六个部分组织：(1) 归因器训练结果（SFT / GRPO）；(2) 用于纠错实验的多智能体系统；
(3) 系统初始失败率；(4) 归因器对失败轨迹的回应；(5) 回应如何交给系统；(6) 语气与角色如何改变系统行为。
所有数字均为修正两个判分 bug（DECISIONS #25/#26）后的最终值；每个实验都附一个真实例子。

---

## 1. 错误归因器（Model A）的训练结果：SFT 与 GRPO

### 1.1 设置

- 模型：Qwen2.5-3B-Instruct + LoRA（r=16, α=32，全部 linear 投影），bf16，3 epoch，lr 1e-4，序列上限 8,192 token（超长样本丢弃不截断）。
- 数据：Who&When Pro 官方文本轨迹（`Leoxx/whowhen_pro`，6,257 条，8 个框架）。**ALFWorld（311 条）整体留出**作为迁移测试；其余 7 个框架按框架分层 90/10 切分 → train 5,353（≤8,192 token 的 4,767 条实际参与训练）、val 593。
- 输入：官方 `all_at_once` 裁判 prompt（任务说明 + 17 类错误分类表 + 用户问题 + 完整轨迹 + 步骤坐标说明 + 输出格式）。损失只算在目标 token 上。
- 输出：三行（Agent Name / Step Number / Error Mode）。官方 `whowhen_eval.score` 按框架各自的规则判分（例如 debate/dylan 只比对轮次，mathchat 用 2·round+position）。

### 1.2 一条 SFT 训练样本（`math_debate_0122`，prompt 24,212 字符）

**prompt 开头（任务说明 + 分类表节选）**

```
# Task

You are an expert at diagnosing failures in agentic systems.

You will be given the transcript of an agentic system attempting to answer a user question.
The system failed because of a decisive error somewhere in the transcript. Your job is to
identify the first decisive error: the step that most directly causes the system to go wrong
and eventually produce an incorrect answer.

Report which agent made that decisive error, the exact step coordinate where it occurred, and
the best matching error mode from the taxonomy below. Then briefly explain your reasoning.

## Error Mode Taxonomy

- P.1: **Visual Misidentification** - Wrong object, entity, or text recognition in image input. ...
- R.1: **Hallucination** - Generating claims that are NOT grounded in any observation the agent
  has retrieved. ... Distinguished from R.2: in R.1 the wrong claim is ABSENT from all
  observations; in R.2 the wrong claim is traceable to something the agent actually saw but misapplied.
- R.2: **Reasoning Error** - The agent has the right high-level frame but makes a localized
  reasoning mistake ...
  (共 17 类：P.1–P.2 感知, R.1–R.4 推理, PL.1–PL.2 规划, A.1–A.4 行动, V.1–V.2 验证, C.1–C.3 协作)
```

**prompt 中段（问题 + 轨迹，每一步一个带坐标和 agent 名的块）**

```
## User Question

Let $a,$ $b,$ and $c$ be positive real numbers such that $a + b^2 + c^3 = \frac{325}{9}.$
Find the minimum value of \[a^2 + b^3 + c^4.\]
After your reasoning, enclose your final answer in \boxed{}. For example: \boxed{42}

## Transcript

Step 0.0 | Agent: agent_0
[output]
Given \( a + b^2 + c^3 = \frac{325}{9} \), we are to find the minimum value of ...
Let's use the method of **Lagrange Multipliers**. ...
[/output]
Step 0.1 | Agent: agent_1
[output] ... [/output]
Step 0.2 | Agent: agent_2
[output] ... [/output]
Step 1.0 | Agent: agent_0
[output]
Let's carefully solve the problem using the sounder approach suggested by Agent 2. ...
[/output]
...
```

**prompt 结尾（坐标格式 + 输出格式）**

```
## Step Coordinate Format

step R.P where R is the round and P is the turn's position within that round. The agent
cannot see the other agents' turns in the same round.

## Response Format

Please answer in the following format, exactly:
Agent Name: (the agent ID whose turn first introduces the error)
Step Number: (the step coordinate, exactly as used in the conversation above)
Error Mode: (one of the error modes listed)
Reason: (one or two sentences explaining the error)
```

**训练目标（target，只对这三行计算损失）**

```
Agent Name: agent_2
Step Number: 1.0
Error Mode: C.3
```

原始金标为 `{"agents": ["agent_2", "agent_1"], "round": 1, "mode": "C.3"}`（C.3 = 过度依赖其他 agent）。
debate 只按轮次判分，所以目标里的坐标取金标轮次的第一个坐标 `1.0`。

### 1.3 SFT 结果

| 模型 | 数据 | agent | step | mode | 三项全对 |
|---|---|---|---|---|---|
| 未微调 Qwen2.5-3B | val（域内，n=593，不含 ALFWorld） | 77.6 | 39.5 | 16.9 | 4.9 |
| 未微调 Qwen2.5-3B | ALFWorld（n=311） | 100 | 8.4 | 18.6 | 2.9 |
| 未微调 Qwen2.5-7B（同一 prompt 当零样本裁判） | val | 90.6 | 62.6 | 16.0 | 9.6 |
| 未微调 Qwen2.5-7B | ALFWorld | 100 | 42.1 | 19.9 | 12.5 |
| **SFT 3B** | **val（域内，n=593）** | 97.6 | 95.6 | 82.8 | **81.6** |
| **SFT 3B** | **ALFWorld（留出迁移，n=311）** | 100 | 53.4 | 26.0 | **20.3** |

val = 7 个训练框架的 10% 留出（smolagents 327 / mathchat 92 / magentic-one 81 / debate 44 / macnet 26 / dylan 15 / metagpt 8），ALFWorld 占 0%；ALFWorld 一行是从未参与训练的框架。

训练曲线：loss 2.04 → 0.014，eval_loss 单调 0.083 → 0.024，无过拟合。SFT-3B 对 7B 裁判的配对 McNemar：三项全对 438 : 11 不一致样本，p=4.7e-114。

解读：
- 归因能力是真学到的（未微调 3B 只有 4.9%）。
- 模型规模本身买不到"错误类型"：3B → 7B 未微调，mode 16.9% → 16.0%。17 类分类是监督学出来的，不是 prompt 里读出来的。
- 迁移时"哪一步"保留一半（53%），"错误类型"接近 17 类的猜测下限（0.26）——分类是真实的迁移缺口。SFT-3B 在 ALFWorld 上（20.3%）高于未微调 3B（2.9%）和未微调 7B（12.5%）：3B 学到的归因能力迁移后仍胜过一个大一倍的零样本裁判。

### 1.4 GRPO 结果

设置：trl 1.12 `GRPOTrainer`，每个 prompt 采样 8 个回答，三个独立 0/1 奖励 = 官方 scorer 的 agent / step / mode 是否正确，β=0.04，与评测共用同一个解析器。

| arm | 起点 / 数据 | 结果 |
|---|---|---|
| 1 | SFT 起点，全部 prompt | **无梯度**：85–95% 的组 8 个采样奖励完全相同（SFT 已记住训练集），GRPO 优势为零 |
| 2 | 基座起点，全部 prompt | 域内 step 0.40 → 0.58，mode 不动；ALFWorld 上与未微调基座统计无差别 |
| 3 | SFT 起点，714 条"难例"（SFT 在其上奖励方差 > 0） | 有信号，但留出集 mode **0.828 → 0.811**（0–2 条修好，11–12 条弄坏，p≤0.013）。原因是采样偏差：难例中 mathchat-R.3 占 43%，RL 学到"mathchat → R.3"捷径；~115 步提前停止 |

结论：在这个规模（~1,000 prompt、200 步、LoRA r=16），稀疏 0/1 可验证奖励**教不会** 17 类错误分类，反而会侵蚀它；SFT 的密集监督才行。GRPO 方向关闭（未试的修复：按 mode 平衡采样、mode 加权奖励、更大规模）。

### 1.4b OPD 结果（第三种训练范式）

on-policy 蒸馏：学生 = 3B 归因器（仅 agent 文字 SFT 初始化）+ 新 LoRA，教师 = 7B 归因器（同格式，冻结）；学生在 5,018 条域内训练 prompt 上自己采样三行答案，逐 token 全词表 KL(教师‖学生)，不用金标签，半个 epoch，两个 arm（token 等权 / Error Mode 行 ×3）。

| 3B 归因器 | 域内三项全对 | ALFWorld step / 三项全对 |
|---|---|---|
| 初始化（SFT） | 82.0 | 62.7 / 23.2 |
| + OPD 等权 | 80.9（p=0.31） | 64.3 / 22.5（n.s.） |
| + OPD mode 加权 | 80.4（p=0.06） | 62.1 / 22.2（n.s.） |
| 教师 7B | 83.8 | 91.3 / 38.9 |

结论：OPD 没有把 7B 的域外能力传给 3B——两端都与初始化无差别。原因在训练日志里：训练 prompt 上学生与教师的逐 token KL 只有约 0.01 nats，两者在训练集上本就一致，on-policy 采样暴露不出可蒸馏的分歧。三种范式合起来：SFT 学到了归因；GRPO（稀疏 0/1 奖励）不提升且可能侵蚀 mode；OPD（同格式更强教师、域内 prompt）是空操作。真正改善 Model A 的是数据（仅 agent 文字）和规模（7B）。

### 1.5 输入消融与压缩重训（老师的问题："真的需要完整轨迹吗？"）

先做推理时消融（Model A 不变，输入被删改），再对三种压缩各重训一个 Model A（干净答案）：

| Model A 的训练输入 | 保留字符 | 域内三项全对 | ALFWorld step | ALFWorld 三项全对 |
|---|---|---|---|---|
| 完整轨迹（参照） | 100% | 81.6 | 53.4 | 20.3 |
| **去掉环境 / 工具返回** | 75% | 81.5（p=1） | **64.3**（p=4.5e-5） | 22.8（p=0.17） |
| 同上，限制为与参照**完全相同的 4,767 条**训练样本 | 75% | 82.0 | **62.7**（p=3.4e-4） | 23.2 |
| 只保留含数字/答案的行 | 65% | 72.5（p=1e-8） | 37.3 | 13.2 |
| 每步只保留第一行 | 10% | 49.2 | 12.5（82% 的题都答"第 1 步"） | 5.1 |

**"去掉环境 / 工具返回"具体是什么。** 官方渲染器把每一步里**不是 agent 自己生成**的文本包在 `[observation] … [/observation]` 标签里：ALFWorld 的环境反馈、smolagents 的代码执行日志。`no_observations` 变体只删这些标签块，agent 的思考（`[think]`）、动作（`[action]`）、输出（`[output]`）一字不动。同一条 ALFWorld 轨迹（`alfworld_alfagent_0001`，13,742 → 11,564 字符）的前两步：

```
完整轨迹                                              去掉环境反馈后
------------------------------------------------      ------------------------------------------------
Step 1 | Agent: agent                                 Step 1 | Agent: agent
[think]                                               [think]
I need to find two pencils to put in a drawer.        I need to find two pencils to put in a drawer.
Pencils are commonly found on desks or shelves ...    Pencils are commonly found on desks or shelves ...
[/think]                                              [/think]
[action]                                              [action]
go to desk 1                                          go to desk 1
[/action]                                             [/action]
[observation]                                         （删除）
You arrive at desk 1. On the desk 1, you see a
book 1, a pencil 2, and a pencil 1.
[/observation]
Step 2 | Agent: agent                                 Step 2 | Agent: agent
[think]                                               [think]
I found two pencils on desk 1 as required ...         I found two pencils on desk 1 as required ...
[/think]                                              [/think]
[action]                                              [action]
take pencil 1 from desk 1                             take pencil 1 from desk 1
[/action]                                             [/action]
[observation]                                         （删除）
You pick up the pencil 1 from the desk 1.
[/observation]
```

smolagents 同理：`[output]` 里 agent 写的 Thought + Code 保留，`[observation]` 里的 `Execution logs: … [5 rows x 46 columns] Index([...])` 删除（`databench_smolagents_0480`，10,445 → 9,032 字符）。

哪些框架受影响：只有 **alfagent（ALFWorld，311/311，保留 82% 字符）和 smolagents（val 327/327，保留 83%）** 有 `[observation]` 块；debate / dylan / mathchat / magentic-one / macnet / metagpt 的渲染器没有这个标签（mathchat 里 `user_proxy` 回传的代码结果是它作为 agent 的 `[output]`，不算 observation），这些框架的 prompt 在两个变体下完全相同。所以"去掉环境反馈"实际改变的是：**训练时** smolagents（训练集里最大的框架，约 55%）的样本变短、不含执行日志；**测试时** ALFWorld 的样本不含环境反馈。两者都是"单 agent + 环境/工具"型轨迹，这也解释了为什么收益出现在 ALFWorld 迁移上：模型在 smolagents 上学会了"只读 agent 的思考和动作来定位错误"，这个习惯直接迁移到了同类型的 ALFWorld。

**"限制为与参照完全相同的 4,767 条训练样本"是什么意思。** 训练时超过 8,192 token 的样本会被丢弃（不截断）。完整轨迹版本的训练集 5,353 条里有 586 条超长被丢，实际训练 **4,767** 条；去掉环境反馈后 prompt 短了约四分之一，只有 337 条超长，实际训练 **5,016** 条。这就带来一个混淆：ALFWorld 上的提升可能不是因为"没有环境反馈"，而是因为"多了 249 条训练样本"。为排除它，第三行用 `--keep_ids` 把去掉环境反馈的训练集**限制为完整轨迹版本实际训练的那 4,767 个样本 id**（一条不多、一条不少，只是每条更短），其余设置不变重训。结果和 5,016 条的版本统计无差别（step 18:23，p=0.53），对参照仍显著（46:17，p=3.4e-4），所以提升确实来自输入格式。

解读：
- 归因器需要的是 **agent 自己写的文字**。环境 / 工具返回在域内没有信息量，在域外是噪声：去掉它，域内不掉，迁移定位 +9～11 个点（数据量混淆已由第三行排除）。**这是目前唯一能缩小迁移差距的手段**，归因器的标准输入应改为"仅 agent 文字"。
- 任何删 agent 文字的压缩都单调掉点，"错误类型"最先受损：错误的性质写在推理正文里，第一行或含数字的行里看不出来。
- 附带发现：决定性错误的位置随任务类型变——辩论 / 数学聊天类 76% 在轨迹后半段，ALFWorld 82% 在前半段；只给前半段时，step 准确率几乎等于"金标步骤是否可见"的比例。因此**在线归因**（任务进行中就发现错误）在具身任务上可行，在辩论类任务上结构性不可行，这不是模型问题。

---

### 1.6 规模测试：7B 归因器（2×2：模型大小 × 输入格式，训练样本完全相同的 4,767 条）

| Model A | 域内三项全对 | ALFWorld "哪一步" | ALFWorld 错误类型 | ALFWorld 三项全对 |
|---|---|---|---|---|
| 3B，完整轨迹 | 81.6 | 53.4 | 26.0 | 20.3 |
| 3B，去环境反馈 | 82.0 | 62.7 | 26.7 | 23.2 |
| 7B，完整轨迹 | 81.8 | 86.5 | 34.1 | 32.8 |
| **7B，去环境反馈** | **83.8** | **91.3** | **39.9** | **38.9** |

配对检验（ALFWorld，n=311）：规模（7B vs 3B，同为完整轨迹）三项全对 +12.5 个点，p=4×10⁻⁸；格式（7B 去环境反馈 vs 7B 完整）+6.1 个点，p=9×10⁻⁴，比 3B 上的 +2.9（p=0.15）更大，而且在 7B 上"错误类型"也显著提升（23:5）。两个效应叠加：20.3 → 23.2 → 32.8 → 38.9。域内四个模型都在 82–84%，规模与格式都不重要。

解读：归因任务在域内 3B 就饱和了；域外则规模和"只看 agent 文字"都重要且互相叠加——去掉环境反馈不是小模型的拐杖，模型越强收益越大。后续标准 Model A：**7B、仅 agent 文字输入**（在从未见过的具身任务上定位 91% 的决定性错误，错误类型 40%，首次离开 0.26 的猜测下限）。

---

## 2. 用于纠错实验的多智能体系统（MAS）

所有系统都由 **Qwen2.5-7B-Instruct（冻结）** 扮演全部角色，在同一批 **593 道题**上做题（math 192 / scibench 225 / mmlu_pro 117 / gpqa 59，题目取自 Who&When Pro 的任务池）。每个系统输出严格遵守 Who&When Pro 对应框架的官方 schema，因此归因器、渲染器、评分器全部复用。三种协作拓扑：

| 框架 | 拓扑 | 流程 | 代码 |
|---|---|---|---|
| **debate** | 平级 | 3 个 agent。第 0 轮各自独立作答；第 1 轮每个 agent 看到自己和同伴上一轮的答案后修订；最后多数投票 | `src/error_attr/selfplay/debate_runner.py` |
| **mathchat** | 推理者 + 工具代理（AutoGen MathChat 复刻） | `assistant` 说思路并写 Python；`user_proxy` 真实执行代码并回传 `(Result of the code run: ...)`，无代码时说"Continue…"；`assistant` 输出 `\boxed{}` 即终止；≤5 轮 | `mathchat_runner.py`，沙盒 `pyexec.py` |
| **magentic-one 式** | 层级 | `orchestrator` 每轮写一条指令并指派一名专家：`coder` 写代码 / `computerterminal` 执行最近一段代码；只有协调者能宣布最终答案；≤6 轮，超时强制作答 | `magentic_runner.py` |

没有联网 / 文件工具，所以只用自含推理类 benchmark。

**debate 的两个系统提示（原文）**

```
第 0 轮：You are one of several independent expert agents solving a problem. Think step by
step, then give your final answer in exactly the format the task specifies.

第 1 轮：You are one of several expert agents in a debate. You previously answered the problem;
you now see the other agents' answers from the last round. Reconsider carefully -- keep your
answer if you are confident, change it only if another agent's reasoning is genuinely more
convincing. Think step by step, then give your final answer in exactly the format the task specifies.
```

**一条 mathchat 轨迹（`selfplay_mathchat_math_0147`，金标 351，系统答 360）**

```
[user_proxy r0p0]  （AutoGen MathChat 的固定开场：要求用 Python、print 输出、三种解法路线、把答案放进 \boxed{}，然后附题目）
[assistant  r0p1]  "To determine the number of cake-eating days from Amaretta's birthday (July 27) to
                    Enzo's birthday (September 3) each year from 2008 to 2016, we need to calculate ...
                    ```python ... ```"
[user_proxy r1p0]  "(Result of the code run:\n360\n)"
[assistant  r1p1]  "The total number of cake-eating days ... is \(\boxed{360}\)."
```

**一条 magentic-one 轨迹（`selfplay_magentic_scibench_0554`，金标 0，系统答 1468.8）**

```
[orchestrator r0p0 -> coder]  "Write the formula for the rate of flow outward through a surface using
                               the divergence theorem. Then express the divergence of the velocity field ..."
[coder        r0p1]           "```python\nfrom sympy import symbols, div, integrate\n... div_v = div(v, (x, y, z)) ..."
[orchestrator r1p0 -> computerterminal]  "Evaluate the integral of the divergence over the volume of the cylinder ..."
[computerterminal r1p1]       "Execution failed. Output:\nError: ... AttributeError: 'Tuple' object has no attribute 'expand'"
[orchestrator r2p0 -> coder]  "Correctly calculate the divergence of the velocity field and then integrate it ..."
[coder        r2p1]           "```python\n# Correct divergence of the velocity field\ndiv_v_correct = z + 2*y**2 + 2*x**2 ..."
[orchestrator r3p0 -> FINAL]  "FINAL: \boxed{1468.8}"
```
（真实散度是 2y，对称圆柱上的通量为 0；编码员第二次直接"写下"了一个错误的散度表达式。）

---

## 3. MAS 的初始失败数量 / 比例

判分规则：解析 `\boxed{}`（花括号配对），字符串归一化后一致 **或** 两边都是数值且相对误差 ≤1%（SciBench 官方用 5%）。

| benchmark（题数） | debate 正确 | mathchat 正确 | magentic-one 正确 |
|---|---|---|---|
| math（192） | 90（47%） | 93（48%） | 52（27%） |
| scibench（225） | 84（37%） | 92（41%） | 57（25%） |
| mmlu_pro（117） | 74（63%） | 67（57%） | 45（38%） |
| gpqa（59） | 16（27%） | 13（22%） | 12（20%） |
| **合计 / 593** | **264（45%）** | **265（45%）** | **166（28%）** |
| **可用失败**（给出了答案且错） | 329 | 326 | 415 |

- debate 与 mathchat 水平相当；层级式 magentic 明显更差，典型模式是"编码员忘了 print → 终端返回空 → 协调者重复同一指令数轮 → 超时被迫作答"（Who&When 分类里的 A.4 循环 / C.1 编排错误）。
- 题目本身偏难：任务池取自 Who&When Pro，即别的系统曾失败过的题。
- 每个框架用固定随机种子抽 **150 条失败**进入纠错实验。debate 的 150 条中有 32 条在修正判分后其实是对的（原判分把 `87.79` vs `+87.8`、`0.5` vs `\frac{1}{2}` 判为错），因此 debate 的分析限于 **118 条真失败**；mathchat / magentic 的样本在判分修正后再抽取，n=150。

---

## 4. 归因器对 MAS 失败轨迹的回应

流程：把系统自己生成的失败轨迹渲染成与训练时完全相同的 `all_at_once` prompt（第 1.2 节的格式，只是轨迹换成 Qwen 团队的），Model A 贪心解码三行；150/150 全部可解析。

### 4.1 一个完整的例子（debate，`selfplay_debate_mmlu_pro_0311`）

**任务**："What tool can be used to perform SNMP enumeration? A. Whois … G. IP Network Browser H. Nmap … J. Tcpdump"。金标 **G**，团队答 **H**（三个 agent 第 0 轮都分析到 Nmap，第 1 轮互相确认后全部维持 H）。

**归因器 prompt 的轨迹部分（节选，共 14,920 字符）**

```
## Transcript

Step 0.0 | Agent: agent_1
[output]
To determine which tool can be used for SNMP (Simple Network Management Protocol) enumeration,
let's analyze each option:

A. Whois - Used for querying information about internet domain names and IP addresses. Not for SNMP.
B. Nslookup - Used for determining network configuration and hostnames. Not for SNMP.
C. Wireshark - A network protocol analyzer ... not primarily an enumeration tool.
...
G. IP Network Browser - A utility for viewing and managing IP networks. Ca...
[/output]
Step 0.1 | Agent: agent_2
[output] ... [/output]
Step 0.2 | Agent: agent_3
[output] ... [/output]
Step 1.0 | Agent: agent_1
[output]
\boxed{H}
[/output]
Step 1.1 | Agent: agent_2
[output]
The task requires choosing the correct tool ... After considering the options and the
explanations provided by the other agents, the best choice remains Nmap ...
[/output]
Step 1.2 | Agent: agent_3
[output] ... my conclusion remains unchanged. The most appropriate tool ... is Nmap ... [/output]
```

**Model A 的输出（原文）**

```
Agent Name: agent_1
Step Number: 0.0
Error Mode: R.1
```

即：第一个决定性错误在 agent_1 第 0 轮的发言（对 IP Network Browser 的功能描述是编造的——R.1 幻觉），后面两个 agent 是跟随。

### 4.2 三个框架上的诊断分布（各 150 条）

| | debate | mathchat | magentic-one |
|---|---|---|---|
| **谁** | 三个 agent 分散 | `assistant` 150/150 | `coder` 80 / `orchestrator` 68 / `terminal` 2 |
| **哪一步** | 集中在第 1 轮（修订轮） | 第 1 步（首次作答）84% | 第 0 轮（首个计划或首段代码）98/150 |
| **错误类型前三** | C.3 过度依赖其他 agent 49% · R.2 推理错误 33% · R.4 任务理解错误 16% | R.4 任务理解错误 57% · R.3 计算错误 24% · R.2 推理错误 13% | R.1 幻觉 46% · R.4 23% · R.2 13%（另有 C.1 编排 / C.2 沟通 12 条） |

上面两条轨迹的诊断：mathchat 例 → `assistant / 1 / R.4`（首次作答的思路错误，代码按错误思路算出 360）；magentic 例 → `coder / 0.1 / R.1`（编码员的散度公式没有根据）。

解读：归因器读懂了协作结构——平级辩论的主要病是"跟风"，工具代理式是纯推理 / 计算错误，层级式是幻觉加编排问题。此前在 232 条有金标可比的轨迹上，用 Model A 的预测标签和用金标签做反馈，效果无差异（每条件仅 2–4 条翻转，p≥0.25），所以它的输出可以直接当反馈内容用。

---

## 5. 归因器的回应如何交给 MAS

### 5.1 从三行输出生成反馈文本

模板固定，错误类型展开为"分类名 + 定义的第一句"。以第 4.1 节的例子为例，五种信息量的反馈原文：

| 条件 | 反馈文本 |
|---|---|
| **incorrect_only**（无信息对照） | Your team's answer to this task was incorrect. Please review your team's work and provide a corrected final answer. |
| **reason_only** | Your team's answer to this task was incorrect. Somewhere in your team's work, the following error occurred: Hallucination -- Generating claims that are NOT grounded in any observation the agent has retrieved. Please review your team's reasoning and provide a corrected final answer. |
| **vague_reason** | Your team's answer to this task was incorrect. The error occurred somewhere in the first quarter of the interaction (roughly steps 1-2 of 6). The error was of this kind: Hallucination -- … Please review that part of your team's work and provide a corrected final answer. |
| **precise_reason** | Your team's answer to this task was incorrect. Specifically, at step 0.0, in agent_1's turn, the following error occurred: Hallucination -- … Please review that step and provide a corrected final answer. |
| **precise_only** | Your team's answer to this task was incorrect. Specifically, at step 0.0, in agent_1's turn, an error occurred. Please review that step and provide a corrected final answer. |

另有三个**故意错误**的对照，用同样模板生成：`wrong_location`（另一个 agent 在另一轮的真实发言位置，内部自洽但错）、`placebo_reason`（随机换成另一种错误类型）、`random_attribution`（agent / step / type 全随机）。

### 5.2 注入方式

**one_shot**：把"# Task + ## Your Team's Attempt (this attempt was INCORRECT) + 完整轨迹 + ## Coordinator Feedback + 反馈文本 + 只输出修正后的答案"交给同一个 Qwen-7B，一次生成。

**框架原生的"再来一轮"（主结果）**：反馈作为系统的下一轮输入进入原系统。

- debate：每个 agent 收到（系统提示同第 1 轮）：

```
Problem:
{题目}

Your previous answer:
{该 agent 上一轮的输出}

Other agents' answers (last round):
[agent_2]: {…}
[agent_3]: {…}

## Coordinator Feedback
{反馈文本}

Give your (possibly revised) final answer in exactly the format the task specifies.
```
  三个 agent 各答一次后多数投票。

- mathchat：`user_proxy` 在同一对话中追加一条消息 `## Coordinator Feedback\n{反馈文本}\n\nContinue solving. Put your corrected final answer in \boxed{}.`，`assistant` 最多再答两轮（可再执行一次代码）。
- magentic-one：`orchestrator` 收到完整团队记录 + `## Coordinator Feedback\n{反馈文本}\n\nYour team's previous final answer was judged INCORRECT. Decide how to proceed.`，可再指派 1–2 轮，然后必须给出 `NEXT_SPEAKER: FINAL`。

### 5.3 结果：改对率（%）

| 条件 | debate（n=118 真失败） | mathchat（n=150） | magentic-one（n=150） | 旁观者*（n=2,328） |
|---|---|---|---|---|
| incorrect_only | 5.1 | 7.3 | 3.3 | — |
| reason_only | 5.1 | 4.7 | 5.3 | 43.6 |
| vague_reason | 6.8 | 5.3 | 6.0 | 41.8 |
| precise_reason | 4.2 | 7.3 | 4.0 | 43.5 |
| precise_only | 5.9 | 6.7 | 4.7 | 40.5 |
| wrong_location（错） | 8.5 | 6.0 | 3.3 | — |
| placebo_reason（错） | 5.1 | 3.3 | 7.3 | — |
| random_attribution（错） | 6.8 | 4.0 | 4.7 | — |

\* 旁观者 = 一个**新的** Qwen-7B 读 Who&When Pro 中**别的系统**的失败轨迹 + 同样的反馈后重做题（mathchat / magentic-one / debate / dylan 四个可判分框架）。

**补充的三个基线（2026-09-12，同 2,328 条、同读者）**：只给题目 14.3%；给轨迹但不说错了 28.5%；给轨迹 + “答案是错的” **42.4%**。在此之上：只给原因 43.6%（+1.2，p=0.017）、精确位置 + 原因 43.5%（+1.1，p=0.04）、模糊位置 + 原因 41.8%（n.s.）、**只给精确位置 40.5%（−1.9，p=3×10⁻⁵）**。原先“原因 ≫ 位置”的 3.1 个点差距，三分之二来自“只给位置反而有害”，三分之一来自“原因略有帮助”。对旁观者真正起作用的是“知道答案错了”（+14 点），诊断类型只值 1 个点，裸的位置指针是干扰。这与自纠正实验的结论统一起来：读者和团队都响应结果信号，而非诊断内容。

统计：
- 对犯错的团队自身：三个框架里没有任何正确内容条件优于"只说你错了"（所有配对 McNemar p≥0.34），错误内容也不比正确内容差；one_shot 与"再来一轮"无差别（p≥0.18）。团队约 60–65% 原样保留旧答案；改了之后也只有 10–15% 改对，低于多选题随机换项的 33%。
- 对旁观者：反馈有用（40–44% 修好），且 **"原因" > "位置"** 显著（precise_only 对每个含原因的条件 p≤4e-3；有原因后再加精确位置无增益 p=0.92）。

**第 4.1 节那条轨迹的实际纠正记录**（8 个条件，同一条轨迹）：

| 条件 | 团队新答案 | 改对？ |
|---|---|---|
| incorrect_only | H | 否（未改） |
| reason_only | H | 否（未改） |
| vague_reason | G | **是** |
| precise_reason | H | 否（未改） |
| precise_only | G | **是** |
| expert_incorrect_only | G | **是** |
| expert_precise_reason | G | **是** |
| deferential_system_precise_reason | H | 否 |

这条轨迹说明了为什么整体是 null：同样的诊断内容（R.1 / agent_1 / 0.0），有位置无原因时改对，有位置有原因时反而没改——团队对反馈的反应接近随机，和内容无关。

解读（含基线修正）：**对外部读者，起作用的是结果标签（+14 点），错误类型只加 1 个点，裸位置反而 −2 点；对当事团队，什么都不起作用**。两边一致：起作用的是“你错了”这个信号，不是诊断内容。这否定了研究计划中"用归因器输出作为反馈内容驱动团队自我修正"的路径（在 7B 规模上），但不否定归因器本身：它准、它迁移、旁观者能用它。

---

## 6. 语气与角色如何改变 MAS 的行为

### 6.1 操作

内容固定，只改呈现（第一句 / 系统提示）：

| 条件 | 改动 |
|---|---|
| **expert_incorrect_only** | 首句改为 "An independent expert reviewer has verified that your team's final answer is WRONG and has audited the full transcript. This diagnosis is reliable and should be acted on." 之后**不给任何诊断** |
| **expert_precise_reason** | 同样的权威首句 + 完整 precise_reason 诊断 |
| **deferential_system_precise_reason**（仅 debate） | 系统提示由 "keep your answer if you are confident, change it only if another agent's reasoning is genuinely more convincing" 改为 "Your team's previous answer has been verified as incorrect by an independent reviewer … Do NOT simply keep your previous answer: re-derive the solution from scratch, taking the reviewer's diagnosis into account" |

在 debate 上先测 150 条，再在 150 条之外**全部** 173 条新的真失败上复验，然后在 mathchat / magentic 各自的 150 条上做跨框架复验。

### 6.2 结果（配对 McNemar，"A 独对 : B 独对"）

| 比较（同一批轨迹，配对） | debate（n=291） | mathchat（n=326） | magentic（n=415） | **合并 1,032** |
|---|---|---|---|---|
| 权威口吻、不给内容 vs 中性"你错了" | 10.7 vs 6.5%，15:4，p=0.012 | 10.1 vs 5.5%，21:6，p=0.006 | 8.0 vs 5.1%，24:12，p=0.065 | **60:22，p=3e-5** |
| 权威 + 诊断 vs 中性 + 诊断 | 10.0 vs 6.2%，13:3，p=0.013 | 8.6 vs 5.5%，18:8，p=0.08 | 7.2 vs 6.7%，16:14，p=0.86 | **47:25，p=0.013** |
| 权威 + 诊断 vs 权威不给诊断（内容在权威之上有增益吗） | 7:9 | 12:17 | 11:14 | 30:40，p=0.28（方向为负） |
| 中性 + 诊断 vs 中性不给（内容本身有增益吗） | 8:9 | 12:12 | 21:14 | 41:35，p=0.57 |
| 系统提示"从头重推"（仅 debate，118） | 5.9%，≈ 中性；改答案率 49%，改对率仅 12% | — | — | — |

（mathchat / magentic 的样本先在 150 条上测，再补跑各自剩余的全部真失败 176 / 265 条；表中为合并值。）

**行为层面**：权威口吻使团队改答案的比例从约 37% 升到约 50%，但"改了之后是否正确"不变（12–19%）；"从头重推"改答案率最高（49%）而改对率最低（3–12%）。收益集中在多选题（gpqa / mmlu_pro），数学题几乎为零。

### 6.3 解读

- 这是整套自纠正实验里**唯一**站得住的效应：三种拓扑方向一致（各 +3～5 个点），两个框架单独显著，合并 1,032 条配对样本 p=3e-5；复验都是在看到结果前就在不相交样本上预先设定的。
- 它的机制是**社会性的**而非**信息性的**：说话者的权威降低了团队对旧答案的执着，而诊断内容加在权威口吻上不增益（16:23，若有也是略负）。
- 对 7B 团队，"谁错、哪步错、什么错"不是杠杆；"被多权威地告知你错了"才是——而这本身也只让它多猜几次，猜对率不变。

---

## 总结与下一步

1. **归因器**：SFT 学到了真实、可迁移的归因能力（域内 81.6%，迁移定位 53%）；GRPO 在这个规模上无效甚至有害；标准输入应改为"仅 agent 文字"（域内持平，迁移定位 +9～11 点，混淆已排除）。
2. **自纠正**：三种协作结构的 7B 团队对反馈**内容**完全不敏感，错的和对的一样；唯一有效的是**权威口吻**，机制是提高放弃旧答案的意愿。
3. **归因器的定位**：作为外部裁判和训练信号（RQ1–3）——它准、它迁移、旁观者能用它的"原因"；不应作为反馈内容喂给团队。
4. **下一步建议**：自纠正若继续，应操纵**接收者**而非消息——把同样的反馈交给新实例或更强的模型、隐藏团队的旧答案强迫重推。这能分清"信息没用"和"这个团队用不了"。

## 附：过程中修正的两个判分问题

- `\boxed{}` 正则遇嵌套花括号截断（`\boxed{\frac{1}{64}}` → `\frac{1`），改为花括号配对解析。
- 判分为字符串精确匹配（`87.79` vs `+87.8` 判错），改为"一致或数值误差 ≤1%"。

两者改变了很多绝对数字（旁观者实验每条件 +11 个点；debate 题库 862 条中 86 条"失败"其实是对的），但**没有改变任何结论的方向或显著性**——所有比较都是配对设计，同一条轨迹在各条件下受同样的判分影响。相关重跑与拼接见 DECISIONS #25/#26。
