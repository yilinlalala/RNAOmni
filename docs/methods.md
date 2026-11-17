# 3. Methods

> 时间戳：`2026-09-01`
> 类型：`METHOD`
> 方法名：**Tri-Axis Model Generation（三轴模型生成）**

---

## 执行过程（按运行逻辑）

以回归任务输入 `dataset/HEK_te_prediction_task.csv` 为例（列：`input_seq`（RNA 序列）+ `predicted_target`（连续标签，回归）+ `length` + 已预划分的 `fold`）。RNAOmni 的一次自主科研据此数据走如下处理环节，由 **AI 科研智能体**在**无推理调度器**与**版本控制（git）状态机**的协同下沿 **数据 / 特征 / 模型三轴**循环推进：

```
① 数据读取与判定
   读 CSV → 识别任务类型（连续标签 → 回归，决策指标 val_spearman）
   → 按 pre 给定 fold（0–9）建立 train/val/test 划分

② 数据处理管线（prepare，人锁定的评估合同，Agent 禁改）
   清洗：T→U · 去非法碱基/非有限标签 · 去重复序列
   编码：input_seq 分段打包 → 特征编码（分块 / 去重 gather / store_len，控内存）
   全量 →（编码超时）→ 聚类感知抽样；回归采样 = 行上限 1000（train 翻倍）

③ 特征生成（特征轴，白名单配方，Agent 可改）
   按 features/active.yaml 生成 L0–L4 表示（one-hot / k-mer / 结构 / L4 预训练大模型）
   经融合（concat/add/mean）→ 得到 input_dim → 供模型输入

④ 模型生成 + 训练（模型轴 + 数据轴，Agent 可改 train.py）
   baseline/预测头骨架（12 菜单）读 input_dim、num_outputs=1（回归）
   → 固定时间预算 TIME_BUDGET 内训练
   → 评估：均值 val_spearman（回归）

⑤ 每轮迭代（核心循环）
   A. [Agent] 读状态（git / 结果表 / 当前配方）
   B. [Agent] 声明 round_type（feature|model）+ 单轮一个主假设
   C. [Agent] 白名单内落地改动（train.py / active.yaml）→ git commit
   D. [调度器] 固定时间训练 + 评估
   E. [调度器] 判定：keep（≥ min_delta 提升）| discard（git reset 回滚）| crash（短签名）
   F. [调度器] 收敛控制：数据轴卡 → 扩采样；模型轴卡 → 平台期切骨架；全局 patience → 停搜归档

⑥ 终止与最终评估
   三轴（最优数据规模 + 最优配方 + 最优骨架）汇合 → 用全量 fold 的 ensemble 评估 test
   （test_RMSE / test_R2 / test_spearman / test_pearson）→ 归档
```

---

## 裸数据处理管线（仅数据处理部分）

当输入是**裸数据**（无预划分 `fold`，仅含 `input_seq` + `predicted_target`）时，仅数据处理环节如下（均为 `prepare` 锁定、Agent 禁改的评估合同）：

```
① 读入 CSV
   → 限定 input_seq + predicted_target 两列

② 任务类型判定（infer_task_type）
   判断标签是离散（分类）还是连续（回归）
   → 决定后续划分方式与决策指标

③ 清洗（clean_dataset）
   1) T→U 碱基归一化
   2) 去除非法碱基的行（非法字符）
   3) 去除标签非有限值（NaN/inf）的行
   4) 按序列去重：相同序列折叠
      · 回归 → 标签取均值
      · 分类 → 保留首个标签

④ 划分 fold / split（ensure_splits，任务相关分支）
   · 回归路线：随机 test holdout（fold=-1）+ 分层 CV（fold 0–N-1）
   · 分类路线：类内 MMseqs 聚类 → 聚类感知 70/20/10 train/val/test
              （保证相似序列不跨集，防泄漏）

⑤ 生成 data card（build_data_card + write_data_card）
   记录：任务类型、n_rows、序列长度分布、
        标签统计（min/max/mean/std 或各类计数）、过滤统计

⑥ 特征编码（encode，构建输入张量）
   input_seq 分词/tokenization（按字母表）
   → 可选特征栈：L0 one-hot / L2 k-mer / L3 结构 / L4 预训练大模型
   → 融合成 input_dim
   工程优化：分块编码 / unique 去重 gather / store_len（长序列控内存）
   全量编码超时 → 聚类感知抽样（分类每类保底 K）或回归行上限 1000

⑦ 打包成 DataLoader（dataloader）
   fold 训练集 → mini-batch
```

**关键点**：
- 第 ② 步任务判定是总开关，把数据流一分二（回归/分类）。
- 第 ③ 步清洗**任务无关**，对所有数据一致。
- 第 ④ 步划分**任务相关**——回归用随机 holdout + CV，分类用 MMseqs 聚类感知划分。
- 第 ⑥ 步编码**可配置**（取决于 `active.yaml` 配方），并有「全量 → 超时抽样」自适应。

---

## 已注册特征与骨架（搜索空间）

### 特征 Skill（`skill-library/registry.yaml`）

| Skill id | 路径 | overnight | 角色 |
|----------|------|-----------|------|
| `rnaomni-feature` | `skill-library/feature/SKILL.md` | ✓ | 人控特征配方 + `train.py`，唯一 overnigh 注入 |

配套文档：`agent-ops.md`（夜间一轮）、`frameworks.md`（预测头骨架）、`registry-spec.md`（recipe 白名单）、`structure-encoding.md`（L3 结构）、`deploy-ops.md`/`human-ops.md`（人上架）。

### 已注册特征（`features/registry.yaml`，分层 L0–L4）

| 特征 | 层 | 维度 | 成本 | Agent 可调 |
|------|-----|------|------|------------|
| `onehot` | L0 | 4 | 低 | — |
| `length_frac` | L1 | 1 | 低 | — |
| `kmer` | L2 | 4^k | 低 | `k` 1–4 |
| `struct_thermo` | L3 | 3 | 中 | — |
| `struct_dotbracket` | L3 | 3 | 中 | — |
| `struct_bpp` | L3 | 2 | 中 | — |
| `rna_fm` | L4 | 640 | 高 | `repr_layer` 0–12 |
| `rinalmo` | L4 | 480 | 高 | — |
| `evo` | L4 | 4096 | 高（重，12GB） | — |
| `evo2` | L4 | 4096 | 高（重，12GB） | — |

- **L0–L2** 轻量统计 / 穷举编码；**L3** 物理结构（ViennaRNA 热力学）；**L4** 预训练大模型表征（RNA-FM / RiNALMo / Evo / Evo2-7B，仅本地权重、禁下载）。
- 融合方式：`concat` / `add` / `mean`；Agent 只改 `active.yaml` 白名单键（enabled 开关、tunable 参数、fusion）。

### 模型骨架（`frameworks.md`，12 个命名序列级预测头）

| 类别 | 骨架 |
|------|------|
| **轻量池化**（6） | `pool_mlp`(mean+max)、`max_pool_mlp`、`mean_deep_mlp`、`attn_pool_mlp`、`cls_mlp`、`gated_pool_mlp` |
| **Transformer**（4） | `tf_mean`、`tf_cls`、`tf_attn`、`tf_conv` |
| **ResNet**（1） | `resnet1d_pool` |
| **循环**（1） | `gru_pool` |

- 统一走 **stem → block → masked pool → MLP** 张量路径；开局 `baseline`（`Predictor = PooledMLP`，非菜单名）。
- 平台期由调度器从**未试用**骨架中均匀随机抽换头；红线：不加 Mamba/GNN、不做 per-token/配对头、不硬编码 `input_dim`/`num_classes`。

### 三轴完整搜索空间映射

```
数据轴：full ↔ sample（编码超时切换）· 扩采样 ×2 · 行/类上限
特征轴：10 个已注册特征（L0–L4）· fusion · tunable 参数
模型轴：12 个骨架（4 类）+ baseline · 平台期切换
```

---

## 3.1 Tri-Axis Model Generation（三轴模型生成）

### 3.1.1 核心定义

我们把 RNA 建模的自主科研，抽象为**从「数据、特征、模型」三个正交轴上生成并组装出一个最优预测模型**的过程——

- **数据轴**决定「喂什么 / 喂多少」；
- **特征轴**决定「怎么表示」；
- **模型轴**决定「用什么骨架」；

三轴由**同一个 AI 科研智能体（Agent）协同驱动**，最终**汇合生成**一个完整的最优 RNA 预测模型。

### 3.1.2 三轴各自「生成」什么

| 轴 | 生成对象 | 策略 |
|----|----------|------|
| **数据轴生成** | 自适应训练规模 | 先粗后精、逐级放大的数据状态（先全量→编码超时抽样→收敛再放大） |
| **特征轴生成** | 分层表示配方 | L0–L4 分层堆栈（轻量统计 ↔ foundation model）+ 白名单约束 + 融合策略 |
| **模型轴生成** | 可枚举迭代的预测头骨架 | 12 头归一化菜单 + 平台期自动切换新架构 |

**数据轴（数据规模）**：以固定时间预算为平准器，配合「先全量、编码超时再抽样、收敛再放大」的自适应管线逐级逼近全量训练规模。

**特征轴（表示配方）**：在分层表示堆栈（L0 穷举 → L2 序列统计 → L3 热力学结构 → L4 预训练大模型表征）上，以白名单配方形式生成表示空间、融合方式（concat/add/mean）与关键参数（k-mer k、repr_layer）。

**模型轴（预测头骨架）**：在归一化预测头菜单上生成并迭代骨架，平台期（连续 discard 达到上限）自动切换新架构，首枪即使不优于全局 best 也先落地为基线。

### 3.1.3 驱动引擎（贯穿三轴）

```
LLM Agent 提假设 → 固定时间训练 → keep/discard（git reset 回滚）→ 调度器控收敛
```

- **LLM Agent**：负责结构化假设生成（改 `train.py` / 特征配方）——高层次架构创意的来源；
- **无推理调度器**：负责搜索收敛与探索-利用平衡（patience、强制切骨架、扩采样）——非 LLM 的策略层；
- **版本控制（git）作为搜索状态机**：每个假设 = 一次 commit；keep/discard 决定推进或 `git reset` 回退，保证**搜索历史可审计、可回放、可复现**；
- **严格评估合同**：`keep` 须相对当前 best 提升 ≥ `min_delta`，评测协议（`prepare.py`）由人锁定、Agent 永不触碰，杜绝自主科研的评估污染。

### 3.1.4 正式贡献声明

> **Tri-Axis Model Generation**：我们提出一种从三个正交轴上自主生成模型的方法。
>
> **数据生成轴**以固定时间预算与「先全量、编码超时抽样、收敛再放大」的自适应管线，逐级逼近全量训练规模；
> **特征生成轴**在分层表示堆栈（L0 穷举 → L2 统计 → L3 热力学结构 → L4 预训练大模型表征）上，以白名单配方形式生成表示空间与融合策略；
> **模型生成轴**在归一化预测头菜单上生成并迭代骨架，平台期自动切换新架构。
>
> 三者由 AI 科研智能体统一调度、最终汇合，在**极低算力（单 GPU、60s/轮）与不可篡改的人控评估合同**下，自主、可复现地生成最优 RNA 预测模型。

---

## 3.2 Thinking 模式（Agent 的思考与决策模式）

### 3.2.1 单假设驱动的迭代决策循环

RNAOmni 的 Agent 不进行"自由发挥"或长链思维，而是遵循**「一次一个结构化假设」**的克制式思考协议。每一轮，Agent ：

```
读状态
  → 声明 round_type（feature | model）+ 一个主假设
  → 白名单内落地（编辑代码/配方）
  → commit（git 记录）
  → [调度器负责] 固定时间训练 + 评估
  → keep / discard（决定推进或 git reset 回退）
```

**单假设原则**：每轮只验证**一个**主假设（One hypothesis），避免把无关改动捆进同一 commit 而无法归因效果。这是严谨科学对照的核心——保证"这一轮的指标变化只由这一个改动引起"。

**思考的组织**：思考被显式划分为两条**正交的探索轴线**，而不是一次"big-picture"模糊发散：

| 轴线 | 内容 | 过程标记 |
|------|------|----------|
| **特征轴** | 表示配方/融合/参数 | `[feature]` |
| **模型轴** | 预测头骨架/框架内旋钮（LR、宽度、dropout、EMA） | `[model]` + `cause=` |

### 3.2.2 过程分层的「对症下药」（过程提示搜索）

Agent 的下一枪不凭"常识"或直觉拍板，而是依据**搜索过程停在哪一层**来决定动哪一层：

```
任务/数据 X ──┬──→ 特征 F ──→ Y
              ├──→ 样本量 S ──→ Y
              ├──→ 训练旋钮 θ ──→ Y
              └──→ 头 H ──→ Y
```

| 过程提示（卡在哪） | 动作 |
|--------------------|------|
| 骨架错 | `do(H)`：调度器从菜单抽下一套 named 头 |
| 特征不够 | `do(F)`：已登记开关 / tunable / 融合 |
| 拧参错 | `do(θ)`：框架内明确旋钮 |
| 数据少 | `do(S)`：调度器扩采样（扩 K / 行数） |
| 实现坏了 | 修 crash（短签名摘要） |

**关键纪律**：一次 discard 只说明"这个改动没抬升指标"，**不能**就此写成生物学结论。只有**同一层上连续干预均无效**（瓶颈卡）才推动跨层决策（如强制切骨架）。`cause=` 是**过程层标记**（这次动的是 H / F / θ / S / crash），不是因果估计。

### 3.2.3 思考的制衡：搜索收敛控制

Agent 的思考并非无边界漫游，而是受**无推理调度器**的探索-利用平衡约束：

- **平台期切骨架**：当前框架连续一定次数 `[model]` discard → 从**尚未试用**的骨架里均匀随机抽一套，强制切换，防止 Agent 陷在单一架构共轭；
- **扩采样**：抽样集连续无显著提升 → 放大训练规模逼近全量，实现"由粗到精"；
- **全局停搜**：达到 patience 上限即归档停止，避免为微小增益无限消耗算力；
- **简洁性准则**：同等指标下更简者胜——克制"堆复杂度换取微增益"的过度思考。

### 3.2.4 防作弊的思考边界

Agent 的思考被限定在**人定义的合法空间**内，保证评估的纯净性：

- 只改 `train.py` 与 `features/active.yaml` 白名单键；
- **不得**修改 `prepare.py` / 评测协议 / 数据划分 / `registry.yaml` / 环境常量；
- 不得安装包或下载新预训练权重。

> 一句话：**RNAOmni 的 Thinking 模式 = 单假设、分轴、依过程对症下药、受调度器收敛控制、且严格限定在人控评估合同内的克制式自主科研思考。**

### 3.2.5 有无 Thinking 的对比（RNAOmni_dev vs RNAOmni_dev_thinking）

> `RNAOmni_dev`（无思考）是较早分支；`RNAOmni_dev_thinking`（有思考）在其上演进。两者**搜索空间相同**（特征 L0–L4、12 骨架、full↔sample 数据轴），差别只在**「换骨架的策略层」，即 Agent 的思考/决策机制**。

| 层面 | 无思考（RNAOmni_dev） | 有思考（RNAOmni_dev_thinking） |
|------|----------------------|--------------------------------|
| **换头触发** | 仅计数：`FRAMEWORK_PATIENCE=10` 次连续 `[model]` discard → 换下一个 | 计数触发 + **瓶颈卡（`next_do`）** 主动干预 |
| **抽法** | 纯均匀随机抽未试用骨架（无因果依据） | 有瓶颈逻辑：头被压制时**不抽**，改推 `[feature]` |
| **决策依据** | 不看过程，机械换头 | **过程因果提示**：按「同层连续失败次数」对症下药 |
| **`cause=` 记账** | 无 | 有：`cause=<head\|feature\|knob\|sample\|crash>` 作为过程层标记 |
| **瓶颈压制** | 无 | 有 `suppressed`：换头无效→压制 head；`[feature]`/样本 keep→清空压制 |
| **对 Agent 的强制** | 无 | 有：如 `next_do=feature` → 本轮**必须** `[feature]`（prompt 注入 MUST） |
| **实现状态** | **仅方案文档**（`docs/causal-guided-search-plan.md`），`search.py`/`prompts.py` 未实现 | **完整实现**（`rnaomni/orch/note.py` + `prompts.py` + `search.py`） |

**关键差异取证：**

- **框架选择**：无思考版 `ensure_next_framework`（`RNAOmni_dev/rnaomni/orch/search.py`）只做「`fw_streak >= patience` → `random.choice(剩余菜单)`」的机械随机；**没有** `next_do` / `suppressed` / `fail_*`。有思考版在同样随机抽基础上叠加瓶颈层（`format_cause_block`、`next_do_token`、`suppressed/head` 压制、`fail_head/feature/knob/sample` 计数），并在 prompt 注入「MUST」指令。
- **迭代决策**：无思考版 Agent 自由声明 `[feature]`/`[model]`，只靠 discard 计数换头；有思考版每轮带 `cause=` 记账，调度器据**过程证据**（哪一层连续无效）给出 `next_do`，**有时强制**下一轮动哪一层——即 `process-hint.md` 的 `π(动哪一层 | 过程证据)`。
- **full↔sample**：两者皆有，但有思考版更精细——区分「首次 encode 超时（重置 best 切 sample）」vs「已有 full baseline 后超时（**保持 full**，只记 crash）」，保护已建立的 full 基线。

> **对比小结：**
> **无思考版** = 靠「连续失败计数 + 均匀随机抽头」的**机械探索**，causal-guided-search 只停留在方案文档。
> **有思考版** = 新增**过程因果提示层（Think/瓶颈卡）**，用 `cause=` 记账 + `next_do`/`suppressed` 压制，让调度器**依据"哪一层连续无效"对症下药、甚至强制下一步动哪一层**，把「随机试探」升级为「有记忆、按过程的引导式搜索」——这就是三轴方法「无推理调度器控收敛」之上的**思考机制**。

### 3.2.6 同任务 AB 验证（思考 vs 非思考）

**关键：这是同数据集、同指标（`val_spearman`）的干净配对对照**，消除了跨任务的混淆变量。

任务：`Reg_TE_Muscle`（回归，决策指标 val_spearman）
- 🔵 **Thinking**：`RNAOmni_dev_thinking/worktrees/enhance_deepseekv4_freev3`（33 轮）
- ⚪ **Non-thinking**：`RNAOmni_v1/worktrees/deepseek-v4-flash_all_poolv2`（22 轮）

| 维度 | 🔵 Thinking | ⚪ Non-thinking |
|------|-------------|----------------|
| 数据集 | Reg_TE_Muscle | Reg_TE_Muscle |
| 决策指标 | val_spearman | val_spearman |
| 轮数 | 33 | 22 |
| **best** | **0.732** | 0.711 |
| 提升幅度 | 0.632→0.732（**+0.100**） | 0.660→0.711（+0.051） |
| `cause=` 分层记账 | ✅ 全程 | ❌ 无 |

**🔵 Thinking 轨迹**（先抓关键特征杠杆，再锁定微调）：
```
0.632
  │ [feature] struct_thermo(keep)→dotbracket(disc)→bpp(keep)
0.719 ← 关键跳跃：开 kmer k=3（单步 +0.08）
  │  [model] cause=head 试 pool_mlp（keep 但未超 baseline）
  │  之后：LR/HIDDEN/batch/dropout/EMA/WD 各 knob 在 winner 配方上双向穷举
0.732 ← 收敛：DROP 0.1→0.05 keep，其余全 disc 验证甜点
```

**⚪ Non-thinking 轨迹**（过早陷入骨架重构）：
```
0.660
  │ [feature] kmer k=3→4（keep）
0.711 ← 换头 pool_mlp + batch 128（keep）
  │  之后几乎全是 [model] pool_mlp 内部 rebuild/深度重构（大量 discard）
  │  再换 cls_mlp（keep 但停在 0.63）
0.711 停滞
```

**结论（同任务 AB）：**

在 `Reg_TE_Muscle` 这一干净配对下，**Thinking 模式优于 Non-thinking**——同样 0.66 起步，thinking 靠「第 4 轮 `kmer k=3` 单点识别关键特征（+0.08 最大单步）→ 随后锁定该配方双向穷举所有 knob 收敛」爬到 **0.732**；而 non-thinking 因**缺乏过程因果引导与特征层面的深挖杠杆，过早陷入 `pool_mlp→cls_mlp` 的骨架反复重构**，特征轴仅探索 2 次，停留在 **0.711**。

> **可归因解释**：Thinking 的价值在于——(1) 用 `cause=` 把预算引导到**最有效的特征/模型动作**（抓 kmer 关键杠杆），而非无差别地重构骨架；(2) 找到 winner 后**收敛不发散**（在锁定配方上穷举验证甜点），而非继续换骨架。这正是「按过程对症下药」与「机械随机试探」的差别。

> **边界条件（诚实记录）**：Thinking 的探索开销并不总是划算——在部分其它任务（如 pc3_te 等短预算/低数据场景）上，thinking 的频繁换头成本可能使其实测低于机械单轴精调。故 Thinking 的优势**存在任务/预算相关的边界条件**，本工作如实呈现正反两侧，不以单次结果定论。

### 3.2.7 两模式的统一公式体系

#### 统一记号

```
X        : 任务数据
F        : 特征配方（表示）          → 动作 do(F)
S        : 采样规模（数据轴）        → 动作 do(S)
M        : 模型骨架（预测头）        → 动作 do(M)
θ        : 模型参数（框架内旋钮）    → 动作 do(θ)
Y        : 预测
L ∈ {F, S, M, θ, crash}  : 动作层（do 的类别）
a = do(L)                : 一次干预动作
D(·)     : 验证决策指标（回归 val_spearman / 分类 val_accuracy）
λ        : min_delta（保留门槛）
best     : 当前全局最优 D
Φ        : 过程因果状态（Thinking 特有）
```

```
X ──► [清洗/划分 S] ──► 特征编码 F ──► 骨架 M ──► 参数 θ ──► Y
Y = f_θ^{M}( F_{S}(X) )
```

#### 1) 共享的 keep / discard（两模式一致）

```
D_before = max_history D              # 本轮前已知 best
D_new    = D( Y )                     # 本轮验证指标
δ        = D_new − D_before           # 提升量（higher_is_better）
keep:    δ ≥ λ  → 采纳 do(L)，更新 best
discard: δ < λ  → 回退 do(L)，best 不变（git reset）
```

#### 2) Thinking 模式（过程因果引导）

**动作空间（分层）：**
```
do(L):  L = F : 改特征配方（L0–L4 开关 / fusion / repr_layer / kmer k）
        L = S : 扩采样（K_train ×2 / full↔sample 切换）
        L = M : 换骨架（12 菜单，framework_base）
        L = θ : 框架内拧参（LR / HIDDEN / batch / dropout / EMA / loss）
        L = crash : 修 crash（短签名）
```

**过程分层结构：**
```
X ──┬──→ F ──→ Y
    ├──→ S ──→ Y
    ├──→ θ ──→ Y
    └──→ M ──→ Y
```

**选层 = 条件分层策略（π(动哪一层 | 过程证据)）：**
```
L_t = π( h_t | Φ_t )                        # Thinking：依过程卡点选层
其中 Φ_t = ( last_do, fail_F, fail_θ, fail_S, fail_H, suppressed )
     fail_L = 该层自上次有效 do(L) 以来连续无效次数
     suppressed ⊆ {H}                       # head 被压制

规则（瓶颈卡 next_do）：
    fail_F 高 → next_do = S   （数据少，扩采样）
    fail_θ 高 → next_do = F   （特征不够）
    head 被压 → next_do = F   （禁止换头）
    其余     → 自由（带 cause= 记账）
```

**过程记账（Thinking 特有）：**
```
每轮: cause_t = 实际动作层 ∈ {F,S,M,θ,crash}
      Φ_{t+1} = update( Φ_t, cause_t, keep/discard )   # 失败计数/压制更新
```

**骨架切换（Thinking 版）：**
```
framework_base(L=M): D_new 即使 < best 也保留为该头基线
换头条件: head ∉ suppressed  ∧  fw_streak ≥ patience
```

**完整 Thinking 单轮：**
```
L_t = π( h_t | Φ_t )                       # 思考选层
a_t = do(L_t) → 训练 → D_new
δ   = D_new − best
if δ ≥ λ:  best ← D_new;  keep(a_t)
else:      revert(a_t);    discard
Φ_{t+1} = update(Φ_t, cause_t, δ≥λ)        # cause 记账 + 压制
```

#### 3) Non-thinking 模式（机械随机）

**动作空间（同 Thinking，但无过程因果层）：**
```
do(L):  L ∈ {F, θ, M}    （无 cause= 记账，无 S 的因果触发）
```

**选层 = 自由/机械（无 π(·|Φ)）：**
```
L_t = 自由声明 feature/model    # 不依赖 Φ_t，不由连续失败决定
```

**骨架切换 = 纯计数随机：**
```
换头触发: fw_streak ≥ FRAMEWORK_PATIENCE     # 连续 discard 计数
新头:     M' = UniformRandom( 未试用菜单 )    # 均匀随机抽
```

**完整 Non-thinking 单轮：**
```
L_t = 自由声明（无 cause 约束，依 fw_streak 触发随机换头）
a_t = do(L_t) → 训练 → D_new
δ   = D_new − best
if δ ≥ λ:  best ← D_new;  keep(a_t)
else:      revert(a_t);    discard
```

#### 4) 两模式差异总结

| 环节 | Thinking | Non-thinking |
|------|----------|--------------|
| 选层 | `L_t = π(h_t \| Φ_t)`（过程因果） | `L_t = 自由`（无 Φ_t） |
| 状态 | 有 `Φ_t`(fail/suppressed) + `cause_t` | 无（仅全局 fw_streak） |
| 换头 | `head∉suppressed ∧ fw_streak≥patience` | `fw_streak ≥ patience` → 均匀随机 |
| 数据轴 | `do(S)` 由过程触发（特征卡→扩采样） | `do(S)` 独立于选层 |
| 收敛 | 依紧瓶颈（下一层 MUST） | 依计数（patience） |

**统一写成单框架 + 模式开关：**
```
单轮迭代:
  L_t = π_思考( h_t | Φ_t )      # Thinking：π_思考 = f(Φ_t)
      = π_机械( h_t )            # Non-thinking：π_机械 = 自由 + 随机换头
  do(L_t) → 训练 → D_new → keep/discard(λ) → 更新状态
```

#### 5) 一句话公式（横轴模型 / 纵轴特征的定量表达）

沿两轴的单步位移：
```
ΔF = 1{L_t=F} · g_F        # 特征轴位移（Thinking 先放大，单步大）
ΔM = 1{L_t∈{M,θ}} · g_M    # 模型轴位移（Non-thinking 反复重构）
ΔS = 1{L_t=S} · g_S        # 数据轴位移
```

**Thinking（先纵后横，特征收敛在前）：**
```
阶段1:  max_F metric(F | M₀, S₀)      → F*
阶段2:  max_M metric(F* | M, S₀)      → M*
落点:   (F*, M*)     ■ 两轴协同
```

**Non-thinking（单轴震荡，特征冻结）：**
```
仅:      max_M metric(M | F₀, S₀)      → M（反复重构）
落点:   (F₀, M')     ■ F 停滞在初始
```
