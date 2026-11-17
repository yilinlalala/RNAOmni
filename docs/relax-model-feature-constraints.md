# 放宽模型 / 特征约束（落地步骤）

> 时间戳：`2026-08-22`  
> 类型：`PLAN`  
> 状态：`PARTIAL`（步骤 1–3 已落地；**B0** / **A1** 已落地；A2 可选；过程提示另文，先不做）  
> 范围：在 **不复活 S5、不改评测合同** 的前提下，扩大夜间可复盘的动作空间。  
> 相关：`features/registry.yaml`、`features/active.yaml`、`skill-library/feature/registry-spec.md`、`skill-library/feature/agent-ops.md`、`skill-library/feature/frameworks.md`、`rnaomni/orch/search.py`、`rnaomni/orch/prompts.py`、`program.md`  
> 对照：`[../devlog/20260816_154000_task_head_skill_PLAN_TODO.md](../devlog/20260816_154000_task_head_skill_PLAN_TODO.md)`（§16 两层搜索仍有效）  
> 过程提示备忘：[`process-hint.md`](process-hint.md)；落地步骤：[`process-hint-plan.md`](process-hint-plan.md)

过夜数字不动：`TIME_BUDGET=60s`、encode 36 min 墙、full-then-sample、`min_delta`、全局 `patience=15`。  
不改：`prepare.py` / `rnaomni/prepare/` / builders / 评测 / folds / `rnaomni.env`。

---

## 本批范围

**做（已落地）：** 扩大已登记菜单上的合法旋钮（L4 开关、`agent_tunable`、`fusion_include`、框架内超参清单）。  
**待实现：** 无（本文件自由度批次）。过程提示仍另文。  
**不做：** 瓶颈卡、`cause=` / `because` / `do`、按诊断提前换头、受控 `joint` 注入、`π(h|X)`。  
**忽略：** `msa_profile` / MSA（不上 registry、不进 recipe、不排期）。

换头仍由编排器在平台期从菜单均匀随机抽。Agent 不得自创框架名。

---

## 一句话

现在窄的不是 keep/discard，而是 **可选积木少、每块积木上能拧的旋钮少**。  
放开方式是：**扩大已登记菜单和 `agent_tunable`**——不是让 agent 写新 builder、自创架构名，也不是先上过程提示。

**原则：** 自由度加在「那一层已经合法的动作」上。加菜单 ≠ 复活 S5。

---

## 现状 → 本批之后

### 特征


| 通道                                                | 夜间能改                                                                                 |
| ------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `onehot` / `length_frac` / `struct_*` / `rinalmo` / `evo` / `evo2` | `enabled`                                                                            |
| `kmer`                                            | `enabled` + `k ∈ 1–4`                                                                |
| `rna_fm`                                          | `enabled` + `repr_layer ∈ 0–12`（默认 12）                                               |
| 融合                                                | `concat` / `add` / `mean`；可选 `fusion_include`（只列 enabled 且已注册的 name；省略 = 全部 enabled） |


`rinalmo` / `evo` / `evo2` 保持 `agent_tunable: []`：只开关。`evo` / `evo2` 为 4096-d / ~12 GB 显存。

### 模型

- 菜单 12 个：`pool_mlp` / `attn_pool_mlp` / `resnet1d_pool` / `cls_mlp` / `tf_mean` / `tf_cls` / `tf_attn` / `tf_conv` / `max_pool_mlp` / `gated_pool_mlp` / `gru_pool` / `mean_deep_mlp`
- 开局 **`baseline`**（不进菜单，现行 `Predictor = PooledMLP`）；连续 10 次 `[model]` discard 后从**完整菜单**均匀随机（含 `pool_mlp`）
- 换头第一枪只换积木，继承 `HIDDEN_DIM` / `LR` / batch / 损失 / EMA
- 框架内可拧（不换骨架）：`HIDDEN_DIM` / `LR` / `BATCH_SIZE` / dropout / 损失 / EMA；`resnet1d_pool`：`kernel ∈ {3, 7}`，residual blocks ≤ 4
- Agent **必须**实现注入的名字，不得发明菜单外类名
- 位点 / 配对头仍禁止。人写好的序列级 Transformer **可以**进菜单（见 A）；禁止 agent 把 Transformer / Mamba / GNN 当「试一枪」自由写入

### 明确不放（全程）

- Agent 改 `features/builders/`、`features/registry.yaml`、`features/pipeline.py`、`prepare.py`、`rnaomni/prepare/`
- Agent 自创框架名，或把 Transformer / Mamba / GNN 当「试一枪」自由写入（人写最小 `forward` 再进菜单 ≠ 这条）
- 位点 / 配对头（评测仍是单序列单标签）
- 装包、下载新权重、改 folds / metric / `TIME_BUDGET`
- 用旧战役 raw best 或 LLM 常识直接当换头策略

---

## 落地顺序（自由度）

### 步骤 1 — 特征：用满已上架通道 — **DONE**

L4（`rna_fm` / `rinalmo` / `evo` / `evo2`）是合法 `[feature]` 动作；缺本地权重 → crash，禁止 pip / 下载。  
`struct_dotbracket` 可开（已注册）。

**触及：** `skill-library/feature/agent-ops.md`、`rnaomni/orch/prompts.py`、L4 handoff

### 步骤 2 — 特征：扩大 `agent_tunable` + 子集融合 — **DONE**

**2a.** 人改 registry：`rna_fm` 开放 `repr_layer`（0–12）。`rinalmo` 仍 `[]`。kmer 维持 `k` 1–4。`struct_`* 仍 `[]`。

**2b.** `fusion_include`：省略 = 全部 enabled concat；列出则只融这些通道（须 enabled 且已注册）。`add` / `mean` 语义不变，仍要求被纳入通道 C 相同。

非法 key / 非法 `repr_layer` / `fusion_include` 含未启用或未注册 name → 加载失败。

**触及：** `features/pipeline.py`、`features/registry.yaml`、`skill-library/feature/registry-spec.md`、agent-ops、prompts、`tools/smoke_kmer_fusion.py`

### 步骤 3 — 模型：框架内超参收成明菜单 — **DONE**

规程清单（不改训练代码路径）：`HIDDEN_DIM` / `LR` / `BATCH_SIZE` / dropout / 损失 / EMA；`resnet1d_pool` kernel 与 blocks。  
换头第一枪规则不变：只换积木，继承 `WIDTH` 与训练旋钮。

**触及：** `skill-library/feature/frameworks.md`、agent-ops、`program.md`、prompts、`train.py` 注释

---

## 待实现：A 空间 / B 日程

两件独立，可只做一件。A 解决「可选骨架太少」；B 解决「`pool_mlp` 先占满前半场预算」。  
**不要**和过程提示捆在一起。抽法仍是菜单均匀随机，不排号、不由 agent 点名。

共用契约（A/B 都不改）：

```
x (B, L, C) + lengths  →  标量 / num_classes
  stem → WIDTH=HIDDEN_DIM → 积木（全程 WIDTH）→ mask 池化 → MLP
```

换头第一枪：只换积木，继承 `WIDTH` 与训练旋钮；`finalize` 仍实例化 `Predictor`。

### 实现顺序（准备落地）

先 **B** 后 **A 首批**。B 不依赖新积木；A 不改开局。两批之间跑一遍 `python tools/smoke_framework_search.py`。不要和特征 yaml / 过程提示捆在一起。

| 批次 | 做什么 | 为什么先/后 | 主要触及 |
|------|--------|-------------|----------|
| **B0** | 开局改成 `baseline`（不进菜单） | **DONE** | `search.py`、prompts、`frameworks.md`、agent-ops、`program.md`、`smoke_framework_search.py` |
| **A1** | 契约 + 菜单 + 可跑的 `tf_mean` / `tf_cls` | **DONE** | `FRAMEWORK_MENU`、`frameworks.md`、`train.py`（`TfMean`/`TfCls`，`Predictor` 仍是 PooledMLP）、`tools/smoke_tf_heads.py` |
| **A2** | （可选，另批）`tf_attn` / `tf_conv` / `max_pool_mlp` / `gated_pool_mlp` / `gru_pool` / `mean_deep_mlp` | 首批过夜先看 `tf_*` 会不会炸 | 同 A1 模式 |

**建议拍板（实现时当约束，不再讨论）：**

- 开局名就叫 `baseline`，不进 `FRAMEWORK_MENU`。`remaining_frameworks()` 现有逻辑已是「菜单 \ tried」，tried=`[baseline]` 时池子自动含 `pool_mlp`。
- `train.py` 开局仍 `Predictor = PooledMLP`（这就是 baseline 的实现）。B 只改编排器标签，不改这份 forward。
- Transformer 类由人写进 `train.py`（或 `train.py` 可 import 的同文件 class），夜间换头 = `Predictor = TfMean` / `TfCls` + 按契约接 `WIDTH`。不要只丢 md 让 agent 现场写 `TransformerEncoder`。
- `FRAMEWORK_PATIENCE=10`、全局 `patience=15`、`min_delta`、`TIME_BUDGET` 不动。
- 旧战役 TSV 里的 `framework=pool_mlp` 回放仍按行解析；只有 **空 TSV / 新战役** 才默认 `baseline`。

#### B0 步骤 — **DONE**

1. `search.py`：`DEFAULT_FRAMEWORK = "baseline"`。`FRAMEWORK_MENU` 本步不动。确认 `baseline` **不**在菜单里。
2. `_reset_framework_tracking`：无 TSV 时 active/tried = `baseline`（现在是 `pool_mlp`）。`keep_active=True`（full→sample）仍保留当前名。
3. `_apply_framework_round`：缺 `framework=` 时回落到 `active` / `baseline`，不要再写死 `pool_mlp`。`framework_base` 切到菜单名后，`baseline` 留在 tried 里但不参与抽签（它本就不在菜单）。
4. `prompts.py` `format_framework_block`：缺 state 时 `active=baseline`；stay 文案不要写死 `pool_mlp`。Iter：未设 `next_framework` 时只拧旋钮、description 带 `framework=baseline`。
5. `frameworks.md` / agent-ops / `program.md` / SKILL：开局是 `baseline`；平台期从完整菜单抽（含 `pool_mlp`）。
6. 改 `tools/smoke_framework_search.py`（旧例假设开局 `pool_mlp` 且第一次抽 ≠ `pool_mlp`，会红）：
   - 开局 `active=baseline`，`remaining` 含 `pool_mlp` 及现有另外三个
   - 未满 10 次 `[model]` discard → `next` 空
   - 满 10 次 → 抽出菜单之一（**可以是** `pool_mlp`）
   - sticky / 抽空 / feature keep 后重进池：原语义保留，把「开局名」从 `pool_mlp` 换成 `baseline`
   - 保留一组「已经在 `pool_mlp` 上搜」的旧路径：tried 含 `pool_mlp` 时下一抽 ≠ `pool_mlp`（回放旧 TSV）
7. 跑 `python tools/smoke_framework_search.py`。不跑整夜、不改 `train.py` 的 `Predictor`。

#### A1 步骤（B0 合并进 master 之后） — **DONE**

1. `frameworks.md` 表加 `tf_mean` / `tf_cls`：`pool_dim=WIDTH`；1–2 层 encoder；`nhead=4`；`dim_ff∈{2,4}×WIDTH`；`src_key_padding_mask` 来自 `lengths`。红线：禁止位点/配对头、禁止自创名；**允许**菜单上的 `tf_*`。
2. `search.py` `FRAMEWORK_MENU` 末尾追加 `"tf_mean", "tf_cls"`。不要改 `DEFAULT_FRAMEWORK`。
3. `train.py` 增加两个 class（最小可训），构造函数仍是 `input_dim, hidden_dim, dropout, num_outputs`；**不**改 `Predictor = PooledMLP`。
4. agent-ops / Iter：被抽到时只改别名 + 按 WIDTH 重接头，禁止顺带拧 LR。菜单列表同步。
5. `tools/smoke_framework_search.py`：菜单长度 6；tried=`[baseline]` 时剩余含 `tf_mean`/`tf_cls`/`pool_mlp`。
6. 短 smoke（对标 `smoke_attn_pool_mlp.py`）：随机 (B,L,C)+lengths，`tf_mean`/`tf_cls` forward 形状对，可选 1 fold 几步不 crash。不进过夜必跑集。
7. 确认 `finalize` 仍只认 `Predictor`（不用改逻辑）。

#### A2 以后再说 — **DONE**

第二批名字按同一模式：先契约，再 class，再进菜单，再 smoke。一次加一个也可以。

### A — 菜单加积木（空间） — **DONE**（A1 + A2）

人先写最小 `forward`，再把名字放进 `FRAMEWORK_MENU`。加菜单 ≠ 自由发明。  
现有 `attn_pool_mlp` 只是长度上一次 attention 池化，**不是** Transformer encoder。

**首批要加的名字**

| name | 积木 | 上限 |
|------|------|------|
| `tf_mean` | 1–2 层 `TransformerEncoder` → mean-pool | `nhead=4`，`dim_ff≈2–4×WIDTH`，mask 来自 `lengths` |
| `tf_cls` | 学一个 CLS，拼在长度前，encoder 后取 CLS | 同上；与现有 `cls_mlp` 的差别是 CLS **过了** self-attention |

**可选第二批（不和首批捆 commit）**

| name | 积木 |
|------|------|
| `tf_attn` | encoder 后再 attn-pool |
| `tf_conv` | k=7 conv 再 1 层 encoder 再 mean |
| `max_pool_mlp` | 只 max-pool |
| `gated_pool_mlp` | 位置门控加权和 |
| `gru_pool` | 一层双向 GRU 再 pool |
| `mean_deep_mlp` | 无 conv，mean-pool 后更深 MLP |

**要做**

1. 在 `frameworks.md` 为每个新名字写死：tensor 路径、`pool_dim`、层数/`nhead`/`dim_ff` 上限、padding mask。红线改为：禁止位点/配对头、禁止 agent 自创名；**允许**菜单上的 `tf_*`。
2. `FRAMEWORK_MENU` 追加首批名字（`tf_mean`、`tf_cls`）。开局逻辑本步不动（那是 B）。
3. 人写最小可跑的 `forward` 示例（可放 `train.py` 注释块或 `skill-library/feature/examples/`），agent 被抽到时必须实现注入的那个名字为 `Predictor`。
4. 换头第一枪规则同步到 agent-ops / `program.md` / Iter prompt：实现该名字，继承 `HIDDEN_DIM` / `LR` / batch / 损失 / EMA，不改 recipe。
5. `tools/smoke_framework_search.py`：菜单含新名；抽签能抽到；抽空不发明新名。
6. 可选短 smoke：`tf_mean` / `tf_cls` 一次 `forward` + 1 fold 不 crash（不进过夜必跑集）。
7. `finalize` 继续只认 `Predictor`，无则回退现有逻辑。

**不做（A）**

- 12 层预训练 Transformer（那是 L4 `rna_fm` / `rinalmo`；`evo` / `evo2` 是另一路 L4 编码器）
- token / 配对头
- 让 agent 自己起 `MyBert` 之类的名字
- 与 B、与特征步骤 2 同一 commit

### B — 开局无名 baseline（日程） — **DONE**

第一枪就是当前 `train.py`，**不**占用菜单名、**不**享受「先搜 10 轮才换」的特权。  
平台期后再从 **完整菜单** 抽，`pool_mlp` 不再免抽。菜单可以仍是现在 4 个（A 未做时），也可以是 A 之后的更长菜单。

**要做**

1. 开局 `active_framework` 改为非菜单名（例如 `baseline` / `current`），**不要**放进 `FRAMEWORK_MENU`。
2. 第一轮：按文件原样训，description 带 `framework=baseline`（或 `current`）。不要在开局强制改成 named 头。
3. `frameworks_tried` 开局只记 `baseline`（或不记菜单名）。`remaining_frameworks()` = 完整菜单，**含** `pool_mlp`。
4. 平台期触发不变：连续 10 次 `[model]` discard（keep 清零；crash / `[sample]` / `framework_base` / `[feature]` 仍不计）。
5. `ensure_next_framework`：从完整剩余菜单均匀随机；sticky 与 run-tag 种子不变。
6. 切到 named 头之后：`baseline` 不再被抽。feature keep / 样本 keep：清空 tried，只留当前 `active`；若仍是 `baseline`，菜单全员可抽。
7. TSV 回放不要默认 `active=pool_mlp`。Iter prompt / frameworks.md / agent-ops / `program.md`：未到 `next_framework` 时停留在 baseline，只拧框架内旋钮。
8. `tools/smoke_framework_search.py` 增例：
   - 开局 `active=baseline`，`remaining` 含 `pool_mlp`
   - 未满 10 次 → `next_framework` 空
   - 满 10 次 → 抽出菜单中的一个（可以是 `pool_mlp`）
   - 菜单抽空 → `next` 空，不发明新名

**不做（B）**

- 开局就从菜单随机抽一套（跨战役不可比；若以后要做，另开，不叫 B）
- 缩短 `FRAMEWORK_PATIENCE` / 改 `min_delta` / 全局 patience
- 让 agent 挑下一个名字
- 与 A 同一 commit（A 未做完时，B 的菜单就是现在 4 个）

---

## 以后再做（本文件不排期）

- **过程提示**（见 `process-hint.md`）：瓶颈卡、`cause=`、按诊断提前换头、特征 keep 后的适配轮注入。在此之前换头保持均匀随机。
- `rinalmo.lm_config`：仅当本地确有多档权重再开。
- 位点/配对头、复活 S5、Agent 改 L0–L2 builder：不做。

---

## 红线（落地时对照）

- 不复活独立 `model` / `experiment` skill（S5）。
- 不改评测指标、划分、`TIME_BUDGET`。
- 不让 agent 成为换头执行者：抽哪个名字仍由编排器在平台期后从菜单抽。
- 不读 `run.log`；crash 仍只给短签名。
- 本批不加 `cause=` 协议。
- MSA / `msa_profile` 本版忽略：不上架、不进 recipe。
- A 与 B 不要捆同一 commit。

---

## 完成定义（本批 1–3）

1. L4 与已声明 `agent_tunable` / 子集 concat 可在只改 yaml 的情况下跑通 encode（`input_dim` 随纳入通道变）。
2. 框架内拧参与换骨架在规程上分账；换头第一枪仍继承 `WIDTH`。
3. builders / registry 写作权、S5、评测合同均未打开。
4. 未引入 `cause=` / 瓶颈卡 / 提前换头。

## 完成定义（A / B，落地后改 DONE）

**A：** `FRAMEWORK_MENU` 含 A1+A2 名字；`frameworks.md` 有契约；被抽到的第一枪即使差于全局 best 也留下 `framework_base`；agent 未发明菜单外名字。 — **DONE**  
**B：** 开局 `framework=baseline` 且不在菜单里；满 10 次 `[model]` discard 后能抽到 `pool_mlp`；未满 10 次不换 named 头。 — **DONE**

