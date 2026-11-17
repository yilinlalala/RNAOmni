# Feature agent-ops（夜间一轮）

> 可改 `features/active.yaml` 白名单键。  
> `prepare.py` / `rnaomni/prepare/` / `features/pipeline.py` / builders / registry / `rnaomni.env` **禁止**修改。  
> **无磁盘特征 cache**：改 recipe 后下次 `train.py` 启动时按当前 recipe 编码（谁跑 train 见下）。  
> **无人值守**：不问人确认。orchestrator 夜间：agent 做完 1–3 立刻 STOP；4–6 由编排器做。交互会话：agent 做完整 1–6。  
> 阶段状态：[`../../devlog/20260719_000000_feature_skill_PLAN_PARTIAL.md`](../../devlog/20260719_000000_feature_skill_PLAN_PARTIAL.md)。

## Recipe 白名单（最小集）

仅：`recipe_id`、`fusion`（`concat` | `add` | `mean`）、可选 `fusion_include`（非空 list，只列 **enabled 且已注册** 的 name；省略 = 全部 enabled）、`features[].name`、`features[].enabled`、以及该 name 的 `agent_tunable` 键（kmer `params.k` 整数 1–4；`rna_fm` `params.repr_layer` 整数 0–12）。  
`name` 必须已在 `features/registry.yaml`。`add` / `mean` 要求**被纳入**通道 C 相同（例如 onehot(4)+kmer(64) 不能 add）。建议一轮只改一个旋钮：一个 `enabled`、一个 tunable、或 `fusion` / `fusion_include`。

## 轮次类型

| `round_type` | 改动 | 额外 prepare |
|--------------|------|----------------|
| `model` | 仅 `train.py` | 否 |
| `feature` | `features/active.yaml` ± `train.py` | 否（train 时现算特征） |
| `joint` | 两者大改 | 否（首版避免） |

建议配额：约 70% model / 25% feature，**仅当战役笔记没有 MUST、且瓶颈卡 `next_do` 为空**。Iter 注入的战役笔记 `MUST this round` 点到具体已注册 name / 明旋钮时，本轮必须执行那一刀，禁止无笔记依据地默认再拧 LR 或再 concat 一个特征。错写 `round_type` 不改 keep/discard。  
`rna_fm` / `rinalmo` / `evo` / `evo2`（L4）是合法 `[feature]` 动作（开关；`rna_fm` 还可拧 `repr_layer`）。缺本地权重 → crash，禁止 pip / 下载。`evo` / `evo2` 为 4096-d / ~12 GB 显存（registry `heavy: true`）；**full 且已有轻量 L4**（`rna_fm` / `rinalmo`）时编排器把它们从合法 MUST 里剔除。discard / crash 过的 feature 整场不再 MUST；旋钮按 `(framework, knob)` 记，换头后可再试。crash 修锅若无新 commit：清 veto，不计入 `max_agent_failures`，下一轮走笔记里仍合法的 next。  
当前框架连续 10 次 `[model]` discard 后，若笔记没有点合法剩余头，编排器仍从剩余菜单均匀抽一张并注入 `next_framework`。笔记在至少一次 keep/discard 之后可以点 remaining 里的名字，编排器采用该名（sticky），不必等满 10 次。开局 `next=head` 忽略。见 [frameworks.md](frameworks.md)。开局是 `baseline`。不要自己另挑名字。否决顺序：crash 块（必须修短签名，禁止换层）> 扩 K > `next_framework` > 笔记 MUST。禁止用 `cause=head` 自挑骨架名。  
`[model]` 不换骨架时只拧框架内菜单（`HIDDEN_DIM` / `LR` / `BATCH_SIZE` / dropout / 损失 / EMA；`resnet1d_pool` 的 kernel 与 residual blocks），见 [frameworks.md](frameworks.md)。框内 keep/discard 对照 **该头** `framework_best`（可把全局 discard 升成 keep）。换骨架第一枪仍在 TSV 留下 `framework_base` 当该头基线；若未抬过全局 best，编排器 `git reset` 回全局最佳 HEAD，不把较差头当成后续 F/θ 的工作底座。

## 单轮步骤

步骤清单与 keep/discard 规则两种模式相同。**谁执行 4–6 不同：**

| 步骤 | orchestrator 夜间 | 交互、不经 orchestrator |
|------|-------------------|-------------------------|
| 1–3 读状态、声明假设、白名单 commit | agent；提交后立刻 STOP。禁止跑 `train.py`、看 `workspace/run.log`、改 TSV。夜间上一轮 crash 的一行摘要由编排器写入 Iter prompt | agent |
| 4–6 `train.py`、写 TSV、keep/discard reset | **orchestrator** | agent 自己做 |

1. 读 `workspace/<stem>.tsv`、`features/active.yaml`、git HEAD、本 skill / handoff。Iter 若注入了战役笔记，先读 `MUST this round`：无否决时执行该 name/旋钮。crash 块 / `next_framework` 仍优先。不要自己点骨架名；换头只实现注入的 `next_framework`。  
2. 声明 `round_type` + 一个主假设（勿重复已 discard 描述）。扩采样 `[sample]` 行已经用新 `K_train` 重训过当前 best：不要再 checkout / 重跑历史 commit，本轮必须是新假设。  
3. 只改白名单文件 → `git commit`（不 commit `workspace/` 下的 tsv）。无新改动则 STOP，不要空 commit（orchestrator 不会对未变 HEAD 再训）。夜间到此 STOP。  
4. `python train.py > workspace/run.log 2>&1`（内部 `load_prepared_data` 按 recipe 编码）。  
5. 记 `workspace/<stem>.tsv`（前缀 `[feature]` / `[model]`）。`[model]` 带 `framework=<name>`（开局 `framework=baseline`）；换骨架第一枪加 `framework_base`。闭集 `cause=<head|feature|knob|crash>`（`sample` 由编排器写）。无 `next_framework` 时不要写 `cause=head`（记账为拧参，不换头）。换骨架第一枪：`[model] framework=<name> framework_base cause=head …`。  
6. 决策指标相对 **current best** 改善 ≥ **min_delta** → keep；更小改善或变差 → `git reset` 到实验前 HEAD。  
   - 回归：**val_spearman** 更高更好  
   - 分类：**val_accuracy** 更高更好  
   - `min_delta` 数值见 `rnaomni.env`；orchestrator Iter prompt 注入，与停搜 patience 共用

## 已注册特征（可开关）

以 `features/registry.yaml` 为准。当前包括：

| 层 | name | 备注 |
|----|------|------|
| L0–L2 | `onehot` / `length_frac` / `kmer` | 低 cost；常作基线 |
| L3 | `struct_thermo` / `struct_dotbracket` / `struct_bpp` | ViennaRNA；比 L0–L2 慢；共享 `_vienna` LRU |
| L4 | `rna_fm` / `rinalmo` / `evo` / `evo2` | 高 cost；见对应 `examples/*.handoff.md`（C4 已允许开关） |

只能 `enabled` 已在 `features/registry.yaml` 的 name。未登记名加载失败。MSA 本版忽略。

开放判定：**已注册 + handoff/规程允许**。`agent_tunable`：kmer `[k]`、`rna_fm` `[repr_layer]`；`rinalmo` 与其余多为 `[]`（只开关 `enabled`）。无单独的 `agent_open` 字段。

## L4 注意

- 全量 encode 墙钟**不计入** `TIME_BUDGET=60s`（探索），但仍挤占启动时间；每轮新进程无跨轮 memo。
- 大 `input_dim` 时须改 `train.py` 适配（读 `data["input_dim"]`），可调 batch / 结构防 OOM。  
- 缺本地权重 → crash；**禁止** pip / 下载。  
- 细则与复验命令：`examples/rna_fm.handoff.md`、`examples/rinalmo.handoff.md`、`examples/evo.handoff.md`、`examples/evo2.handoff.md`。

## 红线

- 不改 `features/builders/`、`features/registry.yaml`、`prepare.py`、`rnaomni/prepare/`、`rnaomni.env`、`.env`、folds / metric / `TIME_BUDGET`。  
- 不装包、不下新权重、不跑 MSA precompute。  
- 决策仅主指标：回归 **val_spearman**；分类 **val_accuracy**；keep 须 ≥ min_delta。禁止改划分 / 评估 / `TIME_BUDGET`。  
- `train.py` 必须用 `data["input_dim"]`，禁止写死 `4`；分类必须用 `data["num_classes"]`，禁止写死类别数。  
- 换框架只实现 Iter 注入的 `next_framework`（契约 [frameworks.md](frameworks.md)）。禁止发明菜单外的类名。  
- Deploy（`python -m rnaomni.deploy`）不在 overnight 职责内。
