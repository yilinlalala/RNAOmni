# RNAOmni Setup

本项目（原 RNAClaw，现更名为 RNAOmni）的主要设置与参数汇总。

> 来源：`rnaomni.env`、`features/registry.yaml`、`features/active.yaml`、`program.md`。
> 环境变量以 `rnaomni.env` 为默认配置表（agent 不得修改）；可选 `.env` overlay 仅覆盖差异键。

## 环境与数据集

| 类别 | 键 | 默认值 | 说明 |
|------|----|--------|------|
| 数据集 | `RNAOMNI_ACTIVE_CSV` | `HEK_te_prediction_task.csv` | 主 CSV 文件（main 管线） |
| 数据集 | `RNAOMNI_USE_COMPARE_SPLITS` | `0` | 0 = 主 CSV 管线；另设 `RNAOMNI_COMPARE_ROOT` 走外部预划分 |
| 数据集 | `RNAOMNI_COMPARE_ROOT` | （未设置） | 命名外部 `train/val/test` 划分根目录（不重新清洗/划分） |
| 运行 | `RNAOMNI_CONDA_ENV` | `myenv3.11` | conda 环境名（ViennaRNA 编码依赖） |
| 运行 | `TIME_BUDGET` | `300` | 每次训练时间预算（秒），编码/启动不计 |
| 运行 | `MAX_SEQ_LEN` | `1024` | 最大序列长度 |
| 运行 | `RNAOMNI_CPU_THREADS` | `4` | 每进程 BLAS/OpenMP/torch 线程数（0 = 不限制） |
| 运行 | `RNAOMNI_MAX_RUNNING_LANES` | `9` | 并行 laner 允许的最大 running 任务数（0/负 = 不限） |
| 运行 | `RNAOMNI_ENCODE_CHUNK` | `4096` | 特征编码分块大小（峰值 RAM ≈ 一块 × 最宽通道） |

## 数据划分（main CSV 路线）

| 键 | 默认值 | 说明 |
|----|--------|------|
| `N_FOLDS` | `5` | 交叉验证折数 |
| `TEST_FRACTION` | `0.1` | 回归测试留出比例 |
| `CLS_TRAIN_FRAC` | `0.7` | 分类训练集比例 |
| `CLS_VAL_FRAC` | `0.2` | 分类验证集比例 |
| `CLS_TEST_FRAC` | `0.1` | 分类测试集比例 |
| `CLS_MMSEQS_MIN_SEQ_ID` | `0.8` | MMseqs 最小序列一致度 |
| `CLS_MMSEQS_COVERAGE` | `0.8` | MMseqs 覆盖度 |

> 分类用 MMseqs → 簇感知划分；回归用测试留出 + CV。分类还需 `mmseqs`（或 `MMSEQS_BIN`）在 PATH。

## 采样（sampling，HEK + COMPARE）

| 键 | 默认值 | 说明 |
|----|--------|------|
| `OVERNIGHT_HOURS` | `6` | 过夜运行时长上限 |
| `ENCODE_TIMEOUT_SEC` | `5400` | 编码墙钟超时（秒） |
| `SAMPLE_ROW_CAP_TRAIN` | `1000` | 回归采样训练行数上限 |
| `COMPARE_PER_CLASS_TRAIN` | `10` | 外部划分分类每类采样数（训练） |
| `COMPARE_PER_CLASS_EVAL` | `10` | 外部划分分类每类采样数（评估） |
| `COMPARE_SAMPLE_SEED` | `0` | 采样随机种子 |
| `RNAOMNI_DATA_MODE` | （由 orchestrator 决定） | `full` / `sample`，不由 env 固定 |

> 采样启动=全量（HEK 与 COMPARE 都从 full 开始）；编码超时后切换 sample 并重置搜索 best。

## 搜索参数

| 键 | 默认值 | 说明 |
|----|--------|------|
| `MIN_DELTA_CLASSIFICATION` | `3e-4` | 分类保留阈值（val_accuracy 提升下限） |
| `MIN_DELTA_REGRESSION` | `1e-3` | 回归保留阈值（val_spearman 提升下限） |
| `PATIENCE` | `15` | 连续非显著轮次后停止搜索 |
| `FRAMEWORK_PATIENCE` | `10` | 单一命名 head 连续 `[model]` 丢弃后切换 |
| `MIN_ITERATIONS` | `10` | 最小迭代数 |
| `SAMPLE_EXPAND_PATIENCE` | `3` | 采样模式下连续非提升后扩展 K_train / 行数上限 |

## Agent / 后端

| 键 | 默认值 | 说明 |
|----|--------|------|
| `AGENT_BACKEND` | `opencode` | 传输后端（`--backend cursor` 用 Cursor SDK） |
| `OPENCODE_MODEL` | `opencode/deepseek-v4-flash-free` | 使用的模型 |
| `OPENCODE_BIN` | （未设置） | 可留空 |
| `CURSOR_MODEL` / `CURSOR_API_KEY` | （未设置） | 仅 `--backend cursor` 时用（密钥只放 `.env`） |

## 特征（features）

当前激活配方（`features/active.yaml`）：

| 键 | 值 |
|----|----|
| `recipe_id` | `evo2_only` |
| `fusion` | `concat` |
| `evo2.enabled` | `true` |
| 其余特征（onehot / length_frac / kmer / struct_* / rna_fm / rinalmo / evo） | `false` |

特征目录（`features/registry.yaml`）：

| 名称 | 层 | 维度 | 成本 | 依赖 | 可调参数（agent_tunable） |
|------|----|------|------|------|--------------------------|
| `onehot` | L0 | 4 | low | — | — |
| `length_frac` | L1 | 1 | low | — | — |
| `kmer` | L2 | 64（默认 k=3） | low | — | `k` 1–4 |
| `struct_thermo` | L3 | 3 | medium | ViennaRNA | — |
| `struct_dotbracket` | L3 | 3 | medium | ViennaRNA | — |
| `struct_bpp` | L3 | 2 | medium | ViennaRNA | — |
| `rna_fm` | L4 | 640 | high | torch | `repr_layer` 0–12（默认 12） |
| `rinalmo` | L4 | 480 | high | torch, ml_collections | — |
| `evo` | L4 | 4096 | high（heavy, ~12 GB VRAM） | torch, transformers, flash_attn | — |
| `evo2` | L4 | 4096 | high（heavy, ~12 GB VRAM） | torch, flash_attn | — |

> 说明：`evo` / `evo2` 为 4096 维、重负载（heavy），本地权重（`tools/Evo`、`tools/Evo2/weights`），不允许在线下载；`rna_fm` / `rinalmo` / `evo` / `evo2` 为 L4 编码，可开关。

## 模型骨架（named frameworks）

来源：`skill-library/feature/frameworks.md`。开局骨架为 `baseline`（当前 `train.py` / `Predictor = PooledMLP`），**不算菜单名**。骨架菜单（12 个命名头）：

`pool_mlp` | `attn_pool_mlp` | `resnet1d_pool` | `cls_mlp` | `tf_mean` | `tf_cls` | `tf_attn` | `tf_conv` | `max_pool_mlp` | `gated_pool_mlp` | `gru_pool` | `mean_deep_mlp`

> Orchestrator 注入 `active_framework` / `next_framework`；agent 只实现注入的名字，不得发明菜单外类名（禁止 Mamba / GNN；token / pairwise head 出契约）。

| name | pool | `pool_dim` | 说明 |
|------|------|------------|------|
| `pool_mlp` | mean+max | `2*WIDTH` | Named rebuild: stem→WIDTH 后 conv residual at WIDTH。开局 `baseline` 可为 Conv(C→ch→C)+residual-on-C |
| `attn_pool_mlp` | attn+max | `2*WIDTH` | 沿长度 attention，`-inf` mask pad；拼 attn-sum 与 max |
| `resnet1d_pool` | mean | `WIDTH` | 1D ResNet，channels=`WIDTH`，**≤4** 个 residual 块，kernel 3 或 7 |
| `cls_mlp` | learned vector | `WIDTH` | 单个学习汇总向量（CLS 式）over length + MLP，**无** Transformer encoder |
| `tf_mean` | mean | `WIDTH` | 1–2 层 `TransformerEncoder`，`nhead=4`，`dim_ff=2×WIDTH`，`src_key_padding_mask` from `lengths` |
| `tf_cls` | CLS token | `WIDTH` | 同 encoder；预置学习 CLS，取编码后 CLS |
| `tf_attn` | attn | `WIDTH` | 同 `tf_mean` encoder，再长度 attention-pool |
| `tf_conv` | mean | `WIDTH` | k=7 conv residual at WIDTH，然后 **1** 层 encoder layer，再 mean |
| `max_pool_mlp` | max | `WIDTH` | 仅 stem，无 conv / encoder |
| `gated_pool_mlp` | gated sum | `WIDTH` | 每位置标量 gate，softmax over length |
| `gru_pool` | mean | `2*WIDTH` | 1 层双向 GRU；mean-pool 输出 |
| `mean_deep_mlp` | mean | `WIDTH` | stem + mean-pool + 更深 MLP，无 conv / encoder |

通用规则：

- 张量路径：`x (B,L,C)` → stem 至 `WIDTH`（C==WIDTH 时恒等）→ blocks → masked pool `(B,pool_dim)` → MLP。`C = data["input_dim"]`，不硬编码。
- `tf_*` 的 `WIDTH` 建议被 4 整除（`nhead=4`；代码回退 `nhead=1`）。这不是 L4 `rna_fm`。
- 头部首层 `Linear(pool_dim, WIDTH)`（浅层 `resnet1d_pool` / `cls_mlp` 可为 `Linear(WIDTH, num_outputs)`）；换骨架时 `pool_dim` 变化需重建该 Linear，勿遗留陈旧 `Linear(2*C, …)`。
- 换骨架第一枪只 swap `Predictor` 别名，继承 `HIDDEN_DIM` / `LR` / `BATCH_SIZE` / loss / EMA / dropout，不回调 LR/batch。

**框架内旋钮（不换骨架）**：

| 旋钮 | 约束 |
|------|------|
| `HIDDEN_DIM` / `LR` / `BATCH_SIZE` / dropout | 正值；换骨架第一枪继承这些 |
| Loss / EMA | 如 `LABEL_SMOOTHING`、`EMA_DECAY` — 保留或去掉，不新造损失族 |
| `resnet1d_pool` only | kernel ∈ {3, 7}；residual 块 ≤ 4 |

`Predictor` 别名：`class ...(nn.Module)` 后 `Predictor = ...`，构造器接受 `input_dim`、`hidden_dim`、`dropout`、`num_outputs`；非 `pool_mlp` 头不应要求 `conv_channels`。

## 决策与搜索协议（orchestrator）

| 规则 | 详情 |
|------|------|
| 决策指标 | 回归 `val_spearman`（越高越好）；分类 `val_accuracy`（越高越好） |
| 保留阈值 | 相对当前 best 提升 ≥ `min_delta` 才保留；否则丢弃 + 重置 |
| 停止条件 | 最大迭代数、连续 agent 失败、或 patience（`15` 次非显著轮次后） |
| Iter 循环 | 每次 Iter = 仅编辑+提交，STOP；orchestrator 在 agent 外部运行 `train.py` 并执行 keep/discard `git reset` |
| VRAM | 软约束，允许适度提升以换取有意义的指标增益 |
| 简单性准则 | 其他条件相同时选更简单方案 |

## FINAL（finalize）关键参数

来源：`rnaomni/finalize.py`。仅记录关键参数（CLI 参数，可覆盖默认值）。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--commit` | `auto` | 要 finalize 的 best commit（hash 或 `auto`） |
| `--early-stop` / `--no-early-stop` | `False` | 早停开关 |
| `--patience` | `5` | 早停下无 val 提升即停的 evals 数 |
| `--min-delta-ratio` | `0.002` | 相对搜索 val RMSE 的阈值 |
| `--seed` | `42` | 随机种子 |
