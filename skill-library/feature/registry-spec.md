# Registry / recipe 字段约定（S1 冻结）

> 拍板：recipe **最小白名单**；验收 **B**（`length_frac` 第二通道）；`prepare.py` / `rnaomni/prepare/` 编排仅人改。  
> 编码：train / `load_prepared_data` **现算**（无磁盘 feature cache / fingerprint）。  
> 阶段状态：[`../../devlog/20260719_000000_feature_skill_PLAN_PARTIAL.md`](../../devlog/20260719_000000_feature_skill_PLAN_PARTIAL.md)。

## 文件

| 文件 | 维护者 | 说明 |
|------|--------|------|
| `features/registry.yaml` | 人（或 L4 Deploy） | 已上架特征目录 |
| `features/builders/*` | 人（或 L4 Deploy） | 特征实现 |
| `features/pipeline.py` | 人 | 加载 registry/recipe 并拼接 |
| `features/active.yaml` | 人 + Agent（白名单键） | 当前启用配置 |
| `skill-library/registry.yaml` | 人 | **Skill** 注册（另一回事） |

## Recipe 白名单（S1 最小集 + kmer k / fusion / 子集融合）

Agent **仅**可改：

| 键 | 约束 |
|----|------|
| `recipe_id` | 字符串 |
| `fusion` | `concat`（默认）/ `add` / `mean`。其它名字加载失败。`add`/`mean` 要求**被纳入**通道 C 相同 |
| `fusion_include` | 可选。非空 list，只列 **enabled 且已注册** 的 name。省略 = 全部 enabled。含未启用 / 未注册 / 重复 → 加载失败。顺序即 concat 通道序 |
| `features[].name` | 必须已在 `features/registry.yaml` |
| `features[].enabled` | bool |
| `features[].params` | **只允许**该特征 `agent_tunable` 列出的键。当前：kmer `k`（1–4；C=`4^k`）；`rna_fm` `repr_layer`（0–12，默认 12）。`rinalmo` / `evo` / `evo2` 仍为 `[]`（只开关） |

禁止改：builder 代码、`features/registry.yaml`、`default_params`、deps、权重路径。

### `features/active.yaml` 形状

```yaml
recipe_id: s3_l0_length_frac_kmer
fusion: concat          # concat | add | mean
# fusion_include: [onehot, kmer]   # 可选；省略 = 全部 enabled
features:
  - name: onehot
    enabled: true
  - name: length_frac
    enabled: true
  - name: kmer
    enabled: true
    params:
      k: 4              # 可选；省略则 registry default k=3（64 维）
  - name: rna_fm
    enabled: false
    # params:
    #   repr_layer: 12   # 可选；仅 rna_fm，0–12
```

验收 B（历史）：将 `length_frac.enabled` 设为 `true` → `input_dim` 从 4→5。  
L0+L1+L2 默认 concat、k=3 → `input_dim=69`。k=2 → kmer C=16；k=4 → 256。registry 里 kmer `dim: 64` 只表示默认 k=3，encode 以现算 C 为准。  
`fusion_include: [onehot, kmer]` 且 k=3 → `input_dim=68`（不拼 `length_frac`）。

**注意：** recipe 中出现但未进 registry 的 name 一旦 `enabled: true` → `KeyError`。MSA 本版忽略，不要写进 recipe。

## Feature registry 条目

```yaml
name: onehot
layer: L0
dim: 4
cost: low
deps: []
cache_key: onehot_v1
module: features.builders.onehot
default_params: {}
agent_tunable: []
```

说明：`agent_tunable` 列出允许写进 recipe `features[].params` 的键。当前开放：kmer `[k]`、`rna_fm` `[repr_layer]`。`rinalmo` / `evo` / `evo2` 与多数条目仍为 `[]`（夜间只开关 `enabled`）。文档中的 `agent_open` **不是** registry 必填字段。

可选：`heavy: true` 和/或 `vram_gb`（≥12 视为 heavy）。`evo` / `evo2` 已标。编排器在 **full 且已启用轻量 L4**（`layer: L4` 且非 heavy，现为 `rna_fm` / `rinalmo`）时，禁止 MUST 这些 heavy 名。

L4 额外常见 `default_params`：`model_path`、`device`、（RNA-FM）`repr_layer`、（RiNALMo）`lm_config` / `code_root` 等，由人/Deploy 锁定。`repr_layer` 已进 `agent_tunable`；`lm_config` / 权重路径仍锁定。

## Builder 接口

```python
def build(sequences, *, max_len: int, alphabet: str = "ACGU", params=None) -> np.ndarray:
    # returns (N, max_len, dim) float32
```

`dim` 必须与 builder 输出的最后一维一致。kmer 的 registry `dim: 64` 是默认 k=3 的文档值；encode 按 `params.k` 现算 `4^k`，不要拿静态 64 去卡其它 k。

## 编码约定

`load_prepared_data` / `train.py` 按当前 `features/active.yaml` 现算特征；**无**磁盘 cache、**无** fingerprint。进程内 memo 仅限同一次 train 进程。

## `prepare.py` / `rnaomni/prepare/` 边界

| 可改（人） | 不可改（人/agent 合同） |
|------------|-------------------------|
| 读 registry/recipe、拼特征（经 pipeline） | folds、metric、`TIME_BUDGET`、`MAX_SEQ_LEN` 合同值 |
| | **Agent 不得修改 `prepare.py` 与 `rnaomni/prepare/` 源码** |
