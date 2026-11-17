# L3 structure encoding（ViennaRNA 描述符）

> 挂在 `rnaomni-feature` 下。  
> **Agent 开放**：三条均已写入 `features/registry.yaml` 与 `features/active.yaml`；夜间可通过白名单改 `enabled`（无单独的 `agent_open` 字段）。  
> 环境：`myenv3.11`（`import RNA` / ViennaRNA）。不依赖 CLI `RNAfold`。  
> 阶段状态：[`../../devlog/20260719_000000_feature_skill_PLAN_PARTIAL.md`](../../devlog/20260719_000000_feature_skill_PLAN_PARTIAL.md)。

## 信息 → 通道

| Registry 名 | 信息 | 形状约定 | dim | agent |
|-------------|------|----------|-----|-------|
| `struct_thermo` | MFE/L、ensemble gap（MFE−ensemble FE）、mean positional entropy | 有效位广播同一标量三元组 | 3 | 可开关 |
| `struct_dotbracket` | MFE 点括号 | per-pos one-hot：`(` / `)` / `.` | 3 | 可开关 |
| `struct_bpp` | 配分函数：配对概率、positional entropy | per-pos 两通道 | 2 | 可开关 |

共享实现：`features/builders/_vienna.py`（`fold_sequence` + 进程内 LRU，多 builder 同批不重折）。

## 上架与开放状态

- `features/registry.yaml`：`layer: L3`，`deps: [ViennaRNA]`，`cost: medium`，`agent_tunable: []`（调参面是 recipe `enabled`）  
- `features/active.yaml`：三条已列出；默认多为 `false`，**agent 可自行打开**  
- Agent **禁止**装包；依赖须已在环境预装  

## 夜间用法

在 feature 轮直接改 `enabled`，例如：

- `[feature] enable struct_thermo`
- `[feature] enable struct_bpp; keep thermo off`
- `[feature] enable struct_dotbracket + struct_thermo`

打开后 `input_dim` += 对应 `dim`。`train.py` 须读 `data["input_dim"]`。

示例 recipe：`examples/active.with_struct_thermo.yaml`

## 验收速检（人）

```bash
conda activate myenv3.11
python tools/smoke_struct_builders.py
```

全量编码在 `train.py` 启动时现算；L3 比 L0–L2 慢，注意挤占启动墙钟（编码不计 `TIME_BUDGET`，但仍影响体感）。
