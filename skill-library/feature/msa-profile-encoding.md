# MSA profile encoding（数据集绑定）

> **状态（2026-07-25）：半成品。** builder + `tools/msa_search/` + recipe 占位行已有；**尚未**写入 `features/registry.yaml` → 打开 `enabled` 会 `KeyError`。HEK 全量 hits/profiles 未覆盖。本地搜库作生产默认仍暂缓。  
> 人控预计算；agent **当前不要**开关 `msa_profile`。阶段见 [`../../devlog/20260719_000000_feature_skill_PLAN_PARTIAL.md`](../../devlog/20260719_000000_feature_skill_PLAN_PARTIAL.md)。

## 与数据集 / 双后端

| 概念 | 规则 |
|------|------|
| `dataset_id` | `Path(csv).stem` |
| 资产路径 | `artifacts/msa/<dataset_id>/`（**online 与 local 共用**） |
| 跳过 | 已有 `hits/.../<hash>.json` 且 `status=ok` → 不重搜（`--force` 除外） |
| 本地库 | `artifacts/msa_dbs/<db_id>/`（与数据集目录分离） |
| 训练读取 | builder 默认用 `prepare.ACTIVE_CSV` 的 stem |

本地安装与下载：**[MANUAL_SETUP_LOCAL.md](../../tools/msa_search/MANUAL_SETUP_LOCAL.md)**（RNAcentral active + pyhmmer）。

## 闭环前须补

1. 在 `features/registry.yaml` 登记 `msa_profile`（dim=6、module、`default_params`）。  
2. 人控跑满当前 `ACTIVE_CSV` 的 precompute（hits + profiles）。  
3. 冒烟训通后再在 handoff/规程中开放夜间。

## 流程（上架完成后）

```bash
python tools/msa_search/run_precompute.py --csv <active>.csv --stage all
# registry 已登记后：recipe 打开 msa_profile → train（只读该 dataset 的 npy）
```

## 通道（dim=6）

`freq_A/C/G/U`, `conservation`, `gap_freq`（见 `contract.PROFILE_CHANNELS`）。

## 红线

- Agent 不跑 precompute、不改 builder 源码。  
- 缺 profile → builder 侧 `missing: zeros`（上架闭环后）。  
- 不同 CSV **不得**共用同一 `artifacts/msa/<id>/`（除非确认序列集合相同且你有意为之）。
