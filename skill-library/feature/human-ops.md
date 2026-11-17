# Feature human-ops（人控上架）

> 编排在 `prepare.py` / `rnaomni/prepare/` / `features/pipeline.py`；agent 不改这些文件，也不改 `rnaomni.env`。  
> 特征在 `train` / `load_prepared_data` 调用时按 recipe **现算**，不写磁盘 feature cache。  
> 阶段状态：[`../../devlog/20260719_000000_feature_skill_PLAN_PARTIAL.md`](../../devlog/20260719_000000_feature_skill_PLAN_PARTIAL.md)。

## 上架清单（手工 / L3）

1. 选型；记录许可、显存、耗时。  
2. 区分**锁定参数**与 agent 可调（S1 recipe 侧实质仅 `enabled`）。  
3. 实现 `features/builders/<name>.py`（`build(...) -> (N,L,C)`）。  
4. 写入 `features/registry.yaml`（`dim`、`module`、`cache_key`、`default_params`、`cost`、`deps`）。  
5. 在 `features/active.yaml` 增加条目（默认 `enabled: false`）。  
6. `python prepare.py`（分 fold + 冒烟编码）；确认打印的 `input_dim`。  
7. 人工跑通 `train.py`。  
8. 文档化（必要时写 examples / encoding 专文）后再对夜间开放。

L3 结构见 [structure-encoding.md](structure-encoding.md)；smoke：`python tools/smoke_struct_builders.py`。

## L4 预训练

优先走 **Deploy**（[deploy-ops.md](deploy-ops.md) + `python -m rnaomni.deploy` + `examples/<name>.assets.yaml`）：

1. 人预放本地权重（禁止 Deploy/agent 自动下载）。  
2. Deploy：Embedding 冒烟 → builder + registry + recipe 占位（`enabled: false`）。  
3. **Call**：打开启用、训通、量全量耗时/显存，写 `examples/<name>.handoff.md`。  
4. handoff 标明 C4 后，夜间才鼓励开关 `enabled`。

已完成：`rna_fm`、`rinalmo`、`evo`、`evo2`（见对应 handoff）。

## MSA

人控预计算资产 + builder；上架须 **registry 与 recipe 同步**。现状见 [msa-profile-encoding.md](msa-profile-encoding.md) 与 SSOT §8.3（半成品：builder/recipe 有、registry/全量资产缺）。

## S1 验收 B（历史）

- Builder：`length_frac`（dim=1）  
- 打开后相对 L0-only：`input_dim` 4→5  

示例：`examples/active.with_length_frac.yaml`
