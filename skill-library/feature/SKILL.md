---
name: rnaomni-feature
description: >-
  Human-controlled RNA feature pipeline for RNAOmni: register builders and
  pretrained encoders, edit features/active.yaml whitelist keys, adapt train.py
  fusion and input_dim. Use when working on features, recipes, INPUT_DIM,
  pretrained RNA embeddings, orchestrator.py overnight feature rounds,
  PROMPT_SETUP/PROMPT_ITER, or workspace/<stem>.tsv rows tagged [feature].
---

# RNAOmni Feature（人控路径 H）

Agent 规程入口。注册见 [`../registry.yaml`](../registry.yaml)。阶段状态见 [`../../devlog/20260719_000000_feature_skill_PLAN_PARTIAL.md`](../../devlog/20260719_000000_feature_skill_PLAN_PARTIAL.md)。

**现状（2026-08-22）：** S0–S3 完成；L3 结构与 L4（`rna_fm` / `rinalmo` / `evo` / `evo2`）Deploy+Call 完成；MSA 半成品；下一阶段 S4 整夜验收。

## 路径（强制）

- **人**：实现 `features/builders/*`、写 `features/registry.yaml`、上架预训练与锁定参数；L4 可用 `python -m rnaomni.deploy`。  
- **Agent**：只改 `features/active.yaml`（白名单键）与 `train.py`。无磁盘 feature cache：下次 `train.py` 启动时按当前 recipe 现算。  
  - **orchestrator 夜间**：只 edit+commit（步骤 1–3），立刻 STOP；`train.py` / 写 TSV / keep/discard reset 由编排器做。  
  - **交互、不经 orchestrator**：agent 自己跑完整 1–6。  
- **禁止**：改 builders / registry / `prepare.py` / `rnaomni/prepare/` / 评测 / folds / `TIME_BUDGET` / `rnaomni.env` / `.env` / 装包 / 下载新权重 / 跑 MSA precompute。

## 详规（按需打开）

| 角色 | 文档 |
|------|------|
| 人上架（手工 / 结构 / MSA） | [human-ops.md](human-ops.md) |
| L4 Deploy（人触发，非 overnight） | [deploy-ops.md](deploy-ops.md) |
| 夜间一轮 | [agent-ops.md](agent-ops.md) |
| 预测头框架 | [frameworks.md](frameworks.md)（开局 `baseline`；菜单含 `tf_*` / `gru_pool` 等） |
| registry / recipe 字段 | [registry-spec.md](registry-spec.md)（白名单：fusion `concat\|add\|mean`；可选 `fusion_include`；kmer `k` 1–4；`rna_fm.repr_layer` 0–12） |
| L3 结构 | [structure-encoding.md](structure-encoding.md) |
| MSA | [msa-profile-encoding.md](msa-profile-encoding.md) |

## Agent 快规则

1. 开轮读 `workspace/<stem>.tsv`、当前 recipe、`program.md` 合同。  
2. 声明 `round_type`: `feature` | `model`（首版避免 `joint`）。Iter 若注入了 `next_framework`，本轮必须 `model` 并实现那个名字（见 [frameworks.md](frameworks.md)）。瓶颈卡 `next_do=feature` 时 MUST 跟。开局是 `baseline`。  
3. 单轮一个主假设；description 前缀 `[feature]` 或 `[model]`。`[model]` 须带 `framework=<name>`；换骨架第一枪再加 `framework_base cause=head`。另写 `cause=<head|feature|knob>`（修 crash 时 `cause=crash`）。无 `next_framework` 时不要写 `cause=head`。  
4. 只 commit 白名单文件。夜间：提交后立刻 STOP，不要跑 `train.py`、不要改 TSV；discard 的 `git reset` 由编排器做。交互会话才由 agent 训、写 TSV、reset。  
5. 决策指标：回归 **val_spearman**（越高越好）/ 分类 **val_accuracy**（越高越好）；keep 须相对 best ≥ min_delta（见 Iter prompt / `agent-ops.md`）。  
6. 只能 `enabled` **已在** `features/registry.yaml` 的 name。MSA 本版忽略。

## 目录

```text
features/registry.yaml      # 人维护
features/builders/          # 人维护（含 L3/L4）
features/pipeline.py        # 人维护；agent 禁改
features/active.yaml        # Agent 可改（最小白名单键）
prepare.py                  # 门面；Agent 不改源码
rnaomni/prepare/            # 编排实现；Agent 不改
rnaomni/                    # 其余实现（orch / results / archive / …）
train.py                    # Agent 可改；用 data["input_dim"]
# L4 Deploy：python -m rnaomni.deploy
```

Recipe 白名单见 [registry-spec.md](registry-spec.md)。默认 concat + kmer k=3 时 L0+L1+L2 为 `input_dim=69`；现行 `features/active.yaml` 还开了部分 L3。夜间可改：已注册 `enabled`（含 L4 `rna_fm` / `rinalmo` / `evo` / `evo2`）、kmer `params.k`（1–4）、`rna_fm` `params.repr_layer`（0–12）、`fusion` / `fusion_include`。`rinalmo` / `evo` / `evo2` 只开关。缺 L4 本地权重会 crash，禁止 pip / 下载。

## 示例

**夜间 feature 轮（orchestrator）**  
读 `features/active.yaml` → 启用已注册特征 → 必要时改 `train.py` 适配 `input_dim` → commit → STOP（`train.py` / 记 TSV / keep/discard 由编排器做）。

**交互 feature 轮（不经 orchestrator）**  
同上，但 agent 自己 `train.py` → 记 TSV → keep/discard。

**人上架 L4**  
读对应 `examples/<name>.assets.yaml` → `python -m rnaomni.deploy` → Call / 写 `examples/<name>.handoff.md` → 再开放夜间 `enabled`。
