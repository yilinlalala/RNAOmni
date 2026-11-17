# Feature deploy-ops（Deploy Agent：L4 部署 + 注册）

> 阶段状态：[`../../devlog/20260719_000000_feature_skill_PLAN_PARTIAL.md`](../../devlog/20260719_000000_feature_skill_PLAN_PARTIAL.md) **§8.1**（S2b Deploy + Call 已完成）。  
> **人触发**（Cursor 对话或 cursor_sdk **单独脚本**）。**不经** `orchestrator.py`。  
> **禁止**注入 overnight / `PROMPT_SETUP` / `PROMPT_ITER`。夜间规程见 [agent-ops.md](agent-ops.md)。  
> **资产驱动**：`examples/<feature>.assets.yaml` + CLI `--code-root` / `--weight`（少写死）；解析见 `tools/pretrained_assets.py`（只认本地路径）。

## 与 Overnight / Call 分界

| 角色 | 文档 | 做什么 | 状态（2026-07-25） |
|------|------|--------|-------------------|
| **Deploy**（本文） | `deploy-ops.md` | 本地预训练 → Embedding 冒烟 → builder + registry + recipe 占位（`enabled: false`）+ handoff 草稿 | ✅ `rna_fm` / `rinalmo` / `evo` / `evo2` |
| **Call** | 人工短跑 + 更新 handoff | 打开 `enabled`、训通、全量耗时/显存、决定是否夜间开放（C4） | ✅ 四模均已 Call；见 `examples/*.handoff.md` |
| **Overnight** | `agent-ops.md` | 只改 recipe 白名单 + `train.py`；升 val_spearman | 可开关已 Call 的 L4 |

Deploy **完成** = 特征已注册且默认 `enabled: false`。  
Deploy **不做** = 打开 `enabled`、正式训通、改 overnight prompts（那些属 Call / Overnight）。

## 触发（参数化一键）

```bash
conda activate myenv3.11
cd /path/to/RNAOmni_v1
export CURSOR_API_KEY=...   # required unless --dry-run

# RiNALMo（默认 assets；权重须已在 README 的 weights/）
python -m rnaomni.deploy \
  --assets skill-library/feature/examples/rinalmo.assets.yaml \
  --code-root tools/RiNALMo \
  --dry-run

python -m rnaomni.deploy \
  --assets skill-library/feature/examples/rinalmo.assets.yaml \
  --code-root tools/RiNALMo

# RNA-FM（显式 assets）
python -m rnaomni.deploy \
  --assets skill-library/feature/examples/rna_fm.assets.yaml \
  --dry-run

# 可选：锁定某一本地权重（跳过 candidates 解析）
python -m rnaomni.deploy --assets .../rinalmo.assets.yaml \
  --code-root /abs/RiNALMo --weight /abs/RiNALMo/weights/rinalmo_giga_pretrained.pt
```

- **不经** `orchestrator.py`；overnight prompts 不变。  
- 共用解析：`tools/pretrained_assets.py`（本地优先，禁止下载）。

## D1 资产：RiNALMo（第二模；默认 micro）

权威清单：[`examples/rinalmo.assets.yaml`](examples/rinalmo.assets.yaml)。摘要：

| 键 | 约定 |
|----|------|
| `feature_name` / `module` | `rinalmo` / `features.builders.rinalmo` |
| `conda_env` | `myenv3.11` |
| `code_root` | `tools/RiNALMo`；CLI `--code-root` 覆盖 |
| `package_dir` | `rinalmo` |
| 权重落点 | `{code_root}/weights/`（官方 README） |
| 默认解析 | candidates 按 **micro → mega → giga** 取首个已存在文件 |
| `expected_dim_doc` | 480（micro）；mega=640 / giga=1280 |
| recipe | `enabled: false` |

人预放权重（示例只下最小）：

```bash
CODE_ROOT=tools/RiNALMo
mkdir -p "$CODE_ROOT/weights"
# wget .../rinalmo_micro_pretrained.pt -O "$CODE_ROOT/weights/rinalmo_micro_pretrained.pt"
```

冒烟：

```bash
PYTHONPATH="$CODE_ROOT" python tools/smoke_rinalmo.py --code-root "$CODE_ROOT"
python tools/smoke_rinalmo_builder.py --feature-name rinalmo
```

交接：[`examples/rinalmo.handoff.md`](examples/rinalmo.handoff.md)。

## D1 资产：RNA-FM（首模）

权威清单：[`examples/rna_fm.assets.yaml`](examples/rna_fm.assets.yaml)。摘要：

| 键 | 冻结值 |
|----|--------|
| `feature_name` / `module` | `rna_fm` / `features.builders.rna_fm` |
| `conda_env` | `myenv3.11` |
| `code_root` | `tools/RNA-FM`（`PYTHONPATH=<repo>/tools/RNA-FM`） |
| `expected_dim_doc` | `640`（文档提示；**registry.dim 以 D3 冒烟为准**） |
| `repr_layer` | `12` |
| 权重候选（先存在者优先） | `tools/RNA-FM/pretrained/RNA-FM_pretrained.pth` |
| recipe 占位 | `name: rna_fm`, `enabled: false` |

权重放在上游约定目录（README：`.pth` 进 `pretrained/`）：

```bash
mkdir -p tools/RNA-FM/pretrained
cp RNA-FM_pretrained.pth tools/RNA-FM/pretrained/RNA-FM_pretrained.pth
```

注册时把 **解析到的绝对路径** 写入 `default_params.model_path`。  
并行 worktree：`parallel_lanes.provision_batch` 把这两个目录 symlink 到主仓（与 `dataset/forcompare` 同一招）；overnight **禁止** wget / hub 下载。冒烟结果仍在 `artifacts/pretrained/`（ignore）。

## 安装与依赖策略

1. **默认零安装**：`PYTHONPATH=<code_root>`，不 `pip install` 整包，不按官方 `environment.yml` 重建 env。  
2. 目标仅 **Embedding**；RiNALMo 默认 `use_flash_attn=false`（attention 已懒加载 flash_attn）。  
3. **缺库**：停下与用户沟通；可对照 `{code_root}/environment.yml` / `setup.py`；禁止擅自在线装包。  
4. **版本差**：冒烟通过则忽略官方 pin；失败再协商 Plan B。  
5. 权重：**只认本地**；禁止 gdown / hub / Zenodo 自动下载。

## 红线

- 禁止无 `--no-index` 的联网装包；禁止无本地路径的权重下载。  
- 禁止改 `orchestrator.py`、`PROMPT_*`、folds / metric / `TIME_BUDGET`。  
- recipe 新条目必须 **`enabled: false`**。  
- 缺库必须问人（非 overnight 无人值守）。

## 可写路径（Deploy 白名单，随 feature_name）

| 路径 | 动作 |
|------|------|
| `features/builders/<feature>.py` | 新建 / 修改 |
| `features/registry.yaml` | 按 [registry-spec.md](registry-spec.md) L4 字段新增 |
| `features/active.yaml` | 仅增占位行，`enabled: false` |
| `artifacts/pretrained/<feature>/` | 冒烟结果（不 commit 大权重） |
| `skill-library/feature/examples/<feature>.handoff.md` | D5 交接 |
| 本 skill 文档备注 | 可选 |

**不可写（本竖切）**：`train.py`、`prepare.py`、`rnaomni/prepare/`、评测与 fold 合同、overnight prompts。

## 执行清单（D3–D5）

1. 读对应 `*.assets.yaml`；确认 `code_root` 与某一 `weight_candidates` 路径存在。  
2. `conda activate myenv3.11`；跑冒烟（RiNALMo 示例）：

```bash
python tools/smoke_rinalmo.py --code-root <CODE_ROOT>
# → artifacts/pretrained/rinalmo/smoke_result.json
```

3. **D3**：记录真实 `dim` / device / `model_path`（registry.dim 以冒烟为准）。  
4. **D4**：builder + registry + recipe `enabled: false`；`python tools/smoke_rinalmo_builder.py`。  
5. **D5**：写 `examples/<feature>.handoff.md`（已注册、默认 `enabled: false` → 交给 Call 更新 C2–C4）。

### D3 零安装注意

| 模型 | 处理 |
|------|------|
| RNA-FM | stub `ptflops`；`torch.load(..., weights_only=False)` |
| RiNALMo | 勿用 `get_pretrained_model`；`model_config(lm)+load_state_dict`；`use_flash_attn=false` |

## 验收（Deploy）

- [ ] registry 可解析到 `features.builders.<feature>`  
- [ ] builder 冒烟 `C == registry.dim`，无 NaN  
- [ ] `default_params.model_path`（或 auto-resolve）指向存在的本地文件  
- [ ] 默认 `enabled: false`  
- [ ] **未**要求跑通 `train.py`  
- [ ] D5 交接：`examples/<feature>.handoff.md`
