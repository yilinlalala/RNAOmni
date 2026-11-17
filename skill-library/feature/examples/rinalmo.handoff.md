# rinalmo handoff（S2b-Deploy + S2b-Call）

## Status

**Registered and overnight-callable.** Deploy + Call smoke 完成；夜间可开关 `enabled`。

| Item | Value |
|------|--------|
| Feature | `rinalmo` |
| Module | `features.builders.rinalmo` |
| Registry dim | 480 (micro) |
| Pilot recipe | `onehot` + `rinalmo` → `input_dim=484` |
| D3 / D4 | `SMOKE_OK` / `BUILDER_SMOKE_OK` |
| C3 train smoke | n=256, steps=40；encode≈17s；val_RMSE≈0.34；peak_vram≈279 MB |
| Overnight (C4) | **允许**开关 `rinalmo.enabled`（与其它已注册特征相同；调参面为 recipe `enabled`） |
| Env | `myenv3.11` |
| Weight | `tools/RiNALMo/weights/rinalmo_micro_pretrained.pt`（registry `default_params.model_path`） |

## Agent 怎么用

1. 只改 `features/active.yaml` 白名单：`features[].enabled`（及 `recipe_id` / `fusion`∈{concat,add,mean} / 可选 `fusion_include`；kmer `params.k`；`rna_fm` `params.repr_layer`）。  
2. 可开/关 `rinalmo`（`agent_tunable` 仍为 `[]`，不要写 `lm_config`）；常见搭配 `onehot`（注意 `input_dim`；全量 encode 墙钟长、不计入 `TIME_BUDGET`）。  
3. `train.py` 必须读 `data["input_dim"]`；大通道时可改结构/batch。  
4. `python train.py`；按 val_spearman keep/discard。  
5. **禁止**改 builders / registry / 装包 / 下权重。

规程：[agent-ops.md](../agent-ops.md)（含 L4 节）。SSOT：todolist §8.1。

## Redo checks

```bash
conda activate myenv3.11
cd /path/to/RNAOmni
CODE_ROOT=/path/to/RNAOmni/tools/RiNALMo
PYTHONPATH="$CODE_ROOT" python tools/smoke_rinalmo.py --code-root "$CODE_ROOT"
python tools/smoke_rinalmo_builder.py --feature-name rinalmo
python tools/smoke_rinalmo_train.py --n 256 --steps 40
```
