# evo2 handoff（S2b-Deploy + S2b-Call）

## Status

**Registered and overnight-callable.** Deploy + Call 完成；夜间可开关 `enabled`。默认仍 `false`。

| Item | Value |
|------|--------|
| Feature | `evo2` |
| Module | `features.builders.evo2` |
| Registry dim | 4096 |
| Checkpoint | `evo2_7b`（`tools/Evo2/weights/evo2_7b.pt`） |
| Pilot recipe | `onehot` + `evo2` → `input_dim=4100` |
| Encode (HEK unique 9619 / 14410 rows) | ~430 s（含加载）；peak VRAM ~12.7 GB（C2） |
| Train smoke (C3) | n=256, steps=40；encode≈57 s；val_RMSE≈0.275；peak_vram≈13.5 GB |
| Overnight (C4) | **允许**开关 `evo2.enabled`（与其它已注册 L4 相同；调参面仅为 `enabled`） |
| Env | `myenv3.11`（需 `flash_attn`；7B 不需要 Transformer Engine） |
| Weight | `tools/Evo2/weights/evo2_7b.pt`（原目录硬链接） |

## Agent 怎么用

1. 只改 `features/active.yaml` 白名单：`features[].enabled`（及 `recipe_id` / `fusion`∈{concat,add,mean} / 可选 `fusion_include`；kmer `params.k`；`rna_fm` `params.repr_layer`）。  
2. 可开/关 `evo2`（`agent_tunable` 为 `[]`，不要写权重路径）。打开后 `input_dim` **+4096**；`train.py` 必须读 `data["input_dim"]`，建议降 batch。  
3. 全量 encode 墙钟**不计入** `TIME_BUDGET=60s`，但仍挤占启动时间（HEK unique ~7 min，低于 36 min 墙）。显存常驻 ~12–13 GB。  
4. `python train.py`；按 val_spearman keep/discard。  
5. **禁止**改 builders / registry / 装包 / 下权重。

规程：[agent-ops.md](../agent-ops.md)（含 L4 节）。

## Redo checks

```bash
conda activate myenv3.11
PYTHONPATH=tools/Evo2 python tools/smoke_evo2.py
python tools/smoke_evo2_builder.py
CUDA_VISIBLE_DEVICES=3 python tools/smoke_evo2_train.py --n 256 --steps 40
CUDA_VISIBLE_DEVICES=3 python tools/bench_evo2_encode.py --mode evo2_only
```
