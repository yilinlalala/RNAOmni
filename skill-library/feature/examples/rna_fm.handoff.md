# rna_fm handoff（S2b-Deploy + S2b-Call）

## Status

**Registered and overnight-callable.** Deploy + Call（至 C4）完成。

| Item | Value |
|------|--------|
| Feature | `rna_fm` |
| Module | `features.builders.rna_fm` |
| Registry dim | 640 |
| Pilot recipe | `onehot` + `rna_fm` → `input_dim=644`（见 `examples/active.with_rna_fm.yaml`） |
| Encode (full 14410) | ~81 s；peak encode VRAM ~460 MB（C2） |
| Train smoke (C3) | val_RMSE≈0.907；train≈295 s；peak_vram≈873 MB；无 OOM |
| Overnight (C4) | **允许正常开关** `rna_fm.enabled`（与其它已注册特征相同） |
| Env | `myenv3.11` |
| Weight | `tools/RNA-FM/pretrained/RNA-FM_pretrained.pth`（registry `default_params.model_path`） |

## Agent 怎么用

1. 只改 `features/active.yaml` 白名单：`features[].enabled`（及 `recipe_id` / `fusion`∈{concat,add,mean} / 可选 `fusion_include`；kmer `params.k`；`rna_fm` `params.repr_layer` 0–12）。  
2. 可开/关 `rna_fm`；可拧 `repr_layer`（默认 12）；可选搭配 `onehot` / `length_frac` / `kmer`（注意 `input_dim`）。  
3. `train.py` 必须读 `data["input_dim"]`；大通道时可改结构/batch。  
4. `python train.py`；按 val_spearman keep/discard。  
5. **禁止**改 builders / registry / 装包 / 下权重。

规程：[agent-ops.md](../agent-ops.md)（含 L4 节）。SSOT：todolist §8.1。

## Redo checks

```bash
conda activate myenv3.11
python tools/smoke_rna_fm_builder.py
CUDA_VISIBLE_DEVICES=0 python tools/bench_rna_fm_encode.py --mode onehot_rna_fm
```
