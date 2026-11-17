# rna_fm handoff (S2b-Deploy + S2b-Call)

## Status

**Registered and overnight-callable.** Deploy + Call (through C4) done.

| Item | Value |
|------|--------|
| Feature | `rna_fm` |
| Module | `features.builders.rna_fm` |
| Registry dim | 640 |
| Pilot recipe | `onehot` + `rna_fm` → `input_dim=644` (see `examples/active.with_rna_fm.yaml`) |
| Encode (full 14410) | ~81 s; peak encode VRAM ~460 MB (C2) |
| Train smoke (C3) | val_RMSE≈0.907; train≈295 s; peak_vram≈873 MB; no OOM |
| Overnight (C4) | **Allowed** to toggle `rna_fm.enabled` normally (same as other registered features) |
| Env | `myenv3.11` |
| Weight | `tools/RNA-FM/pretrained/RNA-FM_pretrained.pth` (registry `default_params.model_path`) |

## How the agent uses this

1. Only change the `features/active.yaml` whitelist: `features[].enabled` (plus `recipe_id` / `fusion`∈{concat,add,mean} / optional `fusion_include`; kmer `params.k`; `rna_fm` `params.repr_layer` 0–12). 
2. May enable/disable `rna_fm`; may tune `repr_layer` (default 12); optional pairing with `onehot` / `length_frac` / `kmer` (watch `input_dim`). 
3. `train.py` must read `data["input_dim"]`; for wide channels, adjust architecture/batch. 
4. `python train.py`; keep/discard by val_spearman. 
5. **Do not** change builders / registry / install packages / download weights.

Procedure: [agent-ops.md](../agent-ops.md) (includes L4 section). SSOT: todolist §8.1.

## Redo checks

```bash
conda activate myenv3.11
python tools/smoke_rna_fm_builder.py
CUDA_VISIBLE_DEVICES=0 python tools/bench_rna_fm_encode.py --mode onehot_rna_fm
```
