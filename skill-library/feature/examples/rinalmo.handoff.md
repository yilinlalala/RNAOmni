# rinalmo handoff (S2b-Deploy + S2b-Call)

## Status

**Registered and overnight-callable.** Deploy + Call smoke done; overnight may toggle `enabled`.

| Item | Value |
|------|--------|
| Feature | `rinalmo` |
| Module | `features.builders.rinalmo` |
| Registry dim | 480 (micro) |
| Pilot recipe | `onehot` + `rinalmo` → `input_dim=484` |
| D3 / D4 | `SMOKE_OK` / `BUILDER_SMOKE_OK` |
| C3 train smoke | n=256, steps=40; encode≈17s; val_RMSE≈0.34; peak_vram≈279 MB |
| Overnight (C4) | **Allowed** to toggle `rinalmo.enabled` (same as other registered features; tunable surface is recipe `enabled`) |
| Env | `myenv3.11` |
| Weight | `tools/RiNALMo/weights/rinalmo_micro_pretrained.pt` (registry `default_params.model_path`) |

## How the agent uses this

1. Only change the `features/active.yaml` whitelist: `features[].enabled` (plus `recipe_id` / `fusion`∈{concat,add,mean} / optional `fusion_include`; kmer `params.k`; `rna_fm` `params.repr_layer`).  
2. May enable/disable `rinalmo` (`agent_tunable` stays `[]`; do not write `lm_config`); common pairing is `onehot` (watch `input_dim`; full encode wall-clock is long and does not count toward `TIME_BUDGET`).  
3. `train.py` must read `data["input_dim"]`; for wide channels, adjust architecture/batch.  
4. `python train.py`; keep/discard by val_spearman.  
5. **Do not** change builders / registry / install packages / download weights.

Procedure: [agent-ops.md](../agent-ops.md) (includes L4 section). SSOT: todolist §8.1.

## Redo checks

```bash
conda activate myenv3.11
cd /path/to/RNAOmni
CODE_ROOT=/path/to/RNAOmni/tools/RiNALMo
PYTHONPATH="$CODE_ROOT" python tools/smoke_rinalmo.py --code-root "$CODE_ROOT"
python tools/smoke_rinalmo_builder.py --feature-name rinalmo
python tools/smoke_rinalmo_train.py --n 256 --steps 40
```
