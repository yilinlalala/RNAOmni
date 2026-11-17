# evo2 handoff (S2b-Deploy + S2b-Call)

## Status

**Registered and overnight-callable.** Deploy + Call done; overnight may toggle `enabled`. Default remains `false`.

| Item | Value |
|------|--------|
| Feature | `evo2` |
| Module | `features.builders.evo2` |
| Registry dim | 4096 |
| Checkpoint | `evo2_7b` (`tools/Evo2/weights/evo2_7b.pt`) |
| Pilot recipe | `onehot` + `evo2` → `input_dim=4100` |
| Encode (HEK unique 9619 / 14410 rows) | ~430 s (includes load); peak VRAM ~12.7 GB (C2) |
| Train smoke (C3) | n=256, steps=40; encode≈57 s; val_RMSE≈0.275; peak_vram≈13.5 GB |
| Overnight (C4) | **Allowed** to toggle `evo2.enabled` (same as other registered L4; tunable surface is `enabled` only) |
| Env | `rnaomni` (requires `flash_attn`; 7B does not need Transformer Engine) |
| Weight | `tools/Evo2/weights/evo2_7b.pt` (hardlinked from original dir) |

## How the agent uses this

1. Only change the `features/active.yaml` whitelist: `features[].enabled` (plus `recipe_id` / `fusion`∈{concat,add,mean} / optional `fusion_include`; kmer `params.k`; `rna_fm` `params.repr_layer`). 
2. May enable/disable `evo2` (`agent_tunable` is `[]`; do not write weight paths). When on, `input_dim` **+4096**; `train.py` must read `data["input_dim"]`; prefer smaller batch. 
3. Full encode wall-clock **does not count** toward `TIME_BUDGET=60s`, but still consumes startup time (HEK unique ~7 min, under the 36 min wall). Resident VRAM ~12–13 GB. 
4. `python train.py`; keep/discard by val_spearman. 
5. **Do not** change builders / registry / install packages / download weights.

Procedure: [agent-ops.md](../agent-ops.md) (includes L4 section).

## Redo checks

```bash
conda activate rnaomni
# place weights per README, then:
python prepare.py
python train.py
```
