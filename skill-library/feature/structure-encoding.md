# L3 structure encoding (ViennaRNA descriptors)

> Under `rnaomni-feature`. 
> **Agent open:** all three are already in `features/registry.yaml` and `features/active.yaml`; overnight may change `enabled` via the whitelist (no separate `agent_open` field). 
> Env: `myenv3.11` (`import RNA` / ViennaRNA). Does not depend on CLI `RNAfold`.
## Information → channels

| Registry name | Information | Shape convention | dim | agent |
|-------------|------|----------|-----|-------|
| `struct_thermo` | MFE/L, ensemble gap (MFE−ensemble FE), mean positional entropy | Same scalar triple broadcast to valid positions | 3 | on/off |
| `struct_dotbracket` | MFE dot-bracket | per-pos one-hot: `(` / `)` / `.` | 3 | on/off |
| `struct_bpp` | Partition function: pair probability, positional entropy | per-pos two channels | 2 | on/off |

Shared impl: `features/builders/_vienna.py` (`fold_sequence` + in-process LRU; multiple builders in the same batch do not re-fold).

## Listing and open status

- `features/registry.yaml`: `layer: L3`, `deps: [ViennaRNA]`, `cost: medium`, `agent_tunable: []` (tuning surface is recipe `enabled`) 
- `features/active.yaml`: all three listed; defaults are mostly `false`, **agent may turn them on** 
- Agent **must not** install packages; deps must already be preinstalled in the env 

## Overnight usage

In a feature round, edit `enabled` directly, e.g.:

- `[feature] enable struct_thermo`
- `[feature] enable struct_bpp; keep thermo off`
- `[feature] enable struct_dotbracket + struct_thermo`

After enabling, `input_dim` += the corresponding `dim`. `train.py` must read `data["input_dim"]`.

Example recipe: `examples/active.with_struct_thermo.yaml`

## Acceptance quick check (human)

```bash
conda activate myenv3.11
# enable struct_* in features/active.yaml, then:
python prepare.py
python train.py
```

Full encoding is computed when `train.py` starts; L3 is slower than L0–L2—watch startup wall-clock (encoding is not counted in `TIME_BUDGET`, but still affects perceived latency).
