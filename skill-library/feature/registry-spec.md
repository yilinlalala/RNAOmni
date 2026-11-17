# Registry / recipe field conventions (S1 frozen)

> Decision: recipe **minimal whitelist**; acceptance **B** (`length_frac` second channel); `prepare.py` / `rnaomni/prepare/` orchestration is human-only.  
> Encoding: train / `load_prepared_data` **compute on the fly** (no on-disk feature cache / fingerprint).  
> Stage status: [`../../devlog/20260719_000000_feature_skill_PLAN_PARTIAL.md`](../../devlog/20260719_000000_feature_skill_PLAN_PARTIAL.md).

## Files

| File | Maintainer | Notes |
|------|--------|------|
| `features/registry.yaml` | Human (or L4 Deploy) | Listed feature catalog |
| `features/builders/*` | Human (or L4 Deploy) | Feature implementations |
| `features/pipeline.py` | Human | Load registry/recipe and concatenate |
| `features/active.yaml` | Human + Agent (whitelist keys) | Current enabled config |
| `skill-library/registry.yaml` | Human | **Skill** registry (a separate concern) |

## Recipe whitelist (S1 minimal set + kmer k / fusion / subset fusion)

Agent may edit **only**:

| Key | Constraints |
|----|------|
| `recipe_id` | String |
| `fusion` | `concat` (default) / `add` / `mean`. Other names fail to load. `add`/`mean` require the **included** channels to have the same C |
| `fusion_include` | Optional. Non-empty list; only names that are **enabled and registered**. Omit = all enabled. Contains disabled / unregistered / duplicates → load fails. Order is concat channel order |
| `features[].name` | Must already be in `features/registry.yaml` |
| `features[].enabled` | bool |
| `features[].params` | **Only** keys listed in that feature's `agent_tunable`. Current: kmer `k` (1–4; C=`4^k`); `rna_fm` `repr_layer` (0–12, default 12). `rinalmo` / `evo` / `evo2` remain `[]` (on/off only) |

Forbidden to change: builder code, `features/registry.yaml`, `default_params`, deps, weight paths.

### `features/active.yaml` shape

```yaml
recipe_id: s3_l0_length_frac_kmer
fusion: concat          # concat | add | mean
# fusion_include: [onehot, kmer]   # optional; omit = all enabled
features:
  - name: onehot
    enabled: true
  - name: length_frac
    enabled: true
  - name: kmer
    enabled: true
    params:
      k: 4              # optional; omit → registry default k=3 (64-d)
  - name: rna_fm
    enabled: false
    # params:
    #   repr_layer: 12   # optional; rna_fm only, 0–12
```

Acceptance B (historical): set `length_frac.enabled` to `true` → `input_dim` from 4→5.  
Default L0+L1+L2 concat, k=3 → `input_dim=69`. k=2 → kmer C=16; k=4 → 256. registry kmer `dim: 64` only documents default k=3; encode uses the computed C.  
`fusion_include: [onehot, kmer]` with k=3 → `input_dim=68` (does not include `length_frac`).

**Note:** A name present in the recipe but not in the registry with `enabled: true` → `KeyError`. MSA is ignored in this version; do not put it in the recipe.

## Feature registry entry

```yaml
name: onehot
layer: L0
dim: 4
cost: low
deps: []
cache_key: onehot_v1
module: features.builders.onehot
default_params: {}
agent_tunable: []
```

Notes: `agent_tunable` lists keys allowed in recipe `features[].params`. Currently open: kmer `[k]`, `rna_fm` `[repr_layer]`. `rinalmo` / `evo` / `evo2` and most entries remain `[]` (overnight only toggles `enabled`). Documented `agent_open` is **not** a required registry field.

Optional: `heavy: true` and/or `vram_gb` (≥12 treated as heavy). `evo` / `evo2` are marked. When the orchestrator is **full and a light L4 is already enabled** (`layer: L4` and not heavy; currently `rna_fm` / `rinalmo`), it must not MUST these heavy names.

Common extra L4 `default_params`: `model_path`, `device`, (RNA-FM) `repr_layer`, (RiNALMo) `lm_config` / `code_root`, etc., locked by human/Deploy. `repr_layer` is in `agent_tunable`; `lm_config` / weight paths remain locked.

## Builder interface

```python
def build(sequences, *, max_len: int, alphabet: str = "ACGU", params=None) -> np.ndarray:
    # returns (N, max_len, dim) float32
```

`dim` must match the builder output's last dimension. kmer registry `dim: 64` is the default k=3 documentation value; encode computes `4^k` from `params.k`—do not gate other k values on the static 64.

## Encoding conventions

`load_prepared_data` / `train.py` compute features from the current `features/active.yaml`; **no** on-disk cache, **no** fingerprint. In-process memo is limited to the same train process.

## `prepare.py` / `rnaomni/prepare/` boundary

| May change (human) | Must not change (human/agent contract) |
|------------|-------------------------|
| Read registry/recipe, assemble features (via pipeline) | folds, metric, `TIME_BUDGET`, `MAX_SEQ_LEN` contract values |
| | **Agent must not modify `prepare.py` or `rnaomni/prepare/` source** |
