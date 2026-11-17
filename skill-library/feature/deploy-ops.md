# Feature deploy-ops (Deploy Agent: L4 deploy + register)> **Human-triggered** (Cursor chat or cursor_sdk **standalone script**). **Not via** `orchestrator.py`. 
> **Forbidden** to inject overnight / `PROMPT_SETUP` / `PROMPT_ITER`. Overnight procedure: [agent-ops.md](agent-ops.md). 
> **Asset-driven:** `examples/<feature>.assets.yaml` + CLI `--code-root` / `--weight` (few hardcodes); resolution in `tools/pretrained_assets.py` (local paths only).

## Boundary vs Overnight / Call

| Role | Doc | What it does | Status (2026-07-25) |
|------|------|--------|-------------------|
| **Deploy** (this doc) | `deploy-ops.md` | Local pretrained → embedding smoke → builder + registry + recipe placeholder (`enabled: false`) + handoff draft | ✅ `rna_fm` / `rinalmo` / `evo` / `evo2` |
| **Call** | Short human run + update handoff | Turn on `enabled`, train through, full-run time/VRAM, decide overnight open (C4) | ✅ All four models Called; see `examples/*.handoff.md` |
| **Overnight** | `agent-ops.md` | Only recipe whitelist + `train.py`; raise val_spearman | May toggle Called L4 |

Deploy **done** = feature registered with default `enabled: false`. 
Deploy **does not** = turn on `enabled`, full training, change overnight prompts (those belong to Call / Overnight).

## Trigger (parameterized one-shot)

```bash
conda activate myenv3.11
cd /path/to/RNAOmni_v1
export CURSOR_API_KEY=... # required unless --dry-run

# RiNALMo (default assets; weights must already be under README weights/)
python -m rnaomni.deploy \
 --assets skill-library/feature/examples/rinalmo.assets.yaml \
 --code-root tools/RiNALMo \
 --dry-run

python -m rnaomni.deploy \
 --assets skill-library/feature/examples/rinalmo.assets.yaml \
 --code-root tools/RiNALMo

# RNA-FM (explicit assets)
python -m rnaomni.deploy \
 --assets skill-library/feature/examples/rna_fm.assets.yaml \
 --dry-run

# Optional: lock a local weight (skip candidates resolution)
python -m rnaomni.deploy --assets .../rinalmo.assets.yaml \
 --code-root /abs/RiNALMo --weight /abs/RiNALMo/weights/rinalmo_giga_pretrained.pt
```

- **Not via** `orchestrator.py`; overnight prompts unchanged. 
- Shared resolution: `tools/pretrained_assets.py` (local-first; downloads forbidden).

## D1 assets: RiNALMo (second model; default micro)

Authoritative list: [`examples/rinalmo.assets.yaml`](examples/rinalmo.assets.yaml). Summary:

| Key | Convention |
|----|------|
| `feature_name` / `module` | `rinalmo` / `features.builders.rinalmo` |
| `conda_env` | `myenv3.11` |
| `code_root` | `tools/RiNALMo`; CLI `--code-root` overrides |
| `package_dir` | `rinalmo` |
| Weight location | `{code_root}/weights/` (official README) |
| Default resolution | candidates pick first existing file in order **micro → mega → giga** |
| `expected_dim_doc` | 480 (micro); mega=640 / giga=1280 |
| recipe | `enabled: false` |

Human pre-places weights (example downloads only the smallest):

```bash
CODE_ROOT=tools/RiNALMo
mkdir -p "$CODE_ROOT/weights"
# wget .../rinalmo_micro_pretrained.pt -O "$CODE_ROOT/weights/rinalmo_micro_pretrained.pt"
```

Smoke:

```bash
PYTHONPATH="$CODE_ROOT" python tools/smoke_rinalmo.py --code-root "$CODE_ROOT"
python tools/smoke_rinalmo_builder.py --feature-name rinalmo
```

Handoff: [`examples/rinalmo.handoff.md`](examples/rinalmo.handoff.md).

## D1 assets: RNA-FM (first model)

Authoritative list: [`examples/rna_fm.assets.yaml`](examples/rna_fm.assets.yaml). Summary:

| Key | Frozen value |
|----|--------|
| `feature_name` / `module` | `rna_fm` / `features.builders.rna_fm` |
| `conda_env` | `myenv3.11` |
| `code_root` | `tools/RNA-FM` (`PYTHONPATH=<repo>/tools/RNA-FM`) |
| `expected_dim_doc` | `640` (doc hint; **registry.dim follows D3 smoke**) |
| `repr_layer` | `12` |
| Weight candidates (first existing wins) | `tools/RNA-FM/pretrained/RNA-FM_pretrained.pth` |
| recipe placeholder | `name: rna_fm`, `enabled: false` |

Place weights in the upstream-agreed directory (README: `.pth` under `pretrained/`):

```bash
mkdir -p tools/RNA-FM/pretrained
cp RNA-FM_pretrained.pth tools/RNA-FM/pretrained/RNA-FM_pretrained.pth
```

On register, write the **resolved absolute path** into `default_params.model_path`. 
Parallel worktrees: `parallel_lanes.provision_batch` symlinks these two directories into the main repo (same approach as `dataset/forcompare`); overnight **forbids** wget / hub downloads. Smoke results stay under `artifacts/pretrained/` (ignored).

## Install and dependency policy

1. **Default zero-install:** `PYTHONPATH=<code_root>`; do not `pip install` the whole package; do not rebuild the env from the official `environment.yml`. 
2. Target is **Embedding** only; RiNALMo defaults to `use_flash_attn=false` (attention already lazily loads flash_attn). 
3. **Missing libraries:** stop and talk to the user; may compare `{code_root}/environment.yml` / `setup.py`; do not install packages online on your own. 
4. **Version skew:** if smoke passes, ignore official pins; if it fails, negotiate Plan B. 
5. Weights: **local only**; forbid gdown / hub / Zenodo auto-download.

## Red lines

- Forbid networked package installs without `--no-index`; forbid weight downloads without a local path. 
- Do not change `orchestrator.py`, `PROMPT_*`, folds / metric / `TIME_BUDGET`. 
- New recipe entries must be **`enabled: false`**. 
- Missing libraries must ask a human (not unattended overnight).

## Writable paths (Deploy whitelist, per feature_name)

| Path | Action |
|------|------|
| `features/builders/<feature>.py` | Create / modify |
| `features/registry.yaml` | Add L4 fields per [registry-spec.md](registry-spec.md) |
| `features/active.yaml` | Add placeholder row only, `enabled: false` |
| `artifacts/pretrained/<feature>/` | Smoke results (do not commit large weights) |
| `skill-library/feature/examples/<feature>.handoff.md` | D5 handoff |
| Notes in this skill doc | Optional |

**Not writable (this vertical slice):** `train.py`, `prepare.py`, `rnaomni/prepare/`, evaluation and fold contracts, overnight prompts.

## Execution checklist (D3–D5)

1. Read the matching `*.assets.yaml`; confirm `code_root` and one `weight_candidates` path exist. 
2. `conda activate myenv3.11`; run smoke (RiNALMo example):

```bash
python tools/smoke_rinalmo.py --code-root <CODE_ROOT>
# → artifacts/pretrained/rinalmo/smoke_result.json
```

3. **D3:** Record true `dim` / device / `model_path` (registry.dim follows smoke). 
4. **D4:** builder + registry + recipe `enabled: false`; `python tools/smoke_rinalmo_builder.py`. 
5. **D5:** Write `examples/<feature>.handoff.md` (registered, default `enabled: false` → hand to Call to update C2–C4).

### D3 zero-install notes

| Model | Handling |
|------|------|
| RNA-FM | stub `ptflops`; `torch.load(..., weights_only=False)` |
| RiNALMo | Do not use `get_pretrained_model`; `model_config(lm)+load_state_dict`; `use_flash_attn=false` |

## Acceptance (Deploy)

- [ ] registry resolves to `features.builders.<feature>` 
- [ ] builder smoke `C == registry.dim`, no NaN 
- [ ] `default_params.model_path` (or auto-resolve) points to an existing local file 
- [ ] Default `enabled: false` 
- [ ] **Does not** require a successful `train.py` run 
- [ ] D5 handoff: `examples/<feature>.handoff.md`
