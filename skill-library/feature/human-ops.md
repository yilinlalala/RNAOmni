# Feature human-ops (human-controlled listing)

> Orchestration lives in `prepare.py` / `rnaomni/prepare/` / `features/pipeline.py`; the agent does not change these files, nor `rnaomni.env`.  
> Features are **computed on the fly** from the recipe when `train` / `load_prepared_data` runs; no on-disk feature cache is written.  
> Stage status: [`../../devlog/20260719_000000_feature_skill_PLAN_PARTIAL.md`](../../devlog/20260719_000000_feature_skill_PLAN_PARTIAL.md).

## Listing checklist (hand-crafted / L3)

1. Choose the feature; record license, VRAM, and runtime.  
2. Separate **locked parameters** from agent-tunable ones (on the S1 recipe side, effectively only `enabled`).  
3. Implement `features/builders/<name>.py` (`build(...) -> (N,L,C)`).  
4. Write `features/registry.yaml` (`dim`, `module`, `cache_key`, `default_params`, `cost`, `deps`).  
5. Add an entry in `features/active.yaml` (default `enabled: false`).  
6. `python prepare.py` (fold split + smoke encode); confirm printed `input_dim`.  
7. Manually run `train.py` through.  
8. Document (write examples / encoding notes if needed) before opening to overnight.

L3 structure: [structure-encoding.md](structure-encoding.md); smoke: `python tools/smoke_struct_builders.py`.

## L4 pretrained

Prefer **Deploy** ([deploy-ops.md](deploy-ops.md) + `python -m rnaomni.deploy` + `examples/<name>.assets.yaml`):

1. Human pre-places local weights (Deploy/agent must not auto-download).  
2. Deploy: embedding smoke → builder + registry + recipe placeholder (`enabled: false`).  
3. **Call:** enable, train through, measure full-run time/VRAM, write `examples/<name>.handoff.md`.  
4. Only after handoff marks C4 should overnight be encouraged to toggle `enabled`.

Done: `rna_fm`, `rinalmo`, `evo`, `evo2` (see matching handoffs).

## MSA

Human-controlled precompute assets + builder; listing requires **registry and recipe in sync**. Status: [msa-profile-encoding.md](msa-profile-encoding.md) and SSOT §8.3 (incomplete: builder/recipe present; registry/full assets missing).

## S1 acceptance B (historical)

- Builder: `length_frac` (dim=1)  
- After enabling vs L0-only: `input_dim` 4→5  

Example: `examples/active.with_length_frac.yaml`
