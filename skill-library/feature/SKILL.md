---
name: rnaomni-feature
description: >-
  Human-controlled RNA feature pipeline for RNAOmni: register builders and
  pretrained encoders, edit features/active.yaml whitelist keys, adapt train.py
  fusion and input_dim. Use when working on features, recipes, INPUT_DIM,
  pretrained RNA embeddings, orchestrator.py overnight feature rounds,
  PROMPT_SETUP/PROMPT_ITER, or workspace/<stem>.tsv rows tagged [feature].
---

# RNAOmni Feature (human-controlled path H)

Agent procedure entry point. Registration: [`../registry.yaml`](../registry.yaml). Stage status: [`../../devlog/20260719_000000_feature_skill_PLAN_PARTIAL.md`](../../devlog/20260719_000000_feature_skill_PLAN_PARTIAL.md).

**Status (2026-08-22):** S0–S3 done; L3 structure and L4 (`rna_fm` / `rinalmo` / `evo` / `evo2`) Deploy+Call done; MSA is incomplete; next stage is S4 overnight acceptance.

## Path (mandatory)

- **Human:** implement `features/builders/*`, write `features/registry.yaml`, list pretrained models and lock parameters; for L4 use `python -m rnaomni.deploy`.  
- **Agent:** only edit `features/active.yaml` (whitelist keys) and `train.py`. No on-disk feature cache: the next `train.py` start encodes with the current recipe.  
  - **orchestrator overnight:** edit+commit only (steps 1–3), then STOP immediately; `train.py` / writing TSV / keep/discard reset are done by the orchestrator.  
  - **Interactive, without orchestrator:** the agent runs the full steps 1–6.  
- **Forbidden:** change builders / registry / `prepare.py` / `rnaomni/prepare/` / evaluation / folds / `TIME_BUDGET` / `rnaomni.env` / `.env` / install packages / download new weights / run MSA precompute.

## Details (open as needed)

| Role | Doc |
|------|------|
| Human listing (hand-crafted / structure / MSA) | [human-ops.md](human-ops.md) |
| L4 Deploy (human-triggered, not overnight) | [deploy-ops.md](deploy-ops.md) |
| One overnight round | [agent-ops.md](agent-ops.md) |
| Prediction-head frameworks | [frameworks.md](frameworks.md) (start with `baseline`; menu includes `tf_*` / `gru_pool`, etc.) |
| registry / recipe fields | [registry-spec.md](registry-spec.md) (whitelist: fusion `concat\|add\|mean`; optional `fusion_include`; kmer `k` 1–4; `rna_fm.repr_layer` 0–12) |
| L3 structure | [structure-encoding.md](structure-encoding.md) |
| MSA | [msa-profile-encoding.md](msa-profile-encoding.md) |

## Agent quick rules

1. At round start read `workspace/<stem>.tsv`, the current recipe, and the `program.md` contract.  
2. Declare `round_type`: `feature` | `model` (avoid `joint` in v1). If Iter injected `next_framework`, this round MUST be `model` and implement that name (see [frameworks.md](frameworks.md)). When the bottleneck card has `next_do=feature`, MUST follow. Start with `baseline`.  
3. One main hypothesis per round; description prefix `[feature]` or `[model]`. `[model]` must include `framework=<name>`; on the first shot after changing the backbone also add `framework_base cause=head`. Also write `cause=<head|feature|knob>` (use `cause=crash` when fixing a crash). Do not write `cause=head` when there is no `next_framework`.  
4. Commit only whitelist files. Overnight: STOP immediately after commit; do not run `train.py`, do not edit the TSV; discard `git reset` is done by the orchestrator. Only in interactive sessions does the agent train, write the TSV, and reset.  
5. Decision metrics: regression **val_spearman** (higher is better) / classification **val_accuracy** (higher is better); keep requires relative best ≥ min_delta (see Iter prompt / `agent-ops.md`).  
6. May only `enabled` names **already in** `features/registry.yaml`. MSA is ignored in this version.

## Layout

```text
features/registry.yaml      # human-maintained
features/builders/          # human-maintained (incl. L3/L4)
features/pipeline.py        # human-maintained; agent must not edit
features/active.yaml        # Agent may edit (minimal whitelist keys)
prepare.py                  # facade; Agent must not change source
rnaomni/prepare/            # orchestration impl; Agent must not change
rnaomni/                    # other impl (orch / results / archive / …)
train.py                    # Agent may edit; use data["input_dim"]
# L4 Deploy: python -m rnaomni.deploy
```

Recipe whitelist: [registry-spec.md](registry-spec.md). With default concat + kmer k=3, L0+L1+L2 gives `input_dim=69`; current `features/active.yaml` also enables some L3. Overnight may change: registered `enabled` (incl. L4 `rna_fm` / `rinalmo` / `evo` / `evo2`), kmer `params.k` (1–4), `rna_fm` `params.repr_layer` (0–12), `fusion` / `fusion_include`. `rinalmo` / `evo` / `evo2` are on/off only. Missing local L4 weights will crash; pip / downloads are forbidden.

## Examples

**Overnight feature round (orchestrator)**  
Read `features/active.yaml` → enable registered features → adapt `train.py` for `input_dim` if needed → commit → STOP (`train.py` / TSV logging / keep/discard are done by the orchestrator).

**Interactive feature round (without orchestrator)**  
Same as above, but the agent itself runs `train.py` → logs TSV → keep/discard.

**Human listing L4**  
Read the matching `examples/<name>.assets.yaml` → `python -m rnaomni.deploy` → Call / write `examples/<name>.handoff.md` → then open overnight `enabled`.
