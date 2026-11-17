# MSA profile encoding (dataset-bound)

> **Status (2026-07-25): incomplete.** builder + `tools/msa_search/` + recipe placeholder rows exist; **not yet** written to `features/registry.yaml` → turning on `enabled` raises `KeyError`. HEK full hits/profiles not covered. Local search DB as production default remains deferred. 
> Human-controlled precompute; agent **must not** toggle `msa_profile` for now. ## Dataset / dual backend

| Concept | Rule |
|------|------|
| `dataset_id` | `Path(csv).stem` |
| Asset path | `artifacts/msa/<dataset_id>/` (**shared by online and local**) |
| Skip | Existing `hits/.../<hash>.json` with `status=ok` → do not re-search (unless `--force`) |
| Local DB | `artifacts/msa_dbs/<db_id>/` (separate from dataset dirs) |
| Training read | builder defaults to stem of `prepare.ACTIVE_CSV` |

Local install and download: **[MANUAL_SETUP_LOCAL.md](../../tools/msa_search/MANUAL_SETUP_LOCAL.md)** (RNAcentral active + pyhmmer).

## Required before closing the loop

1. Register `msa_profile` in `features/registry.yaml` (dim=6, module, `default_params`). 
2. Human-run precompute for the current `ACTIVE_CSV` to completion (hits + profiles). 
3. Smoke-train through, then open overnight in handoff/procedure.

## Flow (after listing is complete)

```bash
python tools/msa_search/run_precompute.py --csv <active>.csv --stage all
# After registry registration: enable msa_profile in recipe → train (read-only npy for that dataset)
```

## Channels (dim=6)

`freq_A/C/G/U`, `conservation`, `gap_freq` (see `contract.PROFILE_CHANNELS`).

## Red lines

- Agent does not run precompute or change builder source. 
- Missing profile → builder-side `missing: zeros` (after listing loop is closed). 
- Different CSVs **must not** share the same `artifacts/msa/<id>/` (unless you confirm the sequence set is identical and intend that).
