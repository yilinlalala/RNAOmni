# MSA search precompute

Online and local share **one directory per dataset**:

```text
artifacts/msa/<dataset_id>/hits|profiles/   # search once; skip if ok
artifacts/msa_dbs/<db_id>/                  # local sequence DB only
```

## Commands

```bash
# online
python tools/msa_search/run_precompute.py --backend online --csv HEK_te_prediction_task.csv --stage all --limit 1

# local (PyHMMER + RNAcentral active)
python tools/msa_search/run_precompute.py \
  --backend local --db-id rnacentral_active \
  --csv HEK_te_prediction_task.csv --stage all
```

Setup: **[MANUAL_SETUP_LOCAL.md](MANUAL_SETUP_LOCAL.md)**.
