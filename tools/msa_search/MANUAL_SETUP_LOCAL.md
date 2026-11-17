# Local MSA: manual install and download checklist (human-controlled)

Agents **must not** install packages or download. Once the DB and `pyhmmer` are ready, use `--backend local`.

Production DB: **RNAcentral official active** → `artifacts/msa_dbs/rnacentral_active/seqs.fasta`.

Online / local **share** the hits directory; sequences with `status=ok` are skipped by default.

---

## A. Install PyHMMER

```bash
conda activate myenv3.11
conda install -c conda-forge pyhmmer
python -c "import pyhmmer; print(pyhmmer.__version__)"
```

## B. Download RNAcentral active FASTA

```bash
cd /path/to/RNAOmni_v1
mkdir -p artifacts/msa_dbs/rnacentral_active
cd artifacts/msa_dbs/rnacentral_active
wget -c https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_active.fasta.gz
gzip -dk rnacentral_active.fasta.gz
ln -sfn rnacentral_active.fasta seqs.fasta
# Verify (when already in this directory, do not prefix artifacts/...)
ls -lh seqs.fasta
```

Notes: https://www.rnacentral.org/help/ftp  

## C. Local search usage

```bash
conda activate myenv3.11
cd /path/to/RNAOmni_v1

# Same dataset dir as online; already-searched hashes are skipped
python tools/msa_search/run_precompute.py \
  --backend local --db-id rnacentral_active \
  --csv HEK_te_prediction_task.csv --stage all

# Hits only / profiles only
python tools/msa_search/run_precompute.py --backend local --db-id rnacentral_active --stage hits
python tools/msa_search/run_precompute.py --backend local --db-id rnacentral_active --stage profiles

# Force re-search (use carefully; first run preloads ~31G DB into memory — needs large RAM)
python tools/msa_search/run_precompute.py --backend local --db-id rnacentral_active --stage hits --limit 1 --force

# Lower memory, slower: stream DB from disk per query
python tools/msa_search/run_precompute.py --backend local --db-id rnacentral_active --no-preload-db --cpus 8
```

Tiny-DB smoke (optional): `artifacts/msa_dbs/smoke_tiny/` + `--db-id smoke_tiny`.

## D. Layout

```text
artifacts/msa_dbs/<db_id>/seqs.fasta     # search DB (shared across datasets)
artifacts/msa/<dataset_id>/hits|profiles # results (per CSV; shared by online/local)
```
