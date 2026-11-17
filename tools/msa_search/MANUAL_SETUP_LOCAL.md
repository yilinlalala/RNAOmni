# 本地 MSA：手动安装与下载清单（人控）

Agent **禁止**执行装包 / 下载。库与 `pyhmmer` 就绪后可用 `--backend local`。

生产库：**RNAcentral 官方 active** → `artifacts/msa_dbs/rnacentral_active/seqs.fasta`。

Online / local **共用** hits 目录；已有 `status=ok` 的序列默认跳过。

---

## A. 安装 PyHMMER

```bash
conda activate myenv3.11
conda install -c conda-forge pyhmmer
python -c "import pyhmmer; print(pyhmmer.__version__)"
```

## B. 下载 RNAcentral active FASTA

```bash
cd /path/to/RNAOmni_v1
mkdir -p artifacts/msa_dbs/rnacentral_active
cd artifacts/msa_dbs/rnacentral_active
wget -c https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_active.fasta.gz
gzip -dk rnacentral_active.fasta.gz
ln -sfn rnacentral_active.fasta seqs.fasta
# 验收（已在该目录时不要再套 artifacts/... 前缀）
ls -lh seqs.fasta
```

说明：https://www.rnacentral.org/help/ftp  

## C. 本地搜索用法

```bash
conda activate myenv3.11
cd /path/to/RNAOmni_v1

# 与 online 同一数据集目录；已搜过的 hash 会 skip
python tools/msa_search/run_precompute.py \
  --backend local --db-id rnacentral_active \
  --csv HEK_te_prediction_task.csv --stage all

# 仅补 hits / 仅提 profile
python tools/msa_search/run_precompute.py --backend local --db-id rnacentral_active --stage hits
python tools/msa_search/run_precompute.py --backend local --db-id rnacentral_active --stage profiles

# 强制重搜（慎用；首次会把 ~31G DB preload 进内存，需较大 RAM）
python tools/msa_search/run_precompute.py --backend local --db-id rnacentral_active --stage hits --limit 1 --force

# 省内存、更慢：每条 query 流式读盘
python tools/msa_search/run_precompute.py --backend local --db-id rnacentral_active --no-preload-db --cpus 8
```

小库冒烟（可选）：`artifacts/msa_dbs/smoke_tiny/` + `--db-id smoke_tiny`。

## D. 布局

```text
artifacts/msa_dbs/<db_id>/seqs.fasta     # 搜库（跨数据集共用）
artifacts/msa/<dataset_id>/hits|profiles # 结果（按 CSV；online/local 共用）
```
