"""Within-class MMseqs easy-cluster helpers for classification splits."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MIN_SEQ_ID = 0.8
DEFAULT_COVERAGE = 0.8


def resolve_mmseqs_bin(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("MMSEQS_BIN")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    found = shutil.which("mmseqs")
    if found:
        return found
    fallback = Path.home() / "source" / "mmseqs" / "bin" / "mmseqs"
    if fallback.is_file() and os.access(fallback, os.X_OK):
        return str(fallback)
    raise FileNotFoundError(
        "mmseqs not found. Add it to PATH, set MMSEQS_BIN, or install under ~/source/mmseqs/bin/mmseqs"
    )


def _write_fasta(path: Path, seq_ids: list[str], sequences: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for sid, seq in zip(seq_ids, sequences):
            # MMseqs nucleotide alphabet prefers T over U
            dna = seq.upper().replace("U", "T")
            f.write(f">{sid}\n{dna}\n")


def _parse_cluster_tsv(path: Path) -> dict[str, str]:
    """Map member_id -> representative_id from mmseqs *_cluster.tsv."""
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            rep, member = parts[0], parts[1]
            mapping[member] = rep
    return mapping


def cluster_sequences(
    sequences: list[str],
    *,
    min_seq_id: float = DEFAULT_MIN_SEQ_ID,
    coverage: float = DEFAULT_COVERAGE,
    work_dir: str | Path | None = None,
    mmseqs_bin: str | None = None,
    threads: int | None = None,
    prefix: str = "q",
) -> list[str]:
    """
    Cluster sequences with MMseqs easy-cluster.

    Returns a cluster_id (representative id) per input sequence, same order.
    """
    if len(sequences) == 0:
        return []
    if len(sequences) == 1:
        return [f"{prefix}0"]

    mmseqs = resolve_mmseqs_bin(mmseqs_bin)
    n_threads = threads or max(1, min(8, (os.cpu_count() or 4)))

    own_tmp = work_dir is None
    root = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="mmseqs_cls_"))
    root.mkdir(parents=True, exist_ok=True)
    try:
        fasta = root / "query.fasta"
        seq_ids = [f"{prefix}{i}" for i in range(len(sequences))]
        _write_fasta(fasta, seq_ids, sequences)

        cluster_prefix = root / "clusterRes"
        tmp_dir = root / "tmp"
        tmp_dir.mkdir(exist_ok=True)

        cmd = [
            mmseqs,
            "easy-cluster",
            str(fasta),
            str(cluster_prefix),
            str(tmp_dir),
            "--min-seq-id",
            str(min_seq_id),
            "-c",
            str(coverage),
            "--cov-mode",
            "1",
            "--dbtype",
            "2",  # nucleotides
            "--linclust-version",
            "1",  # v2 segfaults on this mmseqs build for nucl clustering
            "--threads",
            str(n_threads),
        ]
        proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"mmseqs easy-cluster failed (exit {proc.returncode}):\n"
                f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
            )

        cluster_tsv = Path(str(cluster_prefix) + "_cluster.tsv")
        if not cluster_tsv.is_file():
            # some builds write clusterRes_cluster.tsv next to prefix
            alt = root / "clusterRes_cluster.tsv"
            if alt.is_file():
                cluster_tsv = alt
            else:
                raise FileNotFoundError(f"MMseqs cluster TSV not found under {root}")

        member_to_rep = _parse_cluster_tsv(cluster_tsv)
        out = []
        for sid in seq_ids:
            rep = member_to_rep.get(sid, sid)
            out.append(rep)
        return out
    finally:
        if own_tmp:
            shutil.rmtree(root, ignore_errors=True)


def assign_cluster_splits_per_class(
    df: pd.DataFrame,
    *,
    sequence_col: str,
    label_col: str,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
    test_frac: float = 0.1,
    min_seq_id: float = DEFAULT_MIN_SEQ_ID,
    coverage: float = DEFAULT_COVERAGE,
    random_state: int = 42,
    cache_path: str | Path | None = None,
    work_root: str | Path | None = None,
    mmseqs_bin: str | None = None,
) -> pd.DataFrame:
    """
    Within-class MMseqs clustering, then assign whole clusters to train/val/test.

    Adds columns: cluster_id, split (train|val|test).
    Caches cluster_id assignments when cache_path is set.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1")

    out = df.copy().reset_index(drop=True)
    n = len(out)

    if cache_path is not None and Path(cache_path).is_file():
        cached = pd.read_csv(cache_path)
        if len(cached) == n and "cluster_id" in cached.columns:
            # Align by row order written previously (same cleaned table).
            out["cluster_id"] = cached["cluster_id"].astype(str).to_numpy()
        else:
            cache_path = Path(cache_path)  # fall through to recompute
            cached = None
    else:
        cached = None

    if "cluster_id" not in out.columns:
        cluster_ids = np.empty(n, dtype=object)
        labels = out[label_col].to_numpy()
        unique_labels = sorted(pd.unique(labels).tolist())
        work_root = Path(work_root) if work_root is not None else Path(tempfile.mkdtemp(prefix="mmseqs_by_class_"))
        work_root.mkdir(parents=True, exist_ok=True)
        print(
            f"Data: MMseqs within-class clustering "
            f"(min_seq_id={min_seq_id}, classes={len(unique_labels)})..."
        )
        for lab in unique_labels:
            idx = np.where(labels == lab)[0]
            seqs = out.loc[idx, sequence_col].astype(str).tolist()
            class_dir = work_root / f"class_{lab}"
            class_dir.mkdir(exist_ok=True)
            reps = cluster_sequences(
                seqs,
                min_seq_id=min_seq_id,
                coverage=coverage,
                work_dir=class_dir,
                mmseqs_bin=mmseqs_bin,
                prefix=f"c{lab}_",
            )
            # Global cluster id: class + representative
            for local_i, row_i in enumerate(idx):
                cluster_ids[row_i] = f"{lab}|{reps[local_i]}"
            n_clust = len(set(reps))
            print(f"  class {lab}: n={len(idx)} clusters={n_clust}")
        out["cluster_id"] = cluster_ids
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            out[["cluster_id"]].to_csv(cache_path, index=False)
            print(f"Data: wrote cluster cache -> {cache_path}")

    rng = np.random.RandomState(random_state)
    split = np.full(n, "train", dtype=object)
    labels = out[label_col].to_numpy()
    cluster_ids = out["cluster_id"].astype(str).to_numpy()

    for lab in sorted(pd.unique(labels).tolist()):
        class_mask = labels == lab
        class_idx = np.where(class_mask)[0]
        clusters = list(dict.fromkeys(cluster_ids[class_idx].tolist()))  # preserve order, unique
        rng.shuffle(clusters)
        n_c = len(clusters)
        if n_c == 1:
            # Cannot separate homologs; keep class in train only and warn via print.
            print(f"  warn: class {lab} has 1 cluster — all rows assigned to train")
            continue
        if n_c == 2:
            split_map = {clusters[0]: "train", clusters[1]: "val"}
        else:
            n_test = max(1, int(round(n_c * test_frac)))
            n_val = max(1, int(round(n_c * val_frac)))
            if n_test + n_val >= n_c:
                n_test = 1
                n_val = 1
            n_train = n_c - n_test - n_val
            if n_train < 1:
                n_train = 1
                n_val = max(1, n_c - n_train - 1)
                n_test = n_c - n_train - n_val
            ordered = clusters
            split_map = {}
            i = 0
            for _ in range(n_train):
                split_map[ordered[i]] = "train"
                i += 1
            for _ in range(n_val):
                split_map[ordered[i]] = "val"
                i += 1
            for _ in range(n_test):
                split_map[ordered[i]] = "test"
                i += 1
        for row_i in class_idx:
            split[row_i] = split_map.get(cluster_ids[row_i], "train")

    out["split"] = split

    # Hard constraint: each cluster_id maps to exactly one split.
    check = (
        out.groupby("cluster_id", sort=False)["split"]
        .nunique()
    )
    bad = check[check > 1]
    if len(bad) > 0:
        raise RuntimeError(
            f"Data leakage: {len(bad)} clusters span multiple splits "
            f"(example: {bad.index[0]})"
        )
    return out
