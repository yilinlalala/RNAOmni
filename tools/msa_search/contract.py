"""P0 asset contract for MSA profile precompute.

Layout (dataset-scoped)::

    artifacts/msa/<dataset_id>/
      meta.json                 # csv_name, dataset_id, ...
      hits/by_hash/<sha1>.json
      hits/index.tsv
      profiles/by_hash/<sha1>.npy
      profiles/index.tsv

dataset_id = Path(csv).stem  (e.g. HEK_te_prediction_task)
Must match prepare.ACTIVE_CSV stem when training.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final, Optional, Tuple, Union

CACHE_KEY: Final[str] = "msa_profile_v1"
DIM: Final[int] = 6
MISSING_STRATEGY: Final[str] = "zeros"
MSA_ASSET_ROOT_NAME: Final[str] = "msa"

PROFILE_CHANNELS: Final[Tuple[str, ...]] = (
    "freq_A",
    "freq_C",
    "freq_G",
    "freq_U",
    "conservation",
    "gap_freq",
)

INDEX_HIT_HEADER: Final[str] = (
    "seq_hash\tstatus\tn_hits\telapsed_s\tbackend\tjob_id\tseq_len\trnacentral_md5"
)
INDEX_PROFILE_HEADER: Final[str] = (
    "seq_hash\tstatus\tn_hits\tn_eff\tL\tpath\tbackend\tcache_key"
)


def normalize_rna(seq: str) -> str:
    return "".join(str(seq).upper().replace("T", "U").split())


def normalize_dna_for_rnacentral(seq: str) -> str:
    return "".join(str(seq).upper().replace("U", "T").split())


def seq_hash(seq: str) -> str:
    return hashlib.sha1(normalize_rna(seq).encode("utf-8")).hexdigest()


def rnacentral_md5(seq: str) -> str:
    return hashlib.md5(normalize_dna_for_rnacentral(seq).encode("utf-8")).hexdigest()


def dataset_id_from_csv(csv_path: Union[str, Path]) -> str:
    """Stable id = CSV filename stem (HEK_te_prediction_task.csv → …)."""
    return Path(csv_path).stem


def resolve_dataset_asset_dir(
    *,
    repo_root: Path,
    dataset_id: str,
    asset_root: Optional[Union[str, Path]] = None,
) -> Path:
    """artifacts/msa/<dataset_id> (or custom asset_root/<dataset_id>)."""
    if asset_root is None:
        root = repo_root / "artifacts" / MSA_ASSET_ROOT_NAME
    else:
        root = Path(asset_root)
        if not root.is_absolute():
            root = (repo_root / root).resolve()
        else:
            root = root.resolve()
    return (root / dataset_id).resolve()


def file_sha256(path: Path, *, max_bytes: Optional[int] = None) -> str:
    """Content fingerprint for meta (optional size cap for huge files)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        if max_bytes is None:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        else:
            remaining = max_bytes
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                h.update(chunk)
                remaining -= len(chunk)
    return h.hexdigest()
