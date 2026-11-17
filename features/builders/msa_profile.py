"""L3.5 MSA column profile from dataset-scoped precomputed assets.

Reads::

  artifacts/msa/<dataset_id>/profiles/by_hash/<sha1>.npy

dataset_id defaults to prepare.ACTIVE_CSV stem so switching the active dataset
picks the matching MSA folder. Override with default_params.dataset_id / asset_dir.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from tools.msa_search.contract import (
    CACHE_KEY,
    DIM as _CONTRACT_DIM,
    dataset_id_from_csv,
    normalize_rna,
    resolve_dataset_asset_dir,
    seq_hash,
)

DIM = int(_CONTRACT_DIM)

_REPO = Path(__file__).resolve().parents[2]
_loaded: Dict[str, np.ndarray] = {}


def _active_dataset_id() -> str:
    try:
        from prepare import ACTIVE_CSV

        return dataset_id_from_csv(ACTIVE_CSV)
    except Exception:  # noqa: BLE001
        return "HEK_te_prediction_task"


def _resolve_asset_dir(params) -> Path:
    params = params or {}
    # Explicit full directory wins (legacy / debug).
    raw_dir = params.get("asset_dir")
    if raw_dir:
        path = Path(os.path.expanduser(str(raw_dir)))
        if not path.is_absolute():
            path = (_REPO / path).resolve()
        else:
            path = path.resolve()
        return path

    dataset_id = params.get("dataset_id")
    if not dataset_id:
        dataset_id = _active_dataset_id()
    asset_root = params.get("asset_root", "artifacts/msa")
    return resolve_dataset_asset_dir(
        repo_root=_REPO,
        dataset_id=str(dataset_id),
        asset_root=asset_root,
    )


def _load_one(asset_dir: Path, h: str) -> Optional[np.ndarray]:
    key = f"{asset_dir}::{h}"
    if key in _loaded:
        return _loaded[key]
    path = asset_dir / "profiles" / "by_hash" / f"{h}.npy"
    if not path.is_file():
        return None
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != DIM:
        raise ValueError(
            f"msa_profile bad shape {arr.shape} at {path}; expected (L,{DIM})"
        )
    _loaded[key] = arr.astype(np.float32, copy=False)
    return _loaded[key]


def build(sequences, *, max_len: int, alphabet: str = "ACGU", params=None) -> np.ndarray:
    del alphabet
    params = params or {}
    asset_dir = _resolve_asset_dir(params)
    missing = str(params.get("missing", "zeros"))
    n = len(sequences)
    out = np.zeros((n, max_len, DIM), dtype=np.float32)
    for i, seq in enumerate(sequences):
        norm = normalize_rna(seq)
        h = seq_hash(norm)
        arr = _load_one(asset_dir, h)
        if arr is None:
            if missing == "raise":
                raise FileNotFoundError(
                    f"msa_profile missing hash={h} under {asset_dir} "
                    f"(cache_key={CACHE_KEY}); "
                    f"run: python tools/msa_search/run_precompute.py "
                    f"--csv <matching.csv> --stage all"
                )
            continue
        L = min(int(arr.shape[0]), max_len, len(norm))
        if L > 0:
            out[i, :L, :] = arr[:L]
    return out
