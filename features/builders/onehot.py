"""L0 one-hot ACGU channels."""

from __future__ import annotations

import numpy as np

DIM = 4


def _normalize_base(ch: str) -> str:
    base = ch.upper()
    return "U" if base == "T" else base


def build(sequences, *, max_len: int, alphabet: str = "ACGU", params=None) -> np.ndarray:
    """Return (N, max_len, 4) float32 one-hot encodings."""
    del params
    tokens = tuple(alphabet)
    token_to_idx = {tok: i for i, tok in enumerate(tokens)}
    n = len(sequences)
    out = np.zeros((n, max_len, len(tokens)), dtype=np.float32)
    for i, seq in enumerate(sequences):
        seq = seq[:max_len]
        for j, ch in enumerate(seq):
            idx = token_to_idx.get(_normalize_base(ch))
            if idx is not None:
                out[i, j, idx] = 1.0
    return out
