"""S1 acceptance channel B: per-position length fraction (dim=1).

At valid positions j < len(seq), value = min(len, max_len) / max_len; padding stays 0.
Used to verify recipe toggling changes input_dim — not a strong biological feature.
"""

from __future__ import annotations

import numpy as np

DIM = 1


def build(sequences, *, max_len: int, alphabet: str = "ACGU", params=None) -> np.ndarray:
    """Return (N, max_len, 1) float32."""
    del alphabet, params
    n = len(sequences)
    out = np.zeros((n, max_len, 1), dtype=np.float32)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        if length <= 0:
            continue
        frac = length / float(max_len)
        out[i, :length, 0] = frac
    return out
