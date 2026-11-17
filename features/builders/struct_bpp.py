"""L3 per-position ensemble structure descriptors from partition function.

Channels (dim=2):
  0: pairing probability (sum of base-pair probs involving the site)
  1: positional entropy (ViennaRNA positional_entropy)

Requires ViennaRNA (`import RNA`). Default recipe: enabled false.
"""

from __future__ import annotations

import numpy as np

from features.builders._vienna import fold_sequence

DIM = 2


def build(sequences, *, max_len: int, alphabet: str = "ACGU", params=None) -> np.ndarray:
    """Return (N, max_len, 2) float32; padding stays 0."""
    del alphabet, params
    n = len(sequences)
    out = np.zeros((n, max_len, DIM), dtype=np.float32)
    for i, seq in enumerate(sequences):
        seq = seq[:max_len]
        if not seq:
            continue
        res = fold_sequence(seq)
        length = min(len(seq), max_len, len(res.pair_prob))
        if length <= 0:
            continue
        out[i, :length, 0] = res.pair_prob[:length]
        out[i, :length, 1] = res.positional_entropy[:length]
    return out
