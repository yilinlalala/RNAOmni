"""L3 global thermodynamic descriptors (broadcast to valid positions).

Channels (dim=3), filled at j < len(seq):
  0: MFE / L (kcal/mol per nucleotide)
  1: ensemble gap = MFE - ensemble FE (>=0 → more flexible ensemble)
  2: mean positional entropy from partition function

Requires ViennaRNA (`import RNA`). Default recipe: enabled false.
"""

from __future__ import annotations

import numpy as np

from features.builders._vienna import fold_sequence

DIM = 3


def build(sequences, *, max_len: int, alphabet: str = "ACGU", params=None) -> np.ndarray:
    """Return (N, max_len, 3) float32; padding stays 0."""
    del alphabet, params
    n = len(sequences)
    out = np.zeros((n, max_len, DIM), dtype=np.float32)
    for i, seq in enumerate(sequences):
        seq = seq[:max_len]
        if not seq:
            continue
        res = fold_sequence(seq)
        vals = (res.mfe_per_nt, res.ensemble_gap, res.mean_positional_entropy)
        length = min(len(seq), max_len)
        out[i, :length, :] = vals
    return out
