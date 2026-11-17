"""L3 per-position MFE secondary-structure one-hot (dot-bracket).

Channels (dim=3): '(', ')', '.' from ViennaRNA MFE structure.
Padding and unknown symbols stay 0.

Requires ViennaRNA (`import RNA`). Default recipe: enabled false.
"""

from __future__ import annotations

import numpy as np

from features.builders._vienna import fold_sequence

DIM = 3
_CHAR_TO_IDX = {"(": 0, ")": 1, ".": 2}


def build(sequences, *, max_len: int, alphabet: str = "ACGU", params=None) -> np.ndarray:
    """Return (N, max_len, 3) float32 one-hot over MFE dot-bracket."""
    del alphabet, params
    n = len(sequences)
    out = np.zeros((n, max_len, DIM), dtype=np.float32)
    for i, seq in enumerate(sequences):
        seq = seq[:max_len]
        if not seq:
            continue
        structure = fold_sequence(seq).structure
        for j, ch in enumerate(structure[:max_len]):
            idx = _CHAR_TO_IDX.get(ch)
            if idx is not None:
                out[i, j, idx] = 1.0
    return out
