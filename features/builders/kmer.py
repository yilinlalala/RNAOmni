"""L2 position-wise k-mer one-hot channels.

At valid start index j (j + k <= len(seq) and j + k <= max_len), channel
index encodes seq[j:j+k] over the alphabet (lex order). Padding and
truncated tails stay 0.
"""

from __future__ import annotations

import numpy as np

DEFAULT_K = 3
K_MIN = 1
K_MAX = 4
DIM = 4**DEFAULT_K  # ACGU^3 = 64; recipe may overlay k in 1..4 (C=4^k)


def _normalize_base(ch: str) -> str:
    base = ch.upper()
    return "U" if base == "T" else base


def _dim_for_k(alphabet_size: int, k: int) -> int:
    return int(alphabet_size**k)


def build(sequences, *, max_len: int, alphabet: str = "ACGU", params=None) -> np.ndarray:
    """Return (N, max_len, |A|^k) float32 k-mer one-hot encodings."""
    params = params or {}
    k = int(params.get("k", DEFAULT_K))
    if k < K_MIN or k > K_MAX:
        raise ValueError(f"kmer k must be in {K_MIN}..{K_MAX}, got {k}")

    tokens = tuple(alphabet)
    n_tok = len(tokens)
    token_to_idx = {tok: i for i, tok in enumerate(tokens)}
    dim = _dim_for_k(n_tok, k)
    n = len(sequences)
    out = np.zeros((n, max_len, dim), dtype=np.float32)

    for i, seq in enumerate(sequences):
        seq = seq[:max_len]
        limit = len(seq) - k + 1
        if limit <= 0:
            continue
        for j in range(limit):
            idx = 0
            ok = True
            for t in range(k):
                base_i = token_to_idx.get(_normalize_base(seq[j + t]))
                if base_i is None:
                    ok = False
                    break
                idx = idx * n_tok + base_i
            if ok:
                out[i, j, idx] = 1.0
    return out
