"""Shared ViennaRNA fold helpers for L3 structure descriptors.

Requires the `RNA` package (ViennaRNA Python bindings). CLI `RNAfold` is not needed.
Process-local LRU cache avoids double-folding when multiple struct builders run
in the same encode pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import numpy as np

_CACHE_SIZE = 32768


@dataclass(frozen=True)
class ViennaResult:
    structure: str
    mfe: float
    efe: float
    pair_prob: Tuple[float, ...]
    positional_entropy: Tuple[float, ...]

    @property
    def length(self) -> int:
        return len(self.structure)

    @property
    def mfe_per_nt(self) -> float:
        n = self.length
        return float(self.mfe / n) if n else 0.0

    @property
    def ensemble_gap(self) -> float:
        """mfe - ensemble free energy (>=0 when ensemble is lower than MFE)."""
        return float(self.mfe - self.efe)

    @property
    def mean_positional_entropy(self) -> float:
        if not self.positional_entropy:
            return 0.0
        return float(np.mean(self.positional_entropy))


def _normalize_seq(seq: str) -> str:
    return "".join("U" if c.upper() == "T" else c.upper() for c in seq)


def _import_rna():
    try:
        import RNA  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "ViennaRNA Python bindings required for L3 structure features "
            "(conda/pip package providing `import RNA`). Env: rnaomni."
        ) from exc
    return RNA


@lru_cache(maxsize=_CACHE_SIZE)
def fold_sequence(seq: str) -> ViennaResult:
    """MFE + partition function metrics for one RNA sequence (cached)."""
    RNA = _import_rna()
    seq = _normalize_seq(seq)
    if not seq:
        return ViennaResult("", 0.0, 0.0, (), ())

    fc = RNA.fold_compound(seq)
    structure, mfe = fc.mfe()
    _ss, efe = fc.pf()
    bpp = fc.bpp()
    pe_raw = fc.positional_entropy()

    n = len(seq)
    pair_prob = []
    pos_ent = []
    for i in range(1, n + 1):
        p = 0.0
        for j in range(1, n + 1):
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            p += float(bpp[a][b])
        pair_prob.append(p)
        # pe_raw[0] is unused / length-like; values are 1-indexed.
        pos_ent.append(float(pe_raw[i]) if i < len(pe_raw) else 0.0)

    return ViennaResult(
        structure=structure,
        mfe=float(mfe),
        efe=float(efe),
        pair_prob=tuple(pair_prob),
        positional_entropy=tuple(pos_ent),
    )


def clear_fold_cache() -> None:
    fold_sequence.cache_clear()
