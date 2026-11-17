"""Build query-referenced column counts from RNAcentral pairwise alignments."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from tools.msa_search.contract import DIM, normalize_rna

_BASE_IDX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
_SEQ_LINE = re.compile(
    r"^(?:Query|Sbjct)\s+(\d+)\s+([A-Za-z.\-]+)\s+(\d+)\s*$"
)


def _parse_seq_line(line: str) -> Optional[Tuple[int, str, int]]:
    m = _SEQ_LINE.match(line.strip())
    if not m:
        return None
    return int(m.group(1)), m.group(2), int(m.group(3))


def iter_alignment_blocks(alignment: str) -> Iterable[Tuple[str, str, int]]:
    """Yield (query_aln, sbjct_aln, query_start_1based) for each BLAST block."""
    lines = (alignment or "").splitlines()
    i = 0
    while i < len(lines):
        q = _parse_seq_line(lines[i]) if lines[i].lstrip().startswith("Query") else None
        if q is None:
            i += 1
            continue
        q_start, q_aln, _q_end = q
        i += 1
        while i < len(lines) and not lines[i].lstrip().startswith("Sbjct"):
            i += 1
        if i >= len(lines):
            break
        t = _parse_seq_line(lines[i])
        i += 1
        if t is None:
            continue
        _t_start, t_aln, _t_end = t
        if len(q_aln) != len(t_aln):
            # Truncate to common length rather than dropping the hit.
            n = min(len(q_aln), len(t_aln))
            q_aln, t_aln = q_aln[:n], t_aln[:n]
        yield q_aln, t_aln, q_start


def accumulate_hit(
    base_counts: np.ndarray,
    gap_counts: np.ndarray,
    query_len: int,
    alignment: str,
) -> int:
    """Map one hit onto query coords. Returns number of query columns updated."""
    updated = 0
    for q_aln, t_aln, q_start in iter_alignment_blocks(alignment):
        q_pos = q_start - 1  # 0-based
        for qc, tc in zip(q_aln, t_aln):
            qc_u = qc.upper()
            if qc_u == "-" or qc_u == ".":
                continue
            if q_pos < 0 or q_pos >= query_len:
                q_pos += 1
                continue
            tc_u = tc.upper()
            if tc_u in _BASE_IDX:
                base_counts[q_pos, _BASE_IDX[tc_u]] += 1.0
            else:
                # gap or unknown → treat as gap for gap_freq
                gap_counts[q_pos] += 1.0
            updated += 1
            q_pos += 1
    return updated


def select_top_hits(
    results: Sequence[Dict[str, Any]],
    *,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Prefer higher score, then lower e_value."""

    def key(e: Dict[str, Any]) -> Tuple[float, float]:
        score = float(e.get("score") or 0.0)
        ev = e.get("e_value")
        try:
            e_val = float(ev) if ev is not None else 1e9
        except (TypeError, ValueError):
            e_val = 1e9
        return (-score, e_val)

    ranked = sorted(results, key=key)
    if top_k > 0:
        ranked = ranked[:top_k]
    return ranked


def counts_from_hit_record(
    record: Dict[str, Any],
    *,
    top_k: int = 200,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Returns base_counts (L,4), gap_counts (L,), meta.

    Query sequence is always included as one observation row.
    """
    seq = normalize_rna(record.get("sequence") or "")
    L = len(seq)
    base_counts = np.zeros((L, 4), dtype=np.float64)
    gap_counts = np.zeros((L,), dtype=np.float64)
    for i, ch in enumerate(seq):
        if ch in _BASE_IDX:
            base_counts[i, _BASE_IDX[ch]] += 1.0

    results = record.get("results") or []
    selected = select_top_hits(results, top_k=top_k)
    n_used = 0
    for hit in selected:
        aln = hit.get("alignment") or ""
        if not aln:
            continue
        if accumulate_hit(base_counts, gap_counts, L, aln) > 0:
            n_used += 1

    meta = {
        "L": L,
        "n_hits_total": int(record.get("hits") or len(results)),
        "n_hits_used": n_used,
        "top_k": top_k,
    }
    return base_counts, gap_counts, meta


def profile_from_counts(
    base_counts: np.ndarray,
    gap_counts: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """(L, DIM) float32 profile + n_eff estimate (mean non-gap depth)."""
    L = base_counts.shape[0]
    out = np.zeros((L, DIM), dtype=np.float32)
    n_base = base_counts.sum(axis=1)
    n_tot = n_base + gap_counts
    # freqs
    with np.errstate(divide="ignore", invalid="ignore"):
        freqs = np.divide(
            base_counts,
            n_base[:, None],
            out=np.zeros_like(base_counts),
            where=n_base[:, None] > 0,
        )
        gap_freq = np.divide(
            gap_counts,
            n_tot,
            out=np.zeros_like(gap_counts),
            where=n_tot > 0,
        )
    out[:, 0:4] = freqs.astype(np.float32)
    out[:, 5] = gap_freq.astype(np.float32)

    # conservation = 1 - H / log(4) over non-gap freqs
    log4 = math.log(4.0)
    cons = np.zeros(L, dtype=np.float64)
    for i in range(L):
        if n_base[i] <= 0:
            continue
        h = 0.0
        for p in freqs[i]:
            if p > 0:
                h -= float(p) * math.log(float(p))
        cons[i] = max(0.0, 1.0 - h / log4)
    out[:, 4] = cons.astype(np.float32)

    n_eff = float(n_base.mean()) if L else 0.0
    return out, n_eff


def profile_from_hit_record(
    record: Dict[str, Any],
    *,
    top_k: int = 200,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    base_counts, gap_counts, meta = counts_from_hit_record(record, top_k=top_k)
    if base_counts.shape[0] == 0:
        empty = np.zeros((0, DIM), dtype=np.float32)
        meta["n_eff"] = 0.0
        return empty, meta
    profile, n_eff = profile_from_counts(base_counts, gap_counts)
    meta["n_eff"] = n_eff
    return profile, meta
