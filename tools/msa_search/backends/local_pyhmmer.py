"""Local PyHMMER nhmmer backend against artifacts/msa_dbs/<db_id>/seqs.fasta.

Writes the same hit-record schema as online_rnacentral so extract_profile /
shared skip logic work unchanged.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from tools.msa_search.assets import make_hit_record
from tools.msa_search.contract import normalize_rna

BACKEND_NAME = "local_pyhmmer"
_DEFAULT_WIDTH = 90
# IUPAC nucleotide + DNA; anything else counts as "bad" (e.g. protein P/E/F…).
_DNA_IUPAC = set("ACGTURYSWKMBDHVN")


def resolve_local_fasta(db_root: Path, db_id: str) -> Path:
    base = Path(db_root) / db_id
    for name in ("seqs.fasta", "rnacentral_active.fasta", "db.fasta"):
        cand = base / name
        if cand.is_file() or cand.is_symlink():
            return cand.resolve()
    raise FileNotFoundError(
        f"No fasta under {base} (expected seqs.fasta). "
        f"See tools/msa_search/MANUAL_SETUP_LOCAL.md"
    )


def _to_dna(seq: str) -> str:
    return normalize_rna(seq).replace("U", "T")


def sanitize_dna_sequence(seq: str, *, max_bad_frac: float = 0.2) -> Optional[str]:
    """Keep nucleotide seqs; map ambiguous→N; skip likely protein / garbage.

    RNAcentral active can contain non-ACGT letters (e.g. 'P'); pyhmmer DNA
    alphabet raises ValueError on those if read with digital=True.
    """
    s = "".join(str(seq).upper().replace("U", "T").split())
    if not s:
        return None
    bad = 0
    out: List[str] = []
    for c in s:
        if c in "ACGTN":
            out.append(c)
        elif c in _DNA_IUPAC:
            out.append("N")
        elif c in "-.":
            continue
        else:
            bad += 1
            out.append("N")
    if bad / len(s) > max_bad_frac:
        return None
    cleaned = "".join(out)
    return cleaned if len(cleaned) >= 1 else None


def _as_str(x: Union[str, bytes, None]) -> str:
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


def _blast_alignment_from_pyhmmer(aln: Any, *, width: int = _DEFAULT_WIDTH) -> str:
    """Format pyhmmer Alignment into BLAST-like blocks for extract_profile."""
    q_seq = _as_str(aln.hmm_sequence).replace("U", "T").replace("u", "t")
    t_seq = _as_str(aln.target_sequence).replace("U", "T").replace("u", "t")
    if len(q_seq) != len(t_seq):
        n = min(len(q_seq), len(t_seq))
        q_seq, t_seq = q_seq[:n], t_seq[:n]
    q_from = int(aln.hmm_from)
    t_from = int(aln.target_from)
    lines: List[str] = []
    pos = 0
    qi, ti = q_from, t_from
    while pos < len(q_seq):
        q_chunk = q_seq[pos : pos + width]
        t_chunk = t_seq[pos : pos + width]
        q_letters = sum(1 for c in q_chunk if c not in "-.")
        t_letters = sum(1 for c in t_chunk if c not in "-.")
        q_end = qi + q_letters - 1 if q_letters else qi
        t_end = ti + t_letters - 1 if t_letters else ti
        mid = "".join(
            "|"
            if a.upper() == b.upper() and a not in "-." and b not in "-."
            else " "
            for a, b in zip(q_chunk, t_chunk)
        )
        lines.append(f"Query {qi} {q_chunk} {q_end}")
        lines.append(f"      {mid}")
        lines.append(f"Sbjct {ti} {t_chunk} {t_end}")
        lines.append("")
        qi = q_end + 1 if q_letters else qi
        ti = t_end + 1 if t_letters else ti
        pos += width
    return "\n".join(lines).rstrip() + "\n"


def _identity_pct(q_aln: str, t_aln: str) -> float:
    match = 0
    total = 0
    for a, b in zip(q_aln.upper(), t_aln.upper()):
        if a in "-." or b in "-.":
            continue
        total += 1
        if a == b:
            match += 1
    return 100.0 * match / total if total else 0.0


class LocalPyhmmerBackend:
    """nhmmer vs local FASTA; optional one-time preload into DigitalSequenceBlock."""

    def __init__(
        self,
        *,
        fasta_path: Path,
        db_id: str = "rnacentral_active",
        cpus: int = 0,
        top_k: int = 200,
        E: float = 10.0,
        incE: float = 0.01,
        preload: bool = True,
    ) -> None:
        self.fasta_path = Path(fasta_path)
        self.db_id = db_id
        self.cpus = int(cpus)
        self.top_k = int(top_k)
        self.E = float(E)
        self.incE = float(incE)
        self.preload = bool(preload)
        self._targets = None
        self._alphabet = None
        if not self.fasta_path.is_file():
            raise FileNotFoundError(f"local fasta not found: {self.fasta_path}")

    def _ensure_alphabet(self):
        if self._alphabet is None:
            import pyhmmer

            # RNAcentral active uses DNA alphabet (T).
            self._alphabet = pyhmmer.easel.Alphabet.dna()
        return self._alphabet

    def _load_sanitized_block(self):
        """Load fasta as text, sanitize, digitize — tolerates illegal chars like P."""
        import pyhmmer
        from pyhmmer.easel import DigitalSequenceBlock, SequenceFile, TextSequence

        alphabet = self._ensure_alphabet()
        block = DigitalSequenceBlock(alphabet)
        kept = skipped = 0
        t0 = time.time()
        print(
            f"[local] preloading+sanitizing DB: {self.fasta_path} "
            f"(db_id={self.db_id}) …",
            flush=True,
        )
        with SequenceFile(str(self.fasta_path), digital=False) as sf:
            for i, seq in enumerate(sf, start=1):
                cleaned = sanitize_dna_sequence(seq.sequence)
                if cleaned is None:
                    skipped += 1
                else:
                    name = seq.name if seq.name else f"seq{i}".encode("ascii")
                    block.append(
                        TextSequence(name=name, sequence=cleaned).digitize(alphabet)
                    )
                    kept += 1
                if i % 500_000 == 0:
                    print(
                        f"[local] … read {i} records, kept={kept} skipped={skipped} "
                        f"elapsed_s={time.time() - t0:.0f}",
                        flush=True,
                    )
        print(
            f"[local] preload done kept={kept} skipped={skipped} "
            f"elapsed_s={time.time() - t0:.1f}",
            flush=True,
        )
        if kept == 0:
            raise RuntimeError(f"no valid nucleotide sequences in {self.fasta_path}")
        return block

    def _iter_sanitized(self):
        """Yield digitized sequences one-by-one (no full preload)."""
        from pyhmmer.easel import SequenceFile, TextSequence

        alphabet = self._ensure_alphabet()
        with SequenceFile(str(self.fasta_path), digital=False) as sf:
            for i, seq in enumerate(sf, start=1):
                cleaned = sanitize_dna_sequence(seq.sequence)
                if cleaned is None:
                    continue
                name = seq.name if seq.name else f"seq{i}".encode("ascii")
                yield TextSequence(name=name, sequence=cleaned).digitize(alphabet)

    def _get_targets(self):
        if self.preload:
            if self._targets is None:
                self._targets = self._load_sanitized_block()
            return self._targets
        return self._iter_sanitized()

    def search_one(self, sequence: str) -> Dict[str, Any]:
        t0 = time.time()
        try:
            import pyhmmer
            from pyhmmer.easel import TextSequence
        except ImportError as exc:
            return make_hit_record(
                sequence,
                backend=BACKEND_NAME,
                status="submit_error",
                elapsed_s=time.time() - t0,
                error=f"pyhmmer not installed: {exc}",
                extra={"db_id": self.db_id},
            )

        dna = _to_dna(sequence)
        dna = sanitize_dna_sequence(dna) or ""
        if not dna:
            return make_hit_record(
                sequence,
                backend=BACKEND_NAME,
                status="submit_error",
                elapsed_s=time.time() - t0,
                error="empty/invalid sequence after DNA sanitize",
                extra={"db_id": self.db_id},
            )

        alphabet = self._ensure_alphabet()
        query = TextSequence(name=b"query", sequence=dna).digitize(alphabet)
        try:
            targets = self._get_targets()
        except Exception as exc:  # noqa: BLE001
            return make_hit_record(
                sequence,
                backend=BACKEND_NAME,
                status="error",
                elapsed_s=time.time() - t0,
                error=f"failed loading local DB: {exc}",
                extra={"db_id": self.db_id},
            )

        try:
            hits = next(
                pyhmmer.hmmer.nhmmer(
                    [query],
                    targets,
                    cpus=self.cpus,
                    E=self.E,
                    incE=self.incE,
                )
            )
            entries = self._hits_to_entries(hits)
        except Exception as exc:  # noqa: BLE001
            return make_hit_record(
                sequence,
                backend=BACKEND_NAME,
                status="error",
                elapsed_s=time.time() - t0,
                error=str(exc),
                extra={"db_id": self.db_id},
            )

        return make_hit_record(
            sequence,
            backend=BACKEND_NAME,
            status="ok",
            job_id=f"local:{self.db_id}",
            elapsed_s=time.time() - t0,
            hits=len(entries),
            results=entries,
            extra={"db_id": self.db_id, "fasta": str(self.fasta_path)},
        )

    def _hits_to_entries(self, hits) -> List[Dict[str, Any]]:
        raw: List[Tuple[float, float, Dict[str, Any]]] = []
        for hit in hits:
            name = _as_str(hit.name)
            try:
                score = float(hit.score)
            except Exception:  # noqa: BLE001
                score = 0.0
            try:
                evalue = float(hit.evalue)
            except Exception:  # noqa: BLE001
                evalue = 1e9
            for dom in hit.domains:
                if not (dom.reported or dom.included):
                    continue
                aln = dom.alignment
                if aln is None:
                    continue
                blast = _blast_alignment_from_pyhmmer(aln)
                q_aln = _as_str(aln.hmm_sequence)
                t_aln = _as_str(aln.target_sequence)
                try:
                    d_score = float(dom.score)
                except Exception:  # noqa: BLE001
                    d_score = score
                try:
                    d_e = float(dom.i_evalue)
                except Exception:  # noqa: BLE001
                    d_e = evalue
                entry = {
                    "alignment": blast,
                    "alignment_length": int(
                        sum(1 for c in q_aln if c not in "-.")
                    ),
                    "alignment_start": float(aln.target_from),
                    "alignment_stop": float(aln.target_to),
                    "bias": float(getattr(dom, "bias", 0.0) or 0.0),
                    "description": name,
                    "e_value": d_e,
                    "gap_count": int(
                        sum(1 for a, b in zip(q_aln, t_aln) if a == "-" or b == "-")
                    ),
                    "gaps": 0.0,
                    "id": 0,
                    "identity": _identity_pct(q_aln, t_aln),
                    "match_count": int(
                        sum(
                            1
                            for a, b in zip(q_aln.upper(), t_aln.upper())
                            if a not in "-." and a == b
                        )
                    ),
                    "query_coverage": 0.0,
                    "query_length": int(aln.hmm_length),
                    "rnacentral_id": name,
                    "score": d_score,
                    "target_coverage": 0.0,
                    "target_length": int(aln.target_length),
                    "strand": _as_str(getattr(dom, "strand", "+")) or "+",
                }
                raw.append((-d_score, d_e, entry))

        raw.sort(key=lambda x: (x[0], x[1]))
        if self.top_k > 0:
            raw = raw[: self.top_k]
        out: List[Dict[str, Any]] = []
        for i, (_ns, _e, entry) in enumerate(raw):
            entry["id"] = i
            out.append(entry)
        return out
