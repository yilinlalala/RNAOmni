"""Read/write MSA precompute assets under artifacts/msa/."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.msa_search.contract import (
    CACHE_KEY,
    DIM,
    INDEX_HIT_HEADER,
    INDEX_PROFILE_HEADER,
    MISSING_STRATEGY,
    PROFILE_CHANNELS,
    normalize_rna,
    rnacentral_md5,
    seq_hash,
)


@dataclass
class AssetRoot:
    root: Path

    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    @property
    def hits_dir(self) -> Path:
        return self.root / "hits" / "by_hash"

    @property
    def hits_index(self) -> Path:
        return self.root / "hits" / "index.tsv"

    @property
    def profiles_dir(self) -> Path:
        return self.root / "profiles" / "by_hash"

    @property
    def profiles_index(self) -> Path:
        return self.root / "profiles" / "index.tsv"

    def ensure_layout(self) -> None:
        self.hits_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        if not self.hits_index.exists():
            self.hits_index.write_text(INDEX_HIT_HEADER + "\n", encoding="utf-8")
        if not self.profiles_index.exists():
            self.profiles_index.write_text(INDEX_PROFILE_HEADER + "\n", encoding="utf-8")


def default_meta(
    *,
    backend: str,
    dataset_id: str,
    csv_name: str,
    csv_path: str = "",
    csv_sha256: str = "",
    top_k: int = 200,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "cache_key": CACHE_KEY,
        "dim": DIM,
        "channels": list(PROFILE_CHANNELS),
        "missing_strategy": MISSING_STRATEGY,
        "backend": backend,
        "dataset_id": dataset_id,
        "csv_name": csv_name,
        "csv_path": csv_path,
        "csv_sha256": csv_sha256,
        "top_k": top_k,
        "created_unix": int(time.time()),
        "notes": (
            "Dataset-scoped MSA assets. Train builder must resolve the same "
            "dataset_id as prepare.ACTIVE_CSV stem."
        ),
    }
    if extra:
        meta.update(extra)
    return meta


def write_meta(root: AssetRoot, meta: Dict[str, Any]) -> None:
    root.ensure_layout()
    root.meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def hit_json_path(root: AssetRoot, h: str) -> Path:
    return root.hits_dir / f"{h}.json"


def profile_npy_path(root: AssetRoot, h: str) -> Path:
    return root.profiles_dir / f"{h}.npy"


def load_done_hashes(index_path: Path) -> Dict[str, str]:
    done: Dict[str, str] = {}
    if not index_path.exists():
        return done
    for line in index_path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            done[parts[0]] = parts[1]
    return done


def is_hit_ok(root: AssetRoot, h: str) -> tuple[bool, str]:
    """True if this seq_hash already has a successful search (any backend).

    Online and local share the same dataset directory; skip re-search when
    hits/by_hash/<h>.json exists with status=ok (index is a secondary signal).
    """
    path = hit_json_path(root, h)
    if path.is_file():
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False, "corrupt_json"
        if rec.get("status") == "ok":
            backend = str(rec.get("backend") or "unknown")
            return True, backend
        return False, f"status={rec.get('status')}"
    if load_done_hashes(root.hits_index).get(h) == "ok":
        return True, "index_only"
    return False, "missing"


def is_profile_ok(root: AssetRoot, h: str) -> bool:
    npy = profile_npy_path(root, h)
    if npy.is_file():
        return True
    return load_done_hashes(root.profiles_index).get(h) == "ok"


def load_meta(root: AssetRoot) -> Dict[str, Any]:
    if not root.meta_path.is_file():
        return {}
    try:
        return json.loads(root.meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def merge_write_meta(
    root: AssetRoot,
    *,
    backend: str,
    dataset_id: str,
    csv_name: str,
    csv_path: str = "",
    csv_sha256: str = "",
    top_k: int = 200,
    db_id: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Upsert meta without wiping prior backends (shared online/local dir)."""
    prev = load_meta(root)
    used = list(prev.get("search_backends_used") or [])
    if prev.get("backend") and prev["backend"] not in used:
        used.append(prev["backend"])
    if backend and backend not in used:
        used.append(backend)
    meta = default_meta(
        backend=backend,
        dataset_id=dataset_id,
        csv_name=csv_name or prev.get("csv_name", ""),
        csv_path=csv_path or prev.get("csv_path", ""),
        csv_sha256=csv_sha256 or prev.get("csv_sha256", ""),
        top_k=top_k,
        extra=extra,
    )
    if db_id:
        meta["db_id"] = db_id
    elif prev.get("db_id"):
        meta["db_id"] = prev["db_id"]
    meta["search_backends_used"] = used
    # Preserve first created_unix
    if prev.get("created_unix"):
        meta["created_unix"] = prev["created_unix"]
        meta["updated_unix"] = int(time.time())
    write_meta(root, meta)
    return meta


def append_hit_index(
    root: AssetRoot,
    *,
    h: str,
    status: str,
    n_hits: int,
    elapsed_s: float,
    backend: str,
    job_id: str,
    seq_len: int,
    md5: str,
) -> None:
    root.ensure_layout()
    line = (
        f"{h}\t{status}\t{n_hits}\t{elapsed_s:.3f}\t{backend}\t{job_id}\t{seq_len}\t{md5}\n"
    )
    with root.hits_index.open("a", encoding="utf-8") as f:
        f.write(line)


def write_hit_payload(root: AssetRoot, h: str, payload: Dict[str, Any]) -> Path:
    root.ensure_layout()
    path = hit_json_path(root, h)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_hit_json(root: AssetRoot, h: str) -> Dict[str, Any]:
    return json.loads(hit_json_path(root, h).read_text(encoding="utf-8"))


def write_profile(
    root: AssetRoot,
    h: str,
    profile: Any,
    *,
    status: str,
    n_hits: int,
    n_eff: float,
    backend: str,
) -> Path:
    import numpy as np

    root.ensure_layout()
    arr = np.asarray(profile, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != DIM:
        raise ValueError(f"profile must be (L,{DIM}), got {arr.shape}")
    path = profile_npy_path(root, h)
    np.save(path, arr)
    rel = path.relative_to(root.root).as_posix()
    L = int(arr.shape[0])
    line = (
        f"{h}\t{status}\t{n_hits}\t{n_eff:.4f}\t{L}\t{rel}\t{backend}\t{CACHE_KEY}\n"
    )
    with root.profiles_index.open("a", encoding="utf-8") as f:
        f.write(line)
    return path


def unique_sequences_from_csv(
    csv_path: Path,
    *,
    seq_col: str = "input_seq",
    limit: Optional[int] = None,
) -> List[str]:
    seen = set()
    out: List[str] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if seq_col not in (reader.fieldnames or []):
            raise KeyError(f"column {seq_col!r} not in {csv_path}")
        for row in reader:
            norm = normalize_rna(row[seq_col])
            if not norm or norm in seen:
                continue
            seen.add(norm)
            out.append(norm)
            if limit is not None and len(out) >= limit:
                break
    return out


@dataclass
class TimingStats:
    samples: List[float] = field(default_factory=list)

    def add(self, seconds: float) -> None:
        self.samples.append(float(seconds))

    def summary(self) -> Dict[str, Any]:
        xs = sorted(self.samples)
        n = len(xs)
        if n == 0:
            return {"n": 0}

        def pct(p: float) -> float:
            if n == 1:
                return xs[0]
            i = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
            return xs[i]

        return {
            "n": n,
            "mean_s": sum(xs) / n,
            "p50_s": pct(50),
            "p95_s": pct(95),
            "min_s": xs[0],
            "max_s": xs[-1],
        }


def make_hit_record(
    seq: str,
    *,
    backend: str,
    status: str,
    job_id: str = "",
    elapsed_s: float = 0.0,
    hits: int = 0,
    results: Optional[List[Any]] = None,
    error: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    h = seq_hash(seq)
    rec: Dict[str, Any] = {
        "seq_hash": h,
        "rnacentral_md5": rnacentral_md5(seq),
        "sequence": normalize_rna(seq),
        "seq_len": len(normalize_rna(seq)),
        "backend": backend,
        "status": status,
        "job_id": job_id,
        "elapsed_s": elapsed_s,
        "hits": hits,
        "results": results or [],
        "error": error,
        "cache_key": CACHE_KEY,
        "written_unix": int(time.time()),
    }
    if extra:
        rec.update(extra)
    return rec


def dump_timing_report(path: Path, stats: TimingStats, **meta: Any) -> None:
    payload = {"timing": stats.summary(), **meta}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
