"""Load committed defaults (`rnaomni.env`) then local overlay (`.env`).

Does not override keys already present in ``os.environ`` (shell / CLI /
launcher win). Missing files are fine. No python-dotenv dependency.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_FILE = "rnaomni.env"
OVERLAY_FILES = (".env", ".env.local")


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines. Skips blanks and ``#`` comments."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    text = path.read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key.startswith("#"):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        out[key] = value
    return out


def load_user_env(repo: str | Path | None = None) -> dict[str, str]:
    """Apply ``rnaomni.env`` then ``.env`` / ``.env.local`` via setdefault-on-absent.

    Overlay files win over the committed table. Keys already in ``os.environ``
    at call time are left unchanged. Returns the merged table that was
    considered (not only newly applied keys).
    """
    root = Path(repo) if repo is not None else REPO
    merged: dict[str, str] = {}
    merged.update(parse_env_file(root / DEFAULT_FILE))
    for name in OVERLAY_FILES:
        merged.update(parse_env_file(root / name))
    existing = set(os.environ)
    for key, value in merged.items():
        if key not in existing:
            os.environ[key] = value
    return merged


def env_get(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip()


def env_int(name: str, default: int) -> int:
    return int(env_get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(env_get(name, str(default)))
