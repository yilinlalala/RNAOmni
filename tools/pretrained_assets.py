#!/usr/bin/env python3
"""Shared helpers for L4 pretrained asset YAML + local weight resolution.

Used by deploy_feature.py and smoke_*.py. No network downloads.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

# Filename token → lm_config / embed dim (RiNALMo README scales)
LM_DIM: dict[str, int] = {
    "nano": 320,
    "micro": 480,
    "mega": 640,
    "giga": 1280,
}

# Prefer smallest pretrained scale when several exist under weights/
DEFAULT_WEIGHT_ORDER: tuple[str, ...] = ("micro", "mega", "giga", "nano")


def expand_path(path_str: str, *, base: Path | None = None) -> Path:
    path = Path(os.path.expanduser(str(path_str)))
    if not path.is_absolute():
        if base is None:
            raise ValueError(f"relative path needs base: {path_str}")
        path = (base / path).resolve()
    else:
        path = path.resolve()
    return path


def parse_assets(path: Path) -> dict[str, Any]:
    """Minimal assets YAML parse (flat keys + weight_candidates[].path)."""
    text = path.read_text(encoding="utf-8")
    out: dict[str, Any] = {"weight_candidates": []}
    weights: list[str] = []
    in_weights = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if re.match(r"^weight_candidates:\s*$", line):
            in_weights = True
            continue
        if in_weights:
            m = re.match(r"^\s+-\s+path:\s*(.+)$", line)
            if m:
                weights.append(m.group(1).strip().strip("'\""))
                continue
            if re.match(r"^[a-zA-Z_]", line):
                in_weights = False
            else:
                continue
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip().strip("'\"")
        if key == "weight_candidates":
            continue
        if val == "" or val in ("|", ">"):
            continue
        if re.fullmatch(r"-?\d+", val):
            out[key] = int(val)
        elif val.lower() in ("true", "false", "null", "~"):
            out[key] = {"true": True, "false": False, "null": None, "~": None}[val.lower()]
        else:
            out[key] = val
    out["weight_candidates"] = weights
    return out


def resolve_code_root(
    assets: dict[str, Any],
    *,
    repo: Path,
    code_root_override: str | Path | None = None,
) -> Path:
    if code_root_override:
        raw = Path(os.path.expanduser(str(code_root_override)))
        if not raw.is_absolute():
            raw = (repo / raw).resolve()
        else:
            raw = raw.resolve()
        return raw
    rel = str(assets.get("code_root") or "")
    if not rel:
        raise FileNotFoundError("assets missing code_root; pass --code-root")
    return expand_path(rel, base=repo)


def infer_lm_config(weight_path: Path, assets: dict[str, Any] | None = None) -> str:
    name = weight_path.name.lower()
    for scale in ("giga", "mega", "micro", "nano"):
        if scale in name:
            return scale
    if assets and assets.get("lm_config"):
        return str(assets["lm_config"])
    return "micro"


def expected_dim_for(lm_config: str, assets: dict[str, Any] | None = None) -> int:
    if lm_config in LM_DIM:
        return LM_DIM[lm_config]
    if assets and assets.get("expected_dim_doc") is not None:
        return int(assets["expected_dim_doc"])
    raise KeyError(f"unknown lm_config={lm_config!r}; known={sorted(LM_DIM)}")


def resolve_local_weight(
    candidates: list[str],
    *,
    code_root: Path,
    repo: Path | None = None,
) -> Path:
    """Return first existing candidate. Paths are relative to code_root, then repo."""
    tried: list[str] = []
    bases = [code_root]
    if repo is not None:
        bases.append(repo)
    for rel in candidates:
        for base in bases:
            p = expand_path(rel, base=base)
            tried.append(str(p))
            if p.is_file():
                return p
        # absolute candidate
        p = Path(os.path.expanduser(rel))
        if p.is_absolute():
            p = p.resolve()
            tried.append(str(p))
            if p.is_file():
                return p
    raise FileNotFoundError(
        "No local weight found (downloads disabled). Tried:\n  - " + "\n  - ".join(tried)
    )


def package_marker_ok(code_root: Path, assets: dict[str, Any]) -> None:
    """Ensure upstream package dir exists under code_root (assets-driven)."""
    marker = str(assets.get("package_dir") or assets.get("feature_name") or "").strip()
    if not marker:
        # fallback: any of common package dirs
        for name in ("rinalmo", "fm", "evo", "evo2"):
            if (code_root / name).is_dir():
                return
        raise FileNotFoundError(
            f"no package_dir in assets and no rinalmo/|fm/|evo/|evo2/ under {code_root}"
        )
    if not (code_root / marker).is_dir():
        raise FileNotFoundError(f"expected {marker}/ under code_root: {code_root}")


def write_whitelist_for(feature_name: str) -> tuple[str, ...]:
    return (
        f"features/builders/{feature_name}.py",
        "features/registry.yaml",
        "features/active.yaml",
        f"artifacts/pretrained/{feature_name}/",
        f"skill-library/feature/examples/{feature_name}.handoff.md",
        "skill-library/feature/deploy-ops.md",
    )
