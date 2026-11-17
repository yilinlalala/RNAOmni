#!/usr/bin/env python3
"""D4 re-smoke: encode via features.builders.<name> + registry params.

Not feature-hardcoded beyond --feature-name (default: rinalmo).

Usage:
  python tools/smoke_rinalmo_builder.py
  python tools/smoke_rinalmo_builder.py --feature-name rinalmo
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from features._yaml import load_yaml  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="L4 builder re-smoke via registry")
    parser.add_argument("--feature-name", default="rinalmo")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: artifacts/pretrained/<feature>/builder_smoke_result.json",
    )
    parser.add_argument("--max-len", type=int, default=128)
    args = parser.parse_args()

    name = args.feature_name
    out = args.out or (
        REPO / "artifacts" / "pretrained" / name / "builder_smoke_result.json"
    )

    registry = load_yaml(REPO / "features" / "registry.yaml")
    try:
        entry = next(e for e in registry["features"] if e["name"] == name)
    except StopIteration:
        print(f"FAIL: {name} not in features/registry.yaml", file=sys.stderr)
        return 1

    module_name = entry.get("module") or f"features.builders.{name}"
    params = dict(entry.get("default_params") or {})
    dim = int(entry["dim"])

    mod = importlib.import_module(module_name)
    seqs = [
        "GGGUGCGAUCAUACCAGCACUAAUGCCCUCCUGGGAAGUCCUCGUGUUGCACCCCU",
        "GGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCA",
    ]
    max_len = args.max_len
    try:
        arr = mod.build(seqs, max_len=max_len, params=params)
    except Exception as err:
        print(f"BUILDER_SMOKE_FAIL: {type(err).__name__}: {err}", file=sys.stderr)
        return 2

    ok = (
        arr.shape == (len(seqs), max_len, dim)
        and arr.dtype == np.float32
        and not np.isnan(arr).any()
        and arr[0, : len(seqs[0])].any()
    )
    payload = {
        "status": "ok" if ok else "fail",
        "feature_name": name,
        "module": module_name,
        "shape": list(arr.shape),
        "dim": dim,
        "model_path": params.get("model_path"),
        "code_root": params.get("code_root"),
        "lm_config": params.get("lm_config"),
        "device": params.get("device"),
        "nonzero_pos0": float(arr[0, : len(seqs[0])].sum()) if ok else None,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not ok:
        print(
            f"BUILDER_SMOKE_FAIL shape={arr.shape} expect=({len(seqs)},{max_len},{dim})",
            file=sys.stderr,
        )
        return 3
    print("BUILDER_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
