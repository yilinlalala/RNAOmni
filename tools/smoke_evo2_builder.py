#!/usr/bin/env python3
"""D4 re-smoke: encode via features.builders.evo2 + registry params."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from features._yaml import load_yaml  # noqa: E402
from features.builders import evo2 as evo2_builder  # noqa: E402

OUT = REPO / "artifacts" / "pretrained" / "evo2" / "builder_smoke_result.json"


def main() -> int:
    registry = load_yaml(REPO / "features" / "registry.yaml")
    entry = next(e for e in registry["features"] if e["name"] == "evo2")
    params = dict(entry.get("default_params") or {})
    dim = int(entry["dim"])
    if dim != evo2_builder.DIM:
        print(f"FAIL registry dim={dim} != builder.DIM={evo2_builder.DIM}", file=sys.stderr)
        return 1

    seqs = [
        "GGGUGCGAUCAUACCAGCACUAAUGCCCUCCUGGGAAGUCCUCGUGUUGCACCCCU",
        "GGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCA",
    ]
    max_len = 128
    arr = evo2_builder.build(seqs, max_len=max_len, params=params)
    ok = (
        arr.shape == (len(seqs), max_len, dim)
        and arr.dtype == __import__("numpy").float32
        and not __import__("numpy").isnan(arr).any()
        and arr[0, : len(seqs[0])].any()
    )
    payload = {
        "status": "ok" if ok else "fail",
        "shape": list(arr.shape),
        "dim": dim,
        "model_path": params.get("model_path"),
        "device": params.get("device"),
        "nonzero_pos0": float(arr[0, : len(seqs[0])].sum()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not ok:
        print("BUILDER_SMOKE_FAIL", file=sys.stderr)
        return 2
    print("BUILDER_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
