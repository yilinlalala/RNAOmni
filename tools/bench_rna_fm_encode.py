#!/usr/bin/env python3
"""C2: full-dataset encode timing for rna_fm (does not edit features/active.yaml).

Example:
  CUDA_VISIBLE_DEVICES=3 conda run -n myenv3.11 python tools/bench_rna_fm_encode.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="C2 full encode bench for rna_fm")
    parser.add_argument(
        "--mode",
        choices=("onehot_rna_fm", "rna_fm_only", "baseline_no_l4"),
        default="onehot_rna_fm",
        help="Feature mix to encode (default: onehot+rna_fm pilot)",
    )
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "pretrained" / "rna_fm" / "encode_bench.json")
    parser.add_argument("--device", default=None, help="Override rna_fm device (default: cuda if available)")
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(REPO))
    from prepare import (  # noqa: E402
        ACTIVE_CSV,
        ALPHABET,
        MAX_SEQ_LEN,
        SEQUENCE_COL,
        LABEL_COL,
        load_dataset,
    )
    from features.pipeline import (  # noqa: E402
        encode_dataframe_from_recipe,
        load_feature_registry,
        resolve_input_dim,
    )

    if args.mode == "onehot_rna_fm":
        recipe = {
            "recipe_id": "c2_onehot_rna_fm",
            "fusion": "concat",
            "features": [
                {"name": "onehot", "enabled": True},
                {"name": "length_frac", "enabled": False},
                {"name": "kmer", "enabled": False},
                {"name": "rna_fm", "enabled": True},
            ],
        }
    elif args.mode == "rna_fm_only":
        recipe = {
            "recipe_id": "c2_rna_fm_only",
            "fusion": "concat",
            "features": [
                {"name": "onehot", "enabled": False},
                {"name": "length_frac", "enabled": False},
                {"name": "kmer", "enabled": False},
                {"name": "rna_fm", "enabled": True},
            ],
        }
    else:
        recipe = {
            "recipe_id": "c2_baseline_no_l4",
            "fusion": "concat",
            "features": [
                {"name": "onehot", "enabled": True},
                {"name": "length_frac", "enabled": True},
                {"name": "kmer", "enabled": True},
                {"name": "rna_fm", "enabled": False},
            ],
        }

    registry = load_feature_registry()
    # Point rna_fm at visible device without mutating registry file permanently:
    # copy entry params for this process only via encode path — registry default_params
    # already has device:cuda; override by patching a shallow copy of registry features.
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    reg = {"version": registry.get("version"), "features": []}
    for entry in registry["features"]:
        e = dict(entry)
        if e["name"] == "rna_fm":
            params = dict(e.get("default_params") or {})
            params["device"] = device
            e["default_params"] = params
        reg["features"].append(e)

    input_dim = resolve_input_dim(recipe, reg)
    df = load_dataset(ACTIVE_CSV)
    n = len(df)

    oom = False
    err = None
    peak_mb = None
    shape = None
    enabled = None
    n_unique = None
    t0 = time.perf_counter()
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    try:
        x, lengths, labels, meta = encode_dataframe_from_recipe(
            df,
            sequence_col=SEQUENCE_COL,
            label_col=LABEL_COL,
            max_seq_len=MAX_SEQ_LEN,
            alphabet=ALPHABET,
            recipe=recipe,
            registry=reg,
        )
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        shape = list(x.shape)
        enabled = meta.get("enabled_features")
        n_unique = meta.get("n_unique")
        if torch.cuda.is_available() and device.startswith("cuda"):
            peak_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 1)
    except RuntimeError as exc:
        elapsed = time.perf_counter() - t0
        err = f"{type(exc).__name__}: {exc}"
        oom = "out of memory" in str(exc).lower() or "oom" in str(exc).lower()
        if torch.cuda.is_available() and device.startswith("cuda"):
            peak_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 1)

    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    # Gate hint vs 5 min train budget (encode happens before training clock)
    budget_s = 300.0
    fits_hint = None
    if err is None:
        # Heuristic: leave >=120s for actual training after encode
        fits_hint = elapsed <= (budget_s - 120.0)

    payload = {
        "status": "ok" if err is None else "fail",
        "mode": args.mode,
        "n_sequences": n,
        "n_unique": n_unique,
        "max_seq_len": MAX_SEQ_LEN,
        "input_dim": input_dim,
        "enabled_features": enabled,
        "x_shape": shape,
        "device": device,
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": gpu_name,
        "wall_clock_sec": round(elapsed, 3),
        "peak_vram_mb": peak_mb,
        "oom": oom,
        "error": err,
        "train_budget_sec": budget_s,
        "fits_5min_heuristic_leave_120s_for_train": fits_hint,
        "note": "Does not modify features/active.yaml; rna_fm stays enabled:false there.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if err is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
