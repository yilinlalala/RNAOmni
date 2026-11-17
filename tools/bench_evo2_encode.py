#!/usr/bin/env python3
"""C2: full-dataset unique encode timing for evo2 (does not edit features/active.yaml).

Example:
  CUDA_VISIBLE_DEVICES=3 python tools/bench_evo2_encode.py --mode evo2_only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="C2 full encode bench for evo2")
    parser.add_argument(
        "--mode",
        choices=("onehot_evo2", "evo2_only"),
        default="evo2_only",
        help="Feature mix to encode (default: evo2 only)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "artifacts" / "pretrained" / "evo2" / "encode_bench.json",
    )
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

    if args.mode == "onehot_evo2":
        recipe = {
            "recipe_id": "c2_onehot_evo2",
            "fusion": "concat",
            "features": [
                {"name": "onehot", "enabled": True},
                {"name": "evo2", "enabled": True},
            ],
        }
    else:
        recipe = {
            "recipe_id": "c2_evo2_only",
            "fusion": "concat",
            "features": [{"name": "evo2", "enabled": True}],
        }

    registry = load_feature_registry()
    input_dim = resolve_input_dim(recipe, registry)
    df = load_dataset(ACTIVE_CSV).dropna(subset=[SEQUENCE_COL, LABEL_COL]).copy()
    print(
        f"bench_evo2: mode={args.mode} rows={len(df)} input_dim={input_dim} "
        f"csv={ACTIVE_CSV} max_len={MAX_SEQ_LEN}",
        flush=True,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    x_np, lengths_np, _y, meta = encode_dataframe_from_recipe(
        df,
        recipe=recipe,
        registry=registry,
        sequence_col=SEQUENCE_COL,
        label_col=LABEL_COL,
        alphabet=ALPHABET,
        max_seq_len=MAX_SEQ_LEN,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    encode_sec = time.time() - t0
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
    )
    payload = {
        "status": "ok",
        "mode": args.mode,
        "n_rows": int(meta.get("n_rows", len(df))),
        "n_unique": int(meta.get("n_unique", -1)),
        "input_dim": int(meta["input_dim"]),
        "shape": list(x_np.shape),
        "max_len": int(lengths_np.max()) if len(lengths_np) else 0,
        "encode_sec": round(encode_sec, 3),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "note": "Does not modify features/active.yaml; evo2 stays enabled:false there.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("ENCODE_BENCH_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
