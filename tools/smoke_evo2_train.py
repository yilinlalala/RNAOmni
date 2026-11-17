#!/usr/bin/env python3
"""C3-style small-scale train smoke for evo2 (does not change TIME_BUDGET).

Uses an in-memory onehot+evo2 recipe — does not edit features/active.yaml.

Example:
  CUDA_VISIBLE_DEVICES=3 python tools/smoke_evo2_train.py --n 256 --steps 40
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from features.pipeline import (  # noqa: E402
    encode_dataframe_from_recipe,
    load_feature_registry,
    resolve_input_dim,
)
from prepare import (  # noqa: E402
    ACTIVE_CSV,
    ALPHABET,
    LABEL_COL,
    MAX_SEQ_LEN,
    SEQUENCE_COL,
    load_dataset,
)
from train import PooledMLP  # noqa: E402

PILOT_RECIPE = {
    "recipe_id": "c3_onehot_evo2",
    "fusion": "concat",
    "features": [
        {"name": "onehot", "enabled": True},
        {"name": "evo2", "enabled": True},
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Small-scale evo2 train smoke")
    parser.add_argument("--n", type=int, default=256, help="Max rows to encode/train")
    parser.add_argument("--steps", type=int, default=40, help="Train steps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "artifacts" / "pretrained" / "evo2" / "train_smoke_result.json",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    recipe = dict(PILOT_RECIPE)
    registry = load_feature_registry()
    enabled = [f["name"] for f in recipe["features"] if f.get("enabled", True)]
    input_dim = resolve_input_dim(recipe, registry)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    df = load_dataset(ACTIVE_CSV).dropna(subset=[SEQUENCE_COL, LABEL_COL]).copy()
    df = df.head(args.n).reset_index(drop=True)
    n = len(df)
    print(
        f"smoke_train: n={n} input_dim={input_dim} features={enabled} "
        f"device={device} steps={args.steps}"
    )

    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    x_np, lengths_np, y_np, meta = encode_dataframe_from_recipe(
        df,
        recipe=recipe,
        registry=registry,
        sequence_col=SEQUENCE_COL,
        label_col=LABEL_COL,
        alphabet=ALPHABET,
        max_seq_len=MAX_SEQ_LEN,
    )
    encode_sec = time.time() - t0
    if meta["input_dim"] != input_dim or x_np.shape[-1] != input_dim:
        print(
            f"FAIL dim mismatch meta={meta['input_dim']} C={x_np.shape[-1]} expect={input_dim}",
            file=sys.stderr,
        )
        return 2
    print(
        f"encode_sec={encode_sec:.2f} shape={x_np.shape} "
        f"n_unique={meta.get('n_unique')} n_rows={meta.get('n_rows', n)}"
    )

    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    n_val = max(1, int(n * args.val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    if len(train_idx) == 0:
        train_idx = val_idx

    uid = torch.from_numpy(np.asarray(meta["row_to_uid"], dtype=np.int64)).to(device)
    x = torch.from_numpy(x_np).to(device)
    lengths = torch.from_numpy(lengths_np).to(device)
    y = torch.from_numpy(y_np).to(device)

    model = PooledMLP(input_dim=input_dim, hidden_dim=args.hidden_dim, dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    t1 = time.time()
    model.train()
    smooth = 0.0
    for step in range(args.steps):
        b = train_idx[rng.randint(0, len(train_idx), size=min(args.batch_size, len(train_idx)))]
        xb, yb, lb = x[uid[b]], y[b], lengths[uid[b]]
        pred = model(xb, lb)
        loss = F.mse_loss(pred, yb)
        if math.isnan(loss.item()) or loss.item() > 1e4:
            print(f"FAIL loss={loss.item()}", file=sys.stderr)
            return 3
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        beta = 0.9
        smooth = beta * smooth + (1 - beta) * loss.item()
        deb = smooth / (1 - beta ** (step + 1))
        print(f"\rstep {step:04d} loss={deb:.6f}", end="", flush=True)
    print()
    train_sec = time.time() - t1

    model.eval()
    with torch.no_grad():
        pred = model(x[uid[val_idx]], lengths[uid[val_idx]])
        val_rmse = float(torch.sqrt(F.mse_loss(pred, y[val_idx])).item())
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0
    )

    payload = {
        "status": "ok",
        "n": n,
        "steps": args.steps,
        "input_dim": input_dim,
        "enabled_features": enabled,
        "recipe_id": recipe.get("recipe_id"),
        "encode_sec": round(encode_sec, 3),
        "train_sec": round(train_sec, 3),
        "val_rmse": val_rmse,
        "peak_vram_mb": round(peak_vram_mb, 1),
        "device": str(device),
        "max_seq_len": MAX_SEQ_LEN,
        "note": "Does not modify features/active.yaml; evo2 stays enabled:false there.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote={args.out}")
    print("TRAIN_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
