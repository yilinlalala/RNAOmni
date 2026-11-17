#!/usr/bin/env python3
"""Named-head forward smokes: A1 tf_* plus A2 extras (not overnight).

Imports the classes from train.py. Predictor must stay PooledMLP.

  python tools/smoke_tf_heads.py
  python tools/smoke_tf_heads.py --cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import rnaomni.threads as _threads

# train.py / finalize call set_num_interop_threads; some torch builds abort
# instead of raising if the pool already started.
_threads._limit_torch = lambda n: None  # type: ignore[assignment]


def _assert_finite(name: str, t) -> None:
    import torch

    if not torch.isfinite(t).all():
        raise SystemExit(f"FAIL {name} not finite: {t}")


def _run_class(cls, device, torch, F) -> None:
    cases = [
        (4, 32, 1),
        (74, 48, 1),
        (266, 64, 1),
        (64, 64, 1),
        (16, 32, 88),
        (8, 33, 1),  # WIDTH not divisible by 4 → nhead=1
    ]
    for input_dim, width, n_out in cases:
        model = cls(
            input_dim=input_dim, hidden_dim=width, dropout=0.0, num_outputs=n_out
        ).to(device)
        bsz, seqlen = 6, 24
        x = torch.randn(bsz, seqlen, input_dim, device=device)
        lengths = torch.randint(4, seqlen + 1, (bsz,), device=device)
        pred = model(x, lengths)
        if n_out == 1:
            assert pred.shape == (bsz,), (cls.__name__, pred.shape)
            y = torch.randn(bsz, device=device)
            loss = F.mse_loss(pred, y)
        else:
            assert pred.shape == (bsz, n_out), (cls.__name__, pred.shape)
            y = torch.randint(0, n_out, (bsz,), device=device)
            loss = F.cross_entropy(pred, y)
        _assert_finite(f"{cls.__name__} loss", loss)
        loss.backward()
        print(f"{cls.__name__} C={input_dim} W={width} out={n_out} loss={loss.item():.4f} OK")


def main() -> int:
    parser = argparse.ArgumentParser(description="A1 tf_mean / tf_cls smoke")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    from train import (
        GatedPoolMLP,
        GruPool,
        MaxPoolMLP,
        MeanDeepMLP,
        PooledMLP,
        Predictor,
        TfAttn,
        TfCls,
        TfConv,
        TfMean,
    )
    from rnaomni.finalize import _model_kwargs, build_model, resolve_predictor_class
    import torch
    import torch.nn.functional as F

    assert Predictor is PooledMLP, Predictor
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"device={device} Predictor={Predictor.__name__}")

    _run_class(TfMean, device, torch, F)
    _run_class(TfCls, device, torch, F)
    a2 = (TfAttn, TfConv, MaxPoolMLP, GatedPoolMLP, GruPool, MeanDeepMLP)
    for cls in a2:
        _run_class(cls, device, torch, F)

    for cls in (TfMean, TfCls) + a2:
        model = cls(input_dim=74, hidden_dim=64, dropout=0.1, num_outputs=1).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
        bsz, seqlen = 8, 32
        x = torch.randn(80, seqlen, 74, device=device)
        lengths = torch.full((80,), 20, device=device, dtype=torch.long)
        y = torch.randn(80, device=device)
        last = None
        model.train()
        for step in range(8):
            idx = torch.randint(0, 80, (bsz,), device=device)
            pred = model(x[idx], lengths[idx])
            loss = F.mse_loss(pred, y[idx])
            _assert_finite(f"{cls.__name__} step{step}", loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last = loss.item()
        print(f"{cls.__name__} 8 steps last_loss={last:.4f} OK")

        snap = ModuleType(f"snap_{cls.__name__}")
        snap.Predictor = cls
        snap.HIDDEN_DIM = 48
        snap.DROPOUT = 0.0
        assert resolve_predictor_class(snap) is cls
        data = {"input_dim": 12, "task_type": "regression"}
        kw = _model_kwargs(snap, data)
        assert "conv_channels" not in kw, kw
        built = build_model(snap, torch.device("cpu"), data)
        assert isinstance(built, cls)
        with torch.no_grad():
            out = built(torch.randn(3, 10, 12), torch.tensor([4, 7, 10]))
        assert out.shape == (3,), out.shape
        print(f"finalize Predictor={cls.__name__} (no conv_channels) OK")

    print("smoke_tf_heads: OK (A1+A2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
