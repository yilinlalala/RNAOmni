#!/usr/bin/env python3
"""16.5: minimal attn_pool_mlp forward + short 1-fold train (not overnight).

Does not change master train.py (Predictor stays PooledMLP).
Default CPU so it does not share a card with overnight/finalize.

  python tools/smoke_attn_pool_mlp.py
  python tools/smoke_attn_pool_mlp.py --cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _make_class(nn, torch):
    class _AttnPoolMLP(nn.Module):
        """Stem → WIDTH conv residual → attn+max → MLP. WIDTH = hidden_dim."""

        def __init__(self, input_dim, hidden_dim=256, dropout=0.1, num_outputs=1):
            super().__init__()
            self.input_dim = int(input_dim)
            self.hidden_dim = int(hidden_dim)
            self.num_outputs = int(num_outputs)
            width = self.hidden_dim
            self.stem = (
                nn.Identity()
                if self.input_dim == width
                else nn.Linear(self.input_dim, width)
            )
            self.conv = nn.Sequential(
                nn.Conv1d(width, width, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv1d(width, width, kernel_size=7, padding=3),
                nn.ReLU(),
            )
            self.attn_proj = nn.Linear(width, 1)
            self.net = nn.Sequential(
                nn.Linear(2 * width, width),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(width, width),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(width, self.num_outputs),
            )

        def forward(self, x, lengths):
            bsz, seqlen, _ = x.shape
            pos = torch.arange(seqlen, device=x.device).unsqueeze(0)
            mask = pos < lengths.clamp(min=1).unsqueeze(1)
            hidden = self.stem(x)
            hidden = hidden + self.conv(hidden.transpose(1, 2)).transpose(1, 2) * mask.unsqueeze(-1).to(
                hidden.dtype
            )
            logits = self.attn_proj(hidden).squeeze(-1)
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
            attn = torch.softmax(logits, dim=-1)
            attn_pooled = (hidden * attn.unsqueeze(-1)).sum(dim=1)
            neg_inf = torch.finfo(hidden.dtype).min
            max_pooled = hidden.masked_fill(~mask.unsqueeze(-1), neg_inf).max(dim=1).values
            out = self.net(torch.cat([attn_pooled, max_pooled], dim=-1))
            if self.num_outputs == 1:
                return out.squeeze(-1)
            return out

    return _AttnPoolMLP


def _assert_finite(name: str, t) -> None:
    import torch

    if not torch.isfinite(t).all():
        raise SystemExit(f"FAIL {name} not finite: {t}")


def main() -> int:
    parser = argparse.ArgumentParser(description="16.5 attn_pool_mlp smoke")
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA if available (default: CPU; GPU 1 had ECC errors)",
    )
    args = parser.parse_args()

    from rnaomni.finalize import _model_kwargs, build_model, resolve_predictor_class
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    cls = _make_class(nn, torch)
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"device={device} cuda_available={torch.cuda.is_available()}")

    # Variable C vs WIDTH, pad mask, regression + classification shapes.
    cases = [
        (4, 32, 1),
        (74, 48, 1),
        (266, 64, 1),
        (64, 64, 1),  # identity stem
        (16, 32, 88),
    ]
    for input_dim, width, n_out in cases:
        model = cls(input_dim=input_dim, hidden_dim=width, dropout=0.0, num_outputs=n_out).to(
            device
        )
        bsz, seqlen = 8, 40
        x = torch.randn(bsz, seqlen, input_dim, device=device)
        lengths = torch.randint(5, seqlen + 1, (bsz,), device=device)
        pred = model(x, lengths)
        if n_out == 1:
            assert pred.shape == (bsz,), pred.shape
            y = torch.randn(bsz, device=device)
            loss = F.mse_loss(pred, y)
        else:
            assert pred.shape == (bsz, n_out), pred.shape
            y = torch.randint(0, n_out, (bsz,), device=device)
            loss = F.cross_entropy(pred, y)
        _assert_finite("loss", loss)
        loss.backward()
        print(f"forward C={input_dim} W={width} out={n_out} loss={loss.item():.4f} OK")

    # Short 1-fold loop: 20 steps, kmer4-like dims.
    width = 64
    input_dim = 74
    model = cls(input_dim=input_dim, hidden_dim=width, dropout=0.1, num_outputs=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    bsz, seqlen = 16, 64
    x = torch.randn(200, seqlen, input_dim, device=device)
    lengths = torch.full((200,), 30, device=device, dtype=torch.long)
    y = torch.randn(200, device=device)
    model.train()
    last = None
    for step in range(20):
        idx = torch.randint(0, 200, (bsz,), device=device)
        pred = model(x[idx], lengths[idx])
        loss = F.mse_loss(pred, y[idx])
        _assert_finite(f"step{step}", loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = loss.item()
    print(f"1-fold 20 steps last_loss={last:.4f} OK")

    snap = ModuleType("snap_attn")
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
        out = built(
            torch.randn(3, 10, 12),
            torch.tensor([4, 7, 10]),
        )
    assert out.shape == (3,), out.shape
    print("finalize Predictor=AttnPoolMLP (no conv_channels) OK")

    print("smoke_attn_pool_mlp: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
