#!/usr/bin/env python3
"""16.0: finalize instantiates Predictor, falls back to PooledMLP (CPU, no train)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    from rnaomni.finalize import build_model, resolve_predictor_class, _model_kwargs
    import torch
    import torch.nn as nn

    src = (REPO / "train.py").read_text(encoding="utf-8")
    assert "Predictor = PooledMLP" in src, "train.py must alias Predictor = PooledMLP"

    class LegacyMLP(nn.Module):
        def __init__(
            self,
            input_dim,
            hidden_dim=8,
            dropout=0.0,
            conv_channels=4,
            num_outputs=1,
        ):
            super().__init__()
            self.fc = nn.Linear(input_dim, num_outputs)

        def forward(self, x, lengths):
            return self.fc(x.mean(dim=1))

    class AltHead(nn.Module):
        def __init__(self, input_dim, hidden_dim=8, dropout=0.0, num_outputs=1):
            super().__init__()
            self.fc = nn.Linear(input_dim, hidden_dim)
            self.out = nn.Linear(hidden_dim, num_outputs)

        def forward(self, x, lengths):
            return self.out(self.fc(x.mean(dim=1)))

    old = ModuleType("snap_old")
    old.PooledMLP = LegacyMLP
    old.HIDDEN_DIM = 8
    old.DROPOUT = 0.0
    assert resolve_predictor_class(old) is LegacyMLP
    data = {"input_dim": 4, "task_type": "regression"}
    built = build_model(old, torch.device("cpu"), data)
    assert isinstance(built, LegacyMLP)
    print("legacy PooledMLP-only snapshot OK")

    new = ModuleType("snap_new")
    new.Predictor = AltHead
    new.HIDDEN_DIM = 8
    new.DROPOUT = 0.0
    assert resolve_predictor_class(new) is AltHead
    kw = _model_kwargs(new, data)
    assert "conv_channels" not in kw, kw
    built = build_model(new, torch.device("cpu"), data)
    assert isinstance(built, AltHead)
    print("Predictor without PooledMLP / conv_channels OK")

    both = ModuleType("snap_both")
    both.PooledMLP = LegacyMLP
    both.Predictor = AltHead
    both.HIDDEN_DIM = 8
    assert resolve_predictor_class(both) is AltHead
    print("Predictor preferred over PooledMLP OK")

    empty = ModuleType("snap_empty")
    assert resolve_predictor_class(empty) is None
    print("smoke_predictor_finalize: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
