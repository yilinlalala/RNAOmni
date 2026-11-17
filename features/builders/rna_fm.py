"""L4 RNA-FM token embeddings (S2b-Deploy).

Returns (N, max_len, 640) float32. Special tokens (BOS/EOS) are stripped so
channel j aligns with sequence position j (same contract as onehot/kmer).

Zero-install: PYTHONPATH includes tools/RNA-FM (or this module adds it).
Workarounds validated in D3 smoke (do not pip-install for these):
  - stub ptflops before `import fm` (downstream FLOPs only)
  - torch.load(..., weights_only=False) for trusted local ckpt (torch>=2.6)
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

DIM = 640
# RNA-FM T12: 0 = token embed, 12 = last transformer block.
REPR_LAYER_MIN = 0
REPR_LAYER_MAX = 12
_REPO = Path(__file__).resolve().parents[2]
_CODE_ROOT = _REPO / "tools" / "RNA-FM"

_model = None
_alphabet = None
_loaded_key: tuple[str, str] | None = None


def _ensure_fm_import() -> None:
    if str(_CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(_CODE_ROOT))
    if "ptflops" not in sys.modules:
        stub = types.ModuleType("ptflops")
        stub.get_model_complexity_info = lambda *a, **k: (0, 0)
        sys.modules["ptflops"] = stub


def parse_repr_layer(params: dict[str, Any]) -> int:
    """Validate recipe/registry `repr_layer` for RNA-FM T12 (0..12)."""
    layer = int(params.get("repr_layer", 12))
    if layer < REPR_LAYER_MIN or layer > REPR_LAYER_MAX:
        raise ValueError(
            f"rna_fm repr_layer must be in {REPR_LAYER_MIN}..{REPR_LAYER_MAX}, got {layer}"
        )
    return layer


def _normalize_seq(seq: str) -> str:
    return seq.upper().replace("T", "U")


def _resolve_model_path(params: dict[str, Any]) -> Path:
    raw = params.get("model_path")
    if not raw:
        raise ValueError(
            "rna_fm requires default_params.model_path (local checkpoint); "
            "see skill-library/feature/examples/rna_fm.assets.yaml"
        )
    path = Path(os.path.expanduser(str(raw)))
    if not path.is_absolute():
        path = (_REPO / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"rna_fm model_path not found: {path}")
    return path


def _load_model(model_path: Path, device: str):
    global _model, _alphabet, _loaded_key
    key = (str(model_path), device)
    if _model is not None and _loaded_key == key:
        return _model, _alphabet

    _ensure_fm_import()
    import torch
    import fm

    _torch_load = torch.load

    def _torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _torch_load(*args, **kwargs)

    torch.load = _torch_load_compat  # type: ignore[assignment]
    try:
        model, alphabet = fm.pretrained.rna_fm_t12(str(model_path))
    finally:
        torch.load = _torch_load  # type: ignore[assignment]

    model = model.to(device)
    model.eval()
    _model, _alphabet, _loaded_key = model, alphabet, key
    return model, alphabet


def build(sequences, *, max_len: int, alphabet: str = "ACGU", params=None) -> np.ndarray:
    """Return (N, max_len, 640) float32 RNA-FM embeddings."""
    del alphabet  # RNA-FM uses its own RNA alphabet
    params = dict(params or {})
    model_path = _resolve_model_path(params)
    repr_layer = parse_repr_layer(params)
    device = str(params.get("device") or "cpu")
    batch_size = int(params.get("batch_size", 8))

    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model, fm_alphabet = _load_model(model_path, device)
    batch_converter = fm_alphabet.get_batch_converter()
    bos = int(fm_alphabet.prepend_bos)

    n = len(sequences)
    out = np.zeros((n, max_len, DIM), dtype=np.float32)
    normed = [_normalize_seq(s)[:max_len] for s in sequences]

    for start in range(0, n, batch_size):
        chunk = normed[start : start + batch_size]
        data = [(f"s{start + i}", seq) for i, seq in enumerate(chunk)]
        _labels, _strs, tokens = batch_converter(data)
        tokens = tokens.to(device)
        with torch.no_grad():
            results = model(tokens, repr_layers=[repr_layer])
        emb = results["representations"][repr_layer]  # (B, T, C)
        if emb.shape[-1] != DIM:
            raise ValueError(f"rna_fm embedding dim {emb.shape[-1]} != {DIM}")

        for i, seq in enumerate(chunk):
            L = len(seq)
            if L <= 0:
                continue
            # Skip BOS; take L residue tokens
            piece = emb[i, bos : bos + L, :].detach().float().cpu().numpy()
            out[start + i, :L, :] = piece

    if np.isnan(out).any():
        raise RuntimeError("rna_fm embeddings contain NaN")
    return out
