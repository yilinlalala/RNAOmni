"""L4 Evo2 token hidden states (encode-only vendor under tools/Evo2).

Returns (N, max_len, 4096) float32. No BOS/EOS — channel j aligns with
sequence position j (same contract as onehot / evo / rna_fm).

Evo2 is a DNA (ACGT) model; U is mapped to T before tokenize.
Hidden states are the final RMSNorm outputs (hook on ``norm``).

Zero-install: PYTHONPATH includes tools/Evo2. Local ``evo2_7b.pt`` only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

DIM = 4096
PAD_ID = 1
_REPO = Path(__file__).resolve().parents[2]
_CODE_ROOT = _REPO / "tools" / "Evo2"
_DEFAULT_WEIGHT = _CODE_ROOT / "weights" / "evo2_7b.pt"

_model = None
_loaded_key: tuple[str, str, str] | None = None


def _normalize_seq(seq: str) -> str:
    return seq.upper().replace("U", "T")


def _tokenize(seq: str) -> list[int]:
    # vortex CharLevelTokenizer.tokenize — utf-8 byte ids
    return list(seq.encode("utf-8"))


def _ensure_import() -> None:
    root = str(_CODE_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _resolve_weight(params: dict[str, Any]) -> Path:
    raw = params.get("model_path") or str(_DEFAULT_WEIGHT)
    path = Path(os.path.expanduser(str(raw)))
    if not path.is_absolute():
        for base in (_REPO, _CODE_ROOT):
            cand = (base / path).resolve()
            if cand.is_file():
                return cand
        path = (_REPO / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"evo2 model_path not found: {path}")
    return path


def _load_model(weight: Path, device: str, model_name: str = "evo2_7b"):
    global _model, _loaded_key
    key = (str(weight), device, model_name)
    if _model is not None and _loaded_key == key:
        return _model

    _ensure_import()
    from evo2 import Evo2

    evo2_model = Evo2(model_name, local_path=str(weight))
    evo2_model.model.eval()
    _model, _loaded_key = evo2_model, key
    return evo2_model


def build(sequences, *, max_len: int, alphabet: str = "ACGU", params=None) -> np.ndarray:
    """Return (N, max_len, 4096) float32 Evo2 hidden states."""
    del alphabet
    params = dict(params or {})
    weight = _resolve_weight(params)
    device = str(params.get("device") or "cpu")
    batch_size = int(params.get("batch_size", 1))
    model_name = str(params.get("model_name") or "evo2_7b")

    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    encode_device = "cuda:0" if device.startswith("cuda") else device

    model = _load_model(weight, device, model_name=model_name)

    n = len(sequences)
    out = np.zeros((n, max_len, DIM), dtype=np.float32)
    normed = [_normalize_seq(s)[:max_len] for s in sequences]

    for start in range(0, n, batch_size):
        chunk = normed[start : start + batch_size]
        lengths = [len(seq) for seq in chunk]
        t_max = max(lengths) if lengths else 0
        if t_max <= 0:
            continue

        ids = []
        for seq, L in zip(chunk, lengths):
            row = _tokenize(seq) + [PAD_ID] * (t_max - L)
            ids.append(row)

        input_ids = torch.tensor(ids, dtype=torch.long, device=encode_device)
        with torch.no_grad():
            _logits, embeddings = model(
                input_ids,
                return_embeddings=True,
                layer_names=["norm"],
            )
        if embeddings is None or "norm" not in embeddings:
            raise RuntimeError("evo2 forward did not return norm embeddings")
        emb = embeddings["norm"]
        if emb.shape[-1] != DIM:
            raise ValueError(f"evo2 embedding dim {emb.shape[-1]} != {DIM}")

        for i, L in enumerate(lengths):
            if L <= 0:
                continue
            piece = emb[i, :L, :].detach().float().cpu().numpy()
            out[start + i, :L, :] = piece

    if np.isnan(out).any():
        raise RuntimeError("evo2 embeddings contain NaN")
    return out
