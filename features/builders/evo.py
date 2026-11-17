"""L4 Evo token hidden states (encode-only vendor under tools/Evo).

Returns (N, max_len, 4096) float32. No BOS/EOS — channel j aligns with
sequence position j (same contract as onehot / rna_fm).

Evo is a DNA (ACGT) model; U is mapped to T before tokenize.
Hidden states are taken after final RMSNorm, before the vocab unembed.

Zero-install: local HF dir at tools/Evo/model (or params.model_path).
Local weights only — no hub download.
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
_CODE_ROOT = _REPO / "tools" / "Evo"
_DEFAULT_MODEL_DIR = _CODE_ROOT / "model"

_model = None
_loaded_key: tuple[str, str] | None = None


def _normalize_seq(seq: str) -> str:
    # Evo CharLevelTokenizer is DNA; keep RNA letters as ACGT.
    return seq.upper().replace("U", "T")


def _tokenize(seq: str) -> list[int]:
    # stripedhyena.CharLevelTokenizer.tokenize — latin-1 byte ids
    return list(seq.encode("latin-1"))


def _resolve_model_dir(params: dict[str, Any]) -> Path:
    raw = params.get("model_path") or str(_DEFAULT_MODEL_DIR)
    path = Path(os.path.expanduser(str(raw)))
    if not path.is_absolute():
        path = (_REPO / path).resolve()
    else:
        path = path.resolve()
    if path.is_file():
        path = path.parent
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(
            f"evo model_path must be a local HF dir with config.json: {path}"
        )
    if not (path / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"evo weight index missing under {path}")
    return path


def _load_model(model_dir: Path, device: str):
    global _model, _loaded_key
    key = (str(model_dir), device)
    if _model is not None and _loaded_key == key:
        return _model

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    model.eval()
    _model, _loaded_key = model, key
    return model


def _hidden_states(backbone, input_ids, padding_mask):
    """Final-norm hidden states (B, T, 4096), no vocab unembed."""
    x = backbone.embedding_layer.embed(input_ids)
    x, _ = backbone.stateless_forward(x, padding_mask=padding_mask)
    if backbone.norm is not None:
        x = backbone.norm(x)
    return x


def build(sequences, *, max_len: int, alphabet: str = "ACGU", params=None) -> np.ndarray:
    """Return (N, max_len, 4096) float32 Evo hidden states."""
    del alphabet
    params = dict(params or {})
    model_dir = _resolve_model_dir(params)
    device = str(params.get("device") or "cpu")
    batch_size = int(params.get("batch_size", 1))

    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    if str(_CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(_CODE_ROOT))

    model = _load_model(model_dir, device)
    backbone = model.backbone

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
        mask_rows = []
        for seq, L in zip(chunk, lengths):
            row = _tokenize(seq) + [PAD_ID] * (t_max - L)
            ids.append(row)
            mask_rows.append([1.0] * L + [0.0] * (t_max - L))

        hid_dtype = next(backbone.parameters()).dtype
        input_ids = torch.tensor(ids, dtype=torch.long, device=device)
        padding_mask = torch.tensor(mask_rows, dtype=hid_dtype, device=device)
        with torch.no_grad():
            emb = _hidden_states(backbone, input_ids, padding_mask)
        if emb.shape[-1] != DIM:
            raise ValueError(f"evo embedding dim {emb.shape[-1]} != {DIM}")

        for i, L in enumerate(lengths):
            if L <= 0:
                continue
            piece = emb[i, :L, :].detach().float().cpu().numpy()
            out[start + i, :L, :] = piece

    if np.isnan(out).any():
        raise RuntimeError("evo embeddings contain NaN")
    return out
