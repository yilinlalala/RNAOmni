"""L4 RiNALMo token embeddings (S2b-Deploy).

Returns (N, max_len, C) float32 where C depends on lm_config:
  micro=480, mega=640, giga=1280 (nano=320).
Special tokens (CLS/EOS) are stripped so channel j aligns with sequence position j.

Zero-install: PYTHONPATH includes code_root (default tools/RiNALMo), or this
module inserts it. Official Zenodo weights use FlashAttention (Wqkv). Default
use_flash_attn=auto: remap official Wqkv weights to std attention (portable).
Set use_flash_attn=true only with compatible flash-attn (RiNALMo pins 2.3.x).
Local weights only — no gdown.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[2]

# Prefer smallest scale when auto-resolving under {code_root}/weights/
_WEIGHT_ORDER = ("micro", "mega", "giga", "nano")
_LM_DIM = {"nano": 320, "micro": 480, "mega": 640, "giga": 1280}

_model = None
_alphabet = None
_loaded_key: tuple[str, str, str, bool] | None = None


def _normalize_seq(seq: str) -> str:
    # Alphabet.encode maps U→T; keep RNA letters uppercased
    return seq.upper().replace("T", "U")


def _ensure_import(code_root: Path) -> None:
    root = str(code_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _infer_lm_config(weight: Path, params: dict[str, Any]) -> str:
    if params.get("lm_config"):
        return str(params["lm_config"])
    name = weight.name.lower()
    for scale in ("giga", "mega", "micro", "nano"):
        if scale in name:
            return scale
    return "micro"


def _expected_dim(lm_config: str, params: dict[str, Any]) -> int:
    if params.get("expected_dim") is not None:
        return int(params["expected_dim"])
    if lm_config not in _LM_DIM:
        raise ValueError(f"rinalmo unknown lm_config={lm_config!r}; known={sorted(_LM_DIM)}")
    return _LM_DIM[lm_config]


def _resolve_code_root(params: dict[str, Any]) -> Path:
    raw = params.get("code_root") or "tools/RiNALMo"
    path = Path(os.path.expanduser(str(raw)))
    if not path.is_absolute():
        path = (_REPO / path).resolve()
    else:
        path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"rinalmo code_root not found: {path}")
    return path


def _auto_weight(code_root: Path, params: dict[str, Any]) -> Path:
    weights_dir = code_root / str(params.get("weights_subdir") or "weights")
    order = params.get("weight_preference") or list(_WEIGHT_ORDER)
    tried: list[str] = []
    for scale in order:
        cand = weights_dir / f"rinalmo_{scale}_pretrained.pt"
        tried.append(str(cand))
        if cand.is_file():
            return cand.resolve()
    # any *.pt under weights/ as last resort (sorted by size ascending)
    if weights_dir.is_dir():
        pts = sorted(weights_dir.glob("*.pt"), key=lambda p: p.stat().st_size)
        if pts:
            return pts[0].resolve()
    raise FileNotFoundError(
        "rinalmo: no local weight under "
        f"{weights_dir} (place README weights there). Tried:\n  - "
        + "\n  - ".join(tried)
    )


def _resolve_model_path(params: dict[str, Any], code_root: Path) -> Path:
    raw = params.get("model_path")
    if not raw:
        return _auto_weight(code_root, params)
    path = Path(os.path.expanduser(str(raw)))
    if not path.is_absolute():
        # try repo-relative, then code_root-relative
        for base in (_REPO, code_root):
            cand = (base / path).resolve()
            if cand.is_file():
                return cand
        path = (_REPO / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"rinalmo model_path not found: {path}")
    return path


def _parse_flash_request(params: dict[str, Any]) -> bool | str | None:
    raw = params.get("use_flash_attn", "auto")
    if raw is None or raw == "auto":
        return "auto"
    if isinstance(raw, str) and raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    return bool(raw)


def _load_model(
    model_path: Path,
    code_root: Path,
    lm_config: str,
    device: str,
    flash_request: bool | str | None,
):
    global _model, _alphabet, _loaded_key

    _ensure_import(code_root)
    if str(_REPO / "tools") not in sys.path:
        sys.path.insert(0, str(_REPO / "tools"))

    import torch
    from rinalmo.config import model_config
    from rinalmo.data.alphabet import Alphabet
    from rinalmo.model.model import RiNALMo
    from rinalmo_weights import flash_attn_available, peek_use_flash_attn

    _torch_load = torch.load

    def _torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _torch_load(*args, **kwargs)

    torch.load = _torch_load_compat  # type: ignore[assignment]
    try:
        raw = torch.load(str(model_path), map_location="cpu")
    finally:
        torch.load = _torch_load  # type: ignore[assignment]

    use_flash, remapped = peek_use_flash_attn(raw, flash_request)
    key = (str(model_path), lm_config, device, use_flash)
    if _model is not None and _loaded_key == key:
        return _model, _alphabet

    if use_flash and not flash_attn_available():
        raise ImportError(
            "Need flash_attn for flash checkpoint path, or set "
            "use_flash_attn=false / auto (remap without flash_attn)."
        )

    config = model_config(lm_config)
    config.model.transformer.use_flash_attn = bool(use_flash)
    model = RiNALMo(config)
    alphabet = Alphabet(**config["alphabet"])

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    missing_critical = [m for m in missing if "rotary_emb" not in m]
    if unexpected or missing_critical:
        raise RuntimeError(
            f"rinalmo load failed unexpected={list(unexpected)[:5]} "
            f"missing_critical={missing_critical[:5]}"
        )

    model = model.to(device)
    model.eval()
    _model, _alphabet, _loaded_key = model, alphabet, key
    return model, alphabet


def build(sequences, *, max_len: int, alphabet: str = "ACGU", params=None) -> np.ndarray:
    """Return (N, max_len, C) float32 RiNALMo embeddings."""
    del alphabet
    params = dict(params or {})
    code_root = _resolve_code_root(params)
    model_path = _resolve_model_path(params, code_root)
    lm_config = _infer_lm_config(model_path, params)
    expected = _expected_dim(lm_config, params)
    device = str(params.get("device") or "cpu")
    batch_size = int(params.get("batch_size", 4))
    flash_request = _parse_flash_request(params)

    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model, tok_alphabet = _load_model(
        model_path, code_root, lm_config, device, flash_request
    )

    n = len(sequences)
    out = np.zeros((n, max_len, expected), dtype=np.float32)
    normed = [_normalize_seq(s)[:max_len] for s in sequences]

    for start in range(0, n, batch_size):
        chunk = normed[start : start + batch_size]
        # pad empty to keep batch_tokenize happy
        safe = [seq if seq else "A" for seq in chunk]
        tokens = torch.tensor(
            tok_alphabet.batch_tokenize(safe),
            dtype=torch.int64,
            device=device,
        )
        with torch.no_grad():
            if device.startswith("cuda"):
                with torch.cuda.amp.autocast():
                    emb = model(tokens)["representation"]
            else:
                emb = model(tokens)["representation"]
        if emb.shape[-1] != expected:
            raise ValueError(
                f"rinalmo embedding dim {emb.shape[-1]} != expected {expected} "
                f"(lm_config={lm_config}; update registry.dim / expected_dim)"
            )
        for i, seq in enumerate(chunk):
            L = len(seq)
            if L <= 0:
                continue
            # tokens: [CLS] + residues + [EOS]; take L residue positions
            piece = emb[i, 1 : 1 + L, :].detach().float().cpu().numpy()
            out[start + i, :L, :] = piece

    if np.isnan(out).any():
        raise RuntimeError("rinalmo embeddings contain NaN")
    return out
