#!/usr/bin/env python3
"""Load RiNALMo local checkpoints; adapt flash ↔ non-flash parameter names.

Official Zenodo pretrained weights use FlashMultiHeadSelfAttention (Wqkv).
Building with use_flash_attn=False expects MultiHeadAttention (to_q/to_k/to_v).
"""

from __future__ import annotations

from typing import Any

import torch


def is_flash_state_dict(state: dict[str, Any]) -> bool:
    return any(".mh_attn.Wqkv." in k for k in state)


def is_std_state_dict(state: dict[str, Any]) -> bool:
    return any(".mh_attn.mh_attn.to_q." in k for k in state)


def flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def convert_flash_state_to_std(state: dict[str, Any]) -> dict[str, Any]:
    """Map FlashMultiHeadSelfAttention keys → MultiHeadSelfAttention keys."""
    out: dict[str, Any] = {}
    for key, val in state.items():
        if key.endswith(".mh_attn.Wqkv.weight"):
            # Wqkv: [3 * dim, dim] → to_q / to_k / to_v
            q, k, v = torch.chunk(val, 3, dim=0)
            prefix = key[: -len("Wqkv.weight")]
            out[prefix + "mh_attn.to_q.weight"] = q.contiguous()
            out[prefix + "mh_attn.to_k.weight"] = k.contiguous()
            out[prefix + "mh_attn.to_v.weight"] = v.contiguous()
            continue
        if key.endswith(".mh_attn.Wqkv.bias"):
            q, k, v = torch.chunk(val, 3, dim=0)
            prefix = key[: -len("Wqkv.bias")]
            out[prefix + "mh_attn.to_q.bias"] = q.contiguous()
            out[prefix + "mh_attn.to_k.bias"] = k.contiguous()
            out[prefix + "mh_attn.to_v.bias"] = v.contiguous()
            continue
        if ".mh_attn.out_proj." in key and ".mh_attn.mh_attn.out_proj." not in key:
            # transformer.blocks.i.mh_attn.out_proj.* → ...mh_attn.mh_attn.out_proj.*
            out[key.replace(".mh_attn.out_proj.", ".mh_attn.mh_attn.out_proj.", 1)] = val
            continue
        out[key] = val
    return out


def unwrap_state_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict) and "state_dict" in raw and not any(
        k.startswith("embedding") or k.startswith("transformer") for k in raw
    ):
        raw = raw["state_dict"]
    if not isinstance(raw, dict):
        raise TypeError(f"expected state_dict mapping, got {type(raw)}")
    return raw


def resolve_use_flash_attn(
    state: dict[str, Any],
    requested: bool | str | None,
) -> tuple[bool, dict[str, Any]]:
    """Return (use_flash_attn, maybe-converted state).

    requested:
      True — require flash architecture (needs compatible flash_attn)
      False / None / "auto" — prefer non-flash; remap Wqkv→to_q/k/v if needed

    Note: "auto" intentionally does **not** enable flash even if flash_attn is
    importable — host flash_attn versions often mismatch RiNALMo's RotaryEmbedding
    kwargs (e.g. pos_idx_in_fp32). Opt in with use_flash_attn=true.
    """
    flash_ckpt = is_flash_state_dict(state)
    std_ckpt = is_std_state_dict(state)

    if requested is True or requested == "true":
        if std_ckpt and not flash_ckpt:
            raise RuntimeError(
                "Checkpoint is non-flash (to_q/to_k/to_v) but use_flash_attn=true"
            )
        if not flash_attn_available():
            raise ImportError(
                "use_flash_attn=true requires flash_attn; "
                "or use auto/false to remap Zenodo weights to std attention"
            )
        return True, state

    # auto / false / None → non-flash path
    if flash_ckpt:
        return False, convert_flash_state_to_std(state)
    return False, state


def load_rinalmo_state_dict(
    model: torch.nn.Module,
    raw: Any,
    *,
    use_flash_attn: bool | str | None = "auto",
) -> bool:
    """Load weights into model; returns the use_flash_attn flag actually used for remap.

    Caller must build `model` with the same use_flash_attn this function returns.
    Prefer: peek checkpoint → resolve_use_flash_attn → build model → load.
    """
    state = unwrap_state_dict(raw)
    use_flash, state = resolve_use_flash_attn(state, use_flash_attn)
    # rotary inv_freq is a buffer re-initialized in RotaryPositionEmbedding
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_critical = [
        m
        for m in missing
        if not m.endswith("rotary_emb.inv_freq")
        and "rotary_emb" not in m
    ]
    if unexpected:
        raise RuntimeError(
            "Unexpected keys after remap: " + ", ".join(unexpected[:8])
            + ("..." if len(unexpected) > 8 else "")
        )
    if missing_critical:
        raise RuntimeError(
            "Missing critical keys after remap: "
            + ", ".join(missing_critical[:8])
            + ("..." if len(missing_critical) > 8 else "")
        )
    return use_flash


def peek_use_flash_attn(
    raw: Any,
    requested: bool | str | None = "auto",
) -> tuple[bool, dict[str, Any]]:
    """Decide architecture + remapped state before constructing the model."""
    state = unwrap_state_dict(raw)
    return resolve_use_flash_attn(state, requested)
