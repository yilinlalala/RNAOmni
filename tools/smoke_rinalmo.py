#!/usr/bin/env python3
"""D3 Embedding smoke for RiNALMo (zero-install PYTHONPATH).

Local weights only. Default: smallest present under {code_root}/weights/
(micro → mega → giga). Does not call get_pretrained_model / gdown.

Usage (from repo root, conda myenv3.11) — do not auto-run in CI without weights:
  PYTHONPATH=tools/RiNALMo python tools/smoke_rinalmo.py \\
    --code-root tools/RiNALMo
  python tools/smoke_rinalmo.py --assets skill-library/feature/examples/rinalmo.assets.yaml
  python tools/smoke_rinalmo.py --weight /abs/path/to/rinalmo_micro_pretrained.pt --lm-config micro
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

from pretrained_assets import (  # noqa: E402
    expected_dim_for,
    infer_lm_config,
    parse_assets,
    resolve_code_root,
    resolve_local_weight,
)

DEFAULT_ASSETS = REPO / "skill-library" / "feature" / "examples" / "rinalmo.assets.yaml"
DEFAULT_OUT = REPO / "artifacts" / "pretrained" / "rinalmo" / "smoke_result.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="RiNALMo Embedding smoke (D3)")
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument(
        "--code-root",
        type=str,
        default=None,
        help="Upstream RiNALMo root (overrides assets.code_root).",
    )
    parser.add_argument("--weight", type=Path, default=None, help="Override local .pt")
    parser.add_argument(
        "--lm-config",
        type=str,
        default=None,
        help="micro|mega|giga|nano (default: infer from weight filename)",
    )
    parser.add_argument("--device", default=None, help="cuda|cpu")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--use-flash-attn",
        choices=("auto", "true", "false"),
        default="auto",
        help="auto/false: remap Zenodo Wqkv→std (default; ignores host flash_attn). "
        "true: native FlashAttention (needs compatible flash-attn==2.3.x)",
    )
    args = parser.parse_args()

    assets_path = args.assets
    if not assets_path.is_file():
        cand = (REPO / assets_path).resolve()
        if cand.is_file():
            assets_path = cand
        else:
            print(f"SMOKE_FAIL assets missing: {args.assets}", file=sys.stderr)
            return 1

    assets = parse_assets(assets_path)
    try:
        code_root = resolve_code_root(
            assets, repo=REPO, code_root_override=args.code_root
        )
        if args.weight is not None:
            weight = args.weight.expanduser().resolve()
            if not weight.is_file():
                raise FileNotFoundError(f"--weight not found: {weight}")
        else:
            weight = resolve_local_weight(
                list(assets.get("weight_candidates") or []),
                code_root=code_root,
                repo=REPO,
            )
    except FileNotFoundError as err:
        print(f"SMOKE_FAIL {err}", file=sys.stderr)
        return 1

    lm_config = args.lm_config or infer_lm_config(weight, assets)
    dim_hint = expected_dim_for(lm_config, assets)

    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    if str(REPO / "tools") not in sys.path:
        sys.path.insert(0, str(REPO / "tools"))

    try:
        import torch
        from rinalmo.config import model_config
        from rinalmo.data.alphabet import Alphabet
        from rinalmo.model.model import RiNALMo
        from rinalmo_weights import peek_use_flash_attn
    except ImportError as err:
        print(
            f"SMOKE_FAIL ImportError: {err}\n"
            "Ask human before installing. See tools/RiNALMo/environment.yml / pyproject.toml.",
            file=sys.stderr,
        )
        return 2

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    flash_req: bool | str | None
    if args.use_flash_attn == "auto":
        flash_req = "auto"
    else:
        flash_req = args.use_flash_attn == "true"

    print(f"code_root={code_root}")
    print(f"weight={weight}")
    print(f"lm_config={lm_config}")
    print(f"dim_hint={dim_hint}")
    print(f"device={device}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")

    _torch_load = torch.load

    def _torch_load_compat(*a, **kw):
        kw.setdefault("weights_only", False)
        return _torch_load(*a, **kw)

    torch.load = _torch_load_compat  # type: ignore[assignment]
    t0 = time.time()
    try:
        try:
            raw = torch.load(str(weight), map_location="cpu")
        except Exception as err:
            print(f"SMOKE_FAIL load: {type(err).__name__}: {err}", file=sys.stderr)
            return 3
    finally:
        torch.load = _torch_load  # type: ignore[assignment]

    try:
        use_flash, remapped = peek_use_flash_attn(raw, flash_req)
    except Exception as err:
        print(f"SMOKE_FAIL remap: {type(err).__name__}: {err}", file=sys.stderr)
        return 3

    print(f"use_flash_attn={use_flash} (request={args.use_flash_attn})")

    config = model_config(lm_config)
    config.model.transformer.use_flash_attn = bool(use_flash)
    model = RiNALMo(config)
    alphabet = Alphabet(**config["alphabet"])

    try:
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        missing_critical = [m for m in missing if "rotary_emb" not in m]
        if unexpected or missing_critical:
            raise RuntimeError(
                f"unexpected={list(unexpected)[:8]} "
                f"missing_critical={missing_critical[:8]}"
            )
    except Exception as err:
        print(f"SMOKE_FAIL state_dict: {type(err).__name__}: {err}", file=sys.stderr)
        return 3

    model = model.to(device)
    model.eval()

    seqs = [
        "GGGUGCGAUCAUACCAGCACUAAUGCCCUCCUGGGAAGUCCUCGUGUUGCACCCCU",
        "CCCGGU",
    ]
    tokens = torch.tensor(alphabet.batch_tokenize(seqs), dtype=torch.int64, device=device)

    try:
        with torch.no_grad():
            if device.startswith("cuda"):
                with torch.cuda.amp.autocast():
                    emb = model(tokens)["representation"]
            else:
                emb = model(tokens)["representation"]
    except Exception as err:
        print(f"SMOKE_FAIL forward: {type(err).__name__}: {err}", file=sys.stderr)
        return 4

    if emb.ndim != 3:
        print(
            f"SMOKE_FAIL unexpected emb ndim={emb.ndim} shape={tuple(emb.shape)}",
            file=sys.stderr,
        )
        return 5

    dim = int(emb.shape[-1])
    has_nan = bool(torch.isnan(emb).any().item())
    elapsed = time.time() - t0
    dim_ok = dim == dim_hint

    payload = {
        "status": "ok" if (not has_nan and dim_ok) else "fail",
        "feature_name": str(assets.get("feature_name") or "rinalmo"),
        "model_path": str(weight),
        "code_root": str(code_root),
        "lm_config": lm_config,
        "device": device,
        "dim": dim,
        "dim_hint": dim_hint,
        "emb_shape": list(emb.shape),
        "has_nan": has_nan,
        "n_seqs": int(emb.shape[0]),
        "torch_version": torch.__version__,
        "cuda": torch.version.cuda,
        "elapsed_sec": round(elapsed, 3),
        "use_flash_attn": use_flash,
        "flash_request": args.use_flash_attn,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("---")
    print(f"dim={dim} (hint={dim_hint})")
    print(f"emb_shape={tuple(emb.shape)}")
    print(f"has_nan={has_nan}")
    print(f"elapsed_sec={elapsed:.3f}")
    print(f"wrote={args.out}")

    if has_nan:
        print("SMOKE_FAIL embeddings contain NaN", file=sys.stderr)
        return 6
    if not dim_ok:
        print(
            f"SMOKE_FAIL dim={dim} != hint={dim_hint} for lm_config={lm_config}",
            file=sys.stderr,
        )
        return 7
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
