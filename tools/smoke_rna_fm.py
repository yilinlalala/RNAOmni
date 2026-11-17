#!/usr/bin/env python3
"""D3 Embedding smoke for RNA-FM (zero-install PYTHONPATH).

Usage (from repo root, conda myenv3.11):
  PYTHONPATH=tools/RNA-FM python tools/smoke_rna_fm.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ASSETS = REPO / "skill-library" / "feature" / "examples" / "rna_fm.assets.yaml"
DEFAULT_OUT = REPO / "artifacts" / "pretrained" / "rna_fm" / "smoke_result.json"


def _expand(p: str) -> Path:
    path = Path(os.path.expanduser(p))
    if not path.is_absolute():
        path = (REPO / path).resolve()
    else:
        path = path.resolve()
    return path


def resolve_weight(candidates: list[str]) -> Path:
    tried = []
    for c in candidates:
        p = _expand(c)
        tried.append(str(p))
        if p.is_file():
            return p
    raise FileNotFoundError("No weight found. Tried:\n  - " + "\n  - ".join(tried))


def parse_weight_candidates(assets_path: Path) -> tuple[list[str], int]:
    import re

    text = assets_path.read_text(encoding="utf-8")
    weights: list[str] = []
    repr_layer = 12
    in_weights = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.strip() == "weight_candidates:":
            in_weights = True
            continue
        if in_weights:
            m = re.match(r"^\s+-\s+path:\s*(.+)$", line)
            if m:
                weights.append(m.group(1).strip().strip("'\""))
                continue
            if re.match(r"^[a-zA-Z_]", line):
                in_weights = False
            else:
                continue
        m = re.match(r"^repr_layer:\s*(\d+)\s*$", line)
        if m:
            repr_layer = int(m.group(1))
    return weights, repr_layer


def main() -> int:
    parser = argparse.ArgumentParser(description="RNA-FM Embedding smoke (D3)")
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--weight", type=Path, default=None, help="Override weight path")
    parser.add_argument("--device", default=None, help="cuda|cpu (default: cuda if available)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repr-layer", type=int, default=None)
    args = parser.parse_args()

    code_root = REPO / "tools" / "RNA-FM"
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))

    # fm/__init__.py does `from .downstream import *`, which imports ptflops only for
    # FLOPs tooling — unused for Embedding. Stub instead of installing (zero-install).
    if "ptflops" not in sys.modules:
        import types

        _stub = types.ModuleType("ptflops")
        _stub.get_model_complexity_info = lambda *a, **k: (0, 0)
        sys.modules["ptflops"] = _stub

    try:
        import torch
        import fm  # noqa: F401 — needs PYTHONPATH/tools/RNA-FM
    except ImportError as err:
        print(
            "SMOKE_FAIL ImportError: "
            f"{err}\n"
            "Ask human before installing. See tools/RNA-FM/setup.py and environment.yml.",
            file=sys.stderr,
        )
        return 2

    weights, assets_layer = parse_weight_candidates(args.assets)
    repr_layer = args.repr_layer if args.repr_layer is not None else assets_layer
    weight = args.weight.resolve() if args.weight else resolve_weight(weights)

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"code_root={code_root}")
    print(f"weight={weight}")
    print(f"device={device}")
    print(f"repr_layer={repr_layer}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")

    # Torch>=2.6 defaults weights_only=True; RNA-FM ckpt needs Namespace unpickle.
    # Trusted local checkpoint — do not change upstream RNA-FM sources.
    _torch_load = torch.load

    def _torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _torch_load(*args, **kwargs)

    torch.load = _torch_load_compat  # type: ignore[assignment]

    t0 = time.time()
    try:
        model, alphabet = fm.pretrained.rna_fm_t12(str(weight))
    except Exception as err:
        print(f"SMOKE_FAIL load: {type(err).__name__}: {err}", file=sys.stderr)
        return 3
    finally:
        torch.load = _torch_load  # type: ignore[assignment]

    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    data = [
        ("RNA1", "GGGUGCGAUCAUACCAGCACUAAUGCCCUCCUGGGAAGUCCUCGUGUUGCACCCCU"),
        ("RNA2", "GGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCA"),
    ]
    _labels, _strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    try:
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[repr_layer])
        emb = results["representations"][repr_layer]
    except Exception as err:
        print(f"SMOKE_FAIL forward: {type(err).__name__}: {err}", file=sys.stderr)
        return 4

    # token embeddings: [B, T, C]; C is channel dim for registry
    if emb.ndim != 3:
        print(f"SMOKE_FAIL unexpected emb ndim={emb.ndim} shape={tuple(emb.shape)}", file=sys.stderr)
        return 5
    dim = int(emb.shape[-1])
    has_nan = bool(torch.isnan(emb).any().item())
    elapsed = time.time() - t0

    payload = {
        "status": "ok" if not has_nan else "nan",
        "feature_name": "rna_fm",
        "model_path": str(weight),
        "device": device,
        "repr_layer": repr_layer,
        "dim": dim,
        "emb_shape": list(emb.shape),
        "has_nan": has_nan,
        "n_seqs": int(emb.shape[0]),
        "torch_version": torch.__version__,
        "cuda": torch.version.cuda,
        "elapsed_sec": round(elapsed, 3),
        "code_root": str(code_root),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("---")
    print(f"dim={dim}")
    print(f"emb_shape={tuple(emb.shape)}")
    print(f"has_nan={has_nan}")
    print(f"elapsed_sec={elapsed:.3f}")
    print(f"wrote={args.out}")

    if has_nan:
        print("SMOKE_FAIL embeddings contain NaN", file=sys.stderr)
        return 6
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
