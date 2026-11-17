#!/usr/bin/env python3
"""D3 Embedding smoke for Evo2 (zero-install PYTHONPATH).

Usage (from repo root, conda myenv3.11):
  PYTHONPATH=tools/Evo2 python tools/smoke_evo2.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "artifacts" / "pretrained" / "evo2" / "smoke_result.json"
DEFAULT_WEIGHT = REPO / "tools" / "Evo2" / "weights" / "evo2_7b.pt"
CODE_ROOT = REPO / "tools" / "Evo2"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evo2 Embedding smoke (D3)")
    parser.add_argument("--weight", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))

    try:
        import torch
        from evo2 import Evo2
    except ImportError as err:
        print(
            f"SMOKE_FAIL ImportError: {err}\n"
            "Ask human before installing. See tools/Evo2/requirements.txt.",
            file=sys.stderr,
        )
        return 2

    weight = args.weight.resolve()
    if not weight.is_file():
        print(f"SMOKE_FAIL missing weight: {weight}", file=sys.stderr)
        return 3
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"code_root={CODE_ROOT}")
    print(f"weight={weight}")
    print(f"device={device}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")

    t0 = time.time()
    try:
        evo2_model = Evo2("evo2_7b", local_path=str(weight))
        evo2_model.model.eval()
    except Exception as err:
        print(f"SMOKE_FAIL load: {type(err).__name__}: {err}", file=sys.stderr)
        return 3

    seq = "GGGUGCGAUCAUACCAGCACUAAUGCCCUCCUGGGAAGUCCUCGUGUUGCACCCCU".upper().replace("U", "T")
    ids = list(seq.encode("utf-8"))
    encode_device = "cuda:0" if str(device).startswith("cuda") else device
    input_ids = torch.tensor([ids], dtype=torch.long, device=encode_device)
    try:
        with torch.no_grad():
            _logits, embeddings = evo2_model(
                input_ids, return_embeddings=True, layer_names=["norm"]
            )
        emb = embeddings["norm"]
    except Exception as err:
        print(f"SMOKE_FAIL forward: {type(err).__name__}: {err}", file=sys.stderr)
        return 4

    if emb.ndim != 3:
        print(f"SMOKE_FAIL unexpected emb ndim={emb.ndim} shape={tuple(emb.shape)}", file=sys.stderr)
        return 5
    dim = int(emb.shape[-1])
    has_nan = bool(torch.isnan(emb).any().item())
    elapsed = time.time() - t0
    payload = {
        "status": "ok" if not has_nan else "nan",
        "feature_name": "evo2",
        "model_path": str(weight),
        "device": device,
        "dim": dim,
        "emb_shape": list(emb.shape),
        "has_nan": has_nan,
        "n_seqs": int(emb.shape[0]),
        "torch_version": torch.__version__,
        "cuda": torch.version.cuda,
        "elapsed_sec": round(elapsed, 3),
        "code_root": str(CODE_ROOT),
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
    if dim != 4096:
        print(f"SMOKE_FAIL expected dim=4096 got {dim}", file=sys.stderr)
        return 7
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
