#!/usr/bin/env python3
"""D3 Embedding smoke for Evo (zero-install local HF dir).

Usage (from repo root, conda myenv3.11):
  PYTHONPATH=tools/Evo python tools/smoke_evo.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ASSETS = REPO / "skill-library" / "feature" / "examples" / "evo.assets.yaml"
DEFAULT_OUT = REPO / "artifacts" / "pretrained" / "evo" / "smoke_result.json"
DEFAULT_MODEL = REPO / "tools" / "Evo" / "model"


def _expand(p: str) -> Path:
    path = Path(os.path.expanduser(p))
    if not path.is_absolute():
        path = (REPO / path).resolve()
    else:
        path = path.resolve()
    return path


def resolve_model_dir(candidates: list[str]) -> Path:
    tried = []
    for c in candidates:
        p = _expand(c)
        tried.append(str(p))
        if p.is_dir() and (p / "config.json").is_file():
            return p
        if p.is_file():
            return p.parent
    if DEFAULT_MODEL.is_dir() and (DEFAULT_MODEL / "config.json").is_file():
        return DEFAULT_MODEL
    raise FileNotFoundError("No local Evo model dir. Tried:\n  - " + "\n  - ".join(tried))


def parse_weight_candidates(assets_path: Path) -> list[str]:
    import re

    text = assets_path.read_text(encoding="utf-8")
    weights: list[str] = []
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
    return weights


def main() -> int:
    parser = argparse.ArgumentParser(description="Evo Embedding smoke (D3)")
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--weight", type=Path, default=None, help="Override model dir")
    parser.add_argument("--device", default=None, help="cuda|cpu (default: cuda if available)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    code_root = REPO / "tools" / "Evo"
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))

    try:
        import torch
    except ImportError as err:
        print(
            "SMOKE_FAIL ImportError: "
            f"{err}\n"
            "Ask human before installing. See tools/Evo/requirements.txt.",
            file=sys.stderr,
        )
        return 2

    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as err:
        print(f"SMOKE_FAIL ImportError: {err}", file=sys.stderr)
        return 2

    weights = parse_weight_candidates(args.assets) if args.assets.is_file() else []
    model_dir = args.weight.resolve() if args.weight else resolve_model_dir(weights)

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"code_root={code_root}")
    print(f"model_dir={model_dir}")
    print(f"device={device}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")

    t0 = time.time()
    try:
        config = AutoConfig.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True
        )
        config.use_cache = False
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            config=config,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
    except Exception as err:
        print(f"SMOKE_FAIL load: {type(err).__name__}: {err}", file=sys.stderr)
        return 3

    model.to(device)
    model.eval()
    backbone = model.backbone

    seqs = [
        "GGGUGCGAUCAUACCAGCACUAAUGCCCUCCUGGGAAGUCCUCGUGUUGCACCCCU",
        "ACGUACGU",
    ]
    seqs = [s.upper().replace("U", "T") for s in seqs]
    ids = [list(s.encode("latin-1")) for s in seqs]
    t_max = max(len(x) for x in ids)
    pad = 1
    batch = [row + [pad] * (t_max - len(row)) for row in ids]
    mask = [[1.0] * len(row) + [0.0] * (t_max - len(row)) for row in ids]
    hid_dtype = next(backbone.parameters()).dtype
    input_ids = torch.tensor(batch, dtype=torch.long, device=device)
    padding_mask = torch.tensor(mask, dtype=hid_dtype, device=device)

    try:
        with torch.no_grad():
            x = backbone.embedding_layer.embed(input_ids)
            x, _ = backbone.stateless_forward(x, padding_mask=padding_mask)
            if backbone.norm is not None:
                x = backbone.norm(x)
        emb = x
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
        "feature_name": "evo",
        "model_path": str(model_dir),
        "device": device,
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
    if dim != 4096:
        print(f"SMOKE_FAIL expected dim=4096 got {dim}", file=sys.stderr)
        return 7
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
