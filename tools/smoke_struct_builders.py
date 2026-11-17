#!/usr/bin/env python3
"""Smoke-test L3 ViennaRNA structure builders (myenv3.11)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from features.builders import struct_bpp, struct_dotbracket, struct_thermo
from features.builders._vienna import clear_fold_cache, fold_sequence
from features.pipeline import load_feature_registry, resolve_input_dim
from features._yaml import load_yaml


SEQS = [
    "GGGAAAUCC",
    "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA",
    "ACGU" * 20,
]
MAX_LEN = 128


def main() -> None:
    import RNA  # noqa: F401

    clear_fold_cache()
    r = fold_sequence(SEQS[1])
    assert len(r.structure) == len(SEQS[1])
    assert r.ensemble_gap >= -1e-3  # numerical slack

    t = struct_thermo.build(SEQS, max_len=MAX_LEN)
    d = struct_dotbracket.build(SEQS, max_len=MAX_LEN)
    b = struct_bpp.build(SEQS, max_len=MAX_LEN)
    assert t.shape == (3, MAX_LEN, 3)
    assert d.shape == (3, MAX_LEN, 3)
    assert b.shape == (3, MAX_LEN, 2)
    assert np.isfinite(t).all() and np.isfinite(b).all()
    # Dot-bracket one-hot sums to 1 on valid sites of folded seq
    L = len(SEQS[1])
    assert np.allclose(d[1, :L].sum(axis=-1), 1.0)

    reg = load_feature_registry()
    names = {e["name"] for e in reg["features"]}
    for name in ("struct_thermo", "struct_dotbracket", "struct_bpp"):
        assert name in names, name

    example = ROOT / "skill-library/feature/examples/active.with_struct_thermo.yaml"
    recipe = load_yaml(example)
    dim = resolve_input_dim(recipe=recipe, registry=reg)
    assert dim == 7, dim  # onehot 4 + thermo 3

    print("smoke_struct_builders: OK")
    print(f"  tRNA-like MFE={r.mfe:.3f} gap={r.ensemble_gap:.3f} meanH={r.mean_positional_entropy:.3f}")
    print(f"  example recipe input_dim={dim}")


if __name__ == "__main__":
    main()
