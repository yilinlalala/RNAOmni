#!/usr/bin/env python3
"""Smoke recipe-tunable kmer k, fusion, fusion_include, rna_fm.repr_layer (no train).

Usage (conda myenv3.11):
  python tools/smoke_kmer_fusion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.builders import kmer as kmer_builder
from features.builders import onehot as onehot_builder
from features.pipeline import (
    encode_dataframe_from_recipe,
    load_active_recipe,
    load_feature_registry,
    recipe_cache_key,
    resolve_input_dim,
)

SEQS = ["ACGUACGU", "GGGA", "U" * 12]
MAX_LEN = 16
ALPHABET = "ACGU"


def _df() -> pd.DataFrame:
    return pd.DataFrame({"input_seq": SEQS, "predicted_target": [0.0, 1.0, 2.0]})


def _recipe(*, fusion: str = "concat", kmer_k: int | None = None) -> dict:
    kmer_item: dict = {"name": "kmer", "enabled": True}
    if kmer_k is not None:
        kmer_item["params"] = {"k": kmer_k}
    features = [
        {"name": "onehot", "enabled": True},
        {"name": "length_frac", "enabled": True},
        kmer_item,
    ]
    return {"recipe_id": "smoke_kmer_fusion", "fusion": fusion, "features": features}


def _encode(recipe: dict):
    return encode_dataframe_from_recipe(
        _df(),
        sequence_col="input_seq",
        label_col="predicted_target",
        max_seq_len=MAX_LEN,
        alphabet=ALPHABET,
        recipe=recipe,
    )


def _assert_raises(fn, match: str) -> None:
    try:
        fn()
    except (ValueError, KeyError) as exc:
        if match not in str(exc):
            raise AssertionError(f"expected {match!r} in {exc!r}") from exc
        return
    raise AssertionError(f"expected error containing {match!r}")


def main() -> None:
    registry = load_feature_registry()
    kmer_entry = next(e for e in registry["features"] if e["name"] == "kmer")
    assert "k" in list(kmer_entry.get("agent_tunable") or []), kmer_entry
    assert int(kmer_entry["dim"]) == 64

    # Bit-compat: omit params → k=3, concat, C_kmer=64.
    x, _, _, meta = _encode(_recipe())
    assert meta["fusion"] == "concat"
    assert meta["channel_dims"] == [4, 1, 64], meta["channel_dims"]
    assert meta["input_dim"] == 69, meta["input_dim"]
    assert x.shape == (len(SEQS), 12, 69), x.shape
    assert resolve_input_dim(_recipe(), registry) == 69

    # k=2 → C_kmer=16; input_dim drops by 48 vs default 69.
    rec_k2 = _recipe(kmer_k=2)
    x2, _, _, meta2 = _encode(rec_k2)
    assert meta2["channel_dims"] == [4, 1, 16], meta2["channel_dims"]
    assert meta2["input_dim"] == 21, meta2["input_dim"]
    assert resolve_input_dim(rec_k2, registry) == 21
    assert x2.shape[-1] == 21

    # k=4 → C_kmer=256.
    rec_k4 = _recipe(kmer_k=4)
    _, _, _, meta4 = _encode(rec_k4)
    assert meta4["channel_dims"][-1] == 256, meta4["channel_dims"]
    assert meta4["input_dim"] == 261, meta4["input_dim"]
    assert resolve_input_dim(rec_k4, registry) == 261

    # k=5 forbidden.
    _assert_raises(lambda: _encode(_recipe(kmer_k=5)), "kmer k must be")
    _assert_raises(lambda: resolve_input_dim(_recipe(kmer_k=5), registry), "kmer k must be")

    # Unknown param key on a non-tunable feature.
    bad = _recipe()
    bad["features"][0]["params"] = {"k": 2}
    _assert_raises(lambda: _encode(bad), "not in agent_tunable")

    # add / mean require equal C: onehot(4)+length_frac(1)+kmer(64) fails.
    _assert_raises(lambda: _encode(_recipe(fusion="add")), "fusion=add needs equal C")
    _assert_raises(lambda: resolve_input_dim(_recipe(fusion="add"), registry), "fusion=add needs equal C")

    # add / mean succeed when C aligns: onehot + kmer k=1 (both 4).
    aligned = {
        "recipe_id": "smoke_add",
        "fusion": "add",
        "features": [
            {"name": "onehot", "enabled": True},
            {"name": "kmer", "enabled": True, "params": {"k": 1}},
        ],
    }
    x_add, _, _, meta_add = _encode(aligned)
    assert meta_add["input_dim"] == 4, meta_add
    assert x_add.shape[-1] == 4
    assert resolve_input_dim(aligned, registry) == 4
    oh = onehot_builder.build(SEQS, max_len=x_add.shape[1], alphabet=ALPHABET, params={})
    km = kmer_builder.build(SEQS, max_len=x_add.shape[1], alphabet=ALPHABET, params={"k": 1})
    np.testing.assert_allclose(x_add, oh + km, rtol=0, atol=0)

    aligned_mean = dict(aligned)
    aligned_mean["fusion"] = "mean"
    x_mean, _, _, meta_mean = _encode(aligned_mean)
    assert meta_mean["input_dim"] == 4
    np.testing.assert_allclose(x_mean, (oh + km) / 2.0, rtol=1e-6, atol=1e-6)

    _assert_raises(
        lambda: encode_dataframe_from_recipe(
            _df(),
            sequence_col="input_seq",
            label_col="predicted_target",
            max_seq_len=MAX_LEN,
            alphabet=ALPHABET,
            recipe={"recipe_id": "bad", "fusion": "attn", "features": [{"name": "onehot", "enabled": True}]},
        ),
        "Unsupported fusion",
    )

    # fusion_include: subset concat; omitted = all enabled.
    subset = _recipe()
    subset["fusion_include"] = ["onehot", "kmer"]
    x_sub, _, _, meta_sub = _encode(subset)
    assert meta_sub["enabled_features"] == ["onehot", "kmer"], meta_sub["enabled_features"]
    assert meta_sub["channel_dims"] == [4, 64], meta_sub["channel_dims"]
    assert meta_sub["input_dim"] == 68, meta_sub["input_dim"]
    assert meta_sub["fusion_include"] == ["onehot", "kmer"]
    assert resolve_input_dim(subset, registry) == 68
    assert x_sub.shape[-1] == 68
    assert recipe_cache_key(subset) != recipe_cache_key(_recipe())

    omitted = _recipe()
    _, _, _, meta_all = _encode(omitted)
    assert meta_all["fusion_include"] is None
    assert meta_all["input_dim"] == 69

    bad_unreg = _recipe()
    bad_unreg["fusion_include"] = ["onehot", "not_a_registered_feature"]
    _assert_raises(lambda: _encode(bad_unreg), "not in features/registry.yaml")
    _assert_raises(lambda: resolve_input_dim(bad_unreg, registry), "not in features/registry.yaml")

    bad_off = _recipe()
    bad_off["features"].append({"name": "struct_dotbracket", "enabled": False})
    bad_off["fusion_include"] = ["onehot", "struct_dotbracket"]
    _assert_raises(lambda: _encode(bad_off), "is not enabled")
    _assert_raises(lambda: resolve_input_dim(bad_off, registry), "is not enabled")

    # rna_fm: repr_layer is tunable; dim stays 640; illegal layer fails without encode.
    rna_fm_entry = next(e for e in registry["features"] if e["name"] == "rna_fm")
    assert "repr_layer" in list(rna_fm_entry.get("agent_tunable") or []), rna_fm_entry
    rinalmo_entry = next(e for e in registry["features"] if e["name"] == "rinalmo")
    assert list(rinalmo_entry.get("agent_tunable") or []) == [], rinalmo_entry

    rec_layer = {
        "recipe_id": "smoke_repr",
        "fusion": "concat",
        "features": [{"name": "rna_fm", "enabled": True, "params": {"repr_layer": 6}}],
    }
    assert resolve_input_dim(rec_layer, registry) == 640
    bad_layer = {
        "recipe_id": "smoke_repr_bad",
        "fusion": "concat",
        "features": [{"name": "rna_fm", "enabled": True, "params": {"repr_layer": 13}}],
    }
    _assert_raises(lambda: resolve_input_dim(bad_layer, registry), "repr_layer must be")

    # Active recipe on disk still concat; dim matches enabled registry defaults.
    active = load_active_recipe()
    assert active.get("fusion", "concat") == "concat"
    active_dim = resolve_input_dim(active, registry)
    expected = 0
    enabled = {item["name"] for item in active["features"] if item.get("enabled", True)}
    for entry in registry["features"]:
        if entry["name"] in enabled:
            expected += int(entry["dim"])
    assert active_dim == expected, (active_dim, expected, enabled)

    print("smoke_kmer_fusion: OK")
    print(f"  default concat input_dim=69; k=2 → 21; k=4 → 261; add/mean aligned C=4")
    print(f"  fusion_include [onehot,kmer] → 68; rna_fm repr_layer 0..12")
    print(f"  active.yaml concat input_dim={active_dim}")


if __name__ == "__main__":
    main()
