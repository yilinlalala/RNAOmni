#!/usr/bin/env python3
"""Smoke 15.3: store_len < cap, chunk invariance, batch pad-to-max."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from features.builders import onehot as onehot_builder
from features.pipeline import encode_dataframe_from_recipe, encode_sequence_batch


def _recipe_onehot() -> dict:
    return {
        "recipe_id": "smoke_chunk",
        "fusion": "concat",
        "features": [{"name": "onehot", "enabled": True}],
    }


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "input_seq": ["A", "ACGUACGU", "GG", "ACGUACGU"],
            "predicted_target": [0.0, 1.0, 2.0, 3.0],
        }
    )


def _encode(*, chunk: int):
    saved = os.environ.get("RNAOMNI_ENCODE_CHUNK")
    os.environ["RNAOMNI_ENCODE_CHUNK"] = str(chunk)
    try:
        return encode_dataframe_from_recipe(
            _df(),
            sequence_col="input_seq",
            label_col="predicted_target",
            max_seq_len=32,
            alphabet="ACGU",
            recipe=_recipe_onehot(),
        )
    finally:
        if saved is None:
            os.environ.pop("RNAOMNI_ENCODE_CHUNK", None)
        else:
            os.environ["RNAOMNI_ENCODE_CHUNK"] = saved


def test_store_len_not_cap() -> None:
    x, lengths, _, meta = _encode(chunk=64)
    assert meta["n_unique"] == 3
    assert meta["store_len"] == 8, meta
    assert x.shape == (3, 8, 4), x.shape
    np.testing.assert_array_equal(lengths, [1, 8, 2])
    print("store_len OK")


def test_chunk_invariance() -> None:
    x1, _, _, _ = _encode(chunk=1)
    x9, _, _, _ = _encode(chunk=9)
    np.testing.assert_allclose(x1, x9, rtol=0, atol=0)
    oh = onehot_builder.build(["A", "ACGUACGU", "GG"], max_len=8, alphabet="ACGU", params={})
    np.testing.assert_allclose(x1, oh, rtol=0, atol=0)
    print("chunk invariance OK")


def test_batch_pads_to_batch_max() -> None:
    x, lengths = encode_sequence_batch(
        ["AC", "ACGUACGU"],
        max_seq_len=1024,
        alphabet="ACGU",
        recipe=_recipe_onehot(),
    )
    assert x.shape == (2, 8, 4), x.shape
    np.testing.assert_array_equal(lengths, [2, 8])
    print("batch pad OK")


def test_cap_clips_store_len() -> None:
    x, lengths, _, meta = encode_dataframe_from_recipe(
        pd.DataFrame({"input_seq": ["A" * 20], "predicted_target": [0.0]}),
        sequence_col="input_seq",
        label_col="predicted_target",
        max_seq_len=8,
        alphabet="ACGU",
        recipe=_recipe_onehot(),
    )
    assert meta["store_len"] == 8
    assert x.shape == (1, 8, 4)
    assert int(lengths[0]) == 8
    print("cap clip OK")


def main() -> int:
    test_store_len_not_cap()
    test_chunk_invariance()
    test_batch_pads_to_batch_max()
    test_cap_clips_store_len()
    print("smoke_encode_chunk: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
