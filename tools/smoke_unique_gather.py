#!/usr/bin/env python3
"""Smoke 15.1: unique seq encode + per-row labels; gather shares features."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from features.builders import onehot as onehot_builder
from features.pipeline import encode_dataframe_from_recipe, unique_sequence_index
from rnaomni.prepare.data import gather_row_features


def test_index_normalizes() -> None:
    unique, row_to_uid = unique_sequence_index(["ACGU", "acgu", "GGGG", "ACGT"])
    assert unique == ["ACGU", "GGGG"], unique
    assert list(row_to_uid) == [0, 0, 1, 0], list(row_to_uid)
    print("seq_id normalize OK")


def test_encode_unique_keeps_row_labels() -> None:
    df = pd.DataFrame(
        {
            "input_seq": ["ACGU", "acgu", "GGGG", "ACGU"],
            "predicted_target": [0.1, 0.2, 0.3, 0.4],
        }
    )
    recipe = {
        "recipe_id": "smoke_unique",
        "fusion": "concat",
        "features": [{"name": "onehot", "enabled": True}],
    }
    x, lengths, labels, meta = encode_dataframe_from_recipe(
        df,
        sequence_col="input_seq",
        label_col="predicted_target",
        max_seq_len=8,
        alphabet="ACGU",
        recipe=recipe,
    )
    assert meta["n_rows"] == 4
    assert meta["n_unique"] == 2
    assert x.shape == (2, 4, 4), x.shape
    assert lengths.shape == (2,)
    np.testing.assert_allclose(labels, [0.1, 0.2, 0.3, 0.4])
    uid = meta["row_to_uid"]
    assert uid[0] == uid[1] == uid[3]
    assert uid[2] != uid[0]
    np.testing.assert_array_equal(x[uid[0]], x[uid[1]])
    assert int(meta["store_len"]) == 4
    oh = onehot_builder.build(["ACGU", "GGGG"], max_len=4, alphabet="ACGU", params={})
    np.testing.assert_allclose(x, oh, rtol=0, atol=0)
    print("encode unique + row labels OK")


def test_gather_row_features() -> None:
    x_u = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    lengths_u = torch.tensor([1, 2, 3])
    row_to_uid = torch.tensor([2, 0, 2, 1], dtype=torch.long)
    labels = torch.tensor([10.0, 20.0, 30.0, 40.0])
    data = {"x": x_u, "lengths": lengths_u, "row_to_uid": row_to_uid, "labels": labels}
    x, lengths = gather_row_features(data, [0, 1, 2, 3])
    np.testing.assert_allclose(x.reshape(-1).numpy(), [3.0, 1.0, 3.0, 2.0])
    np.testing.assert_array_equal(lengths.numpy(), [3, 1, 3, 2])
    y = labels.index_select(0, torch.tensor([0, 1, 2, 3]))
    np.testing.assert_allclose(y.numpy(), [10.0, 20.0, 30.0, 40.0])
    print("gather_row_features OK")


def main() -> int:
    test_index_normalizes()
    test_encode_unique_keeps_row_labels()
    test_gather_row_features()
    print("smoke_unique_gather: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
