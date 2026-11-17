#!/usr/bin/env python3
"""Smoke: forcompare is always the given train/val/test (flat or fold0).

Usage:
  python tools/smoke_compare_layout.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.pop("COMPARE_FULL", None)
os.environ.pop("COMPARE_NO_SAMPLE", None)
os.environ.pop("RNAOMNI_DATA_MODE", None)


def _load(root: str, fold: int | None = None):
    import prepare

    prepare._DATA_CACHE = None
    info = prepare.apply_dataset_route(
        compare_root=str(REPO / root),
        use_compare=True,
        require_splits=True,
        fold_ids=[fold] if fold is not None else None,
    )
    df = prepare.load_compare_splits_dataframe()
    return prepare, info, df


def test_classification_full_flat() -> None:
    prepare, info, df = _load("dataset/forcompare/Class_88fam_200%noise")
    assert info["layout"] == "flat"
    assert info["task_type"] == "classification"
    assert info["csv_name"] == "compare_Class_88fam_200%noise"
    assert not info["csv_name"].endswith("_explore")
    assert prepare.compare_sampling_on() is False
    assert set(df["split"]) == {"train", "val", "test"}
    assert int(df["predicted_target"].nunique()) == 88
    print("classification flat full OK", info["csv_name"], len(df))


def test_classification_sample_flat() -> None:
    import prepare as prep

    prep.apply_data_mode("sample")
    try:
        prepare, info, df = _load("dataset/forcompare/Class_88fam_200%noise")
        assert prepare.compare_sampling_on() is True
        assert info["csv_name"] == "compare_Class_88fam_200%noise"
        assert not info["csv_name"].endswith("_explore")
        assert int(df["predicted_target"].nunique()) == 88
        k_train = prepare.resolve_compare_per_class_train()
        k_eval = prepare.resolve_compare_per_class_eval()
        assert k_train == 10 and k_eval == 10
        for split_name, g in df.groupby("split"):
            k = k_train if split_name == "train" else k_eval
            assert int(g["predicted_target"].value_counts().max()) <= int(k)
        print("classification flat sample OK", info["csv_name"], len(df))
    finally:
        prep.apply_data_mode("full")
        os.environ.pop("RNAOMNI_DATA_MODE", None)


def test_regression_fold_full() -> None:
    prepare, info, df = _load("dataset/forcompare/Reg_TE_Muscle", fold=0)
    assert info["layout"] == "fold"
    assert info["task_type"] == "regression"
    assert not info["csv_name"].endswith("_explore")
    assert prepare.compare_sampling_on() is False
    counts = df["split"].value_counts().to_dict()
    assert counts["train"] == 1006
    assert counts["val"] == 125
    assert counts["test"] == 126
    val_path = prepare.compare_split_paths(0)["val"]
    assert val_path.endswith("val.csv"), val_path
    print("regression fold0 full OK", info["csv_name"], counts)


def test_regression_fold_default_all_ten() -> None:
    prepare, info, df = _load("dataset/forcompare/Reg_TE_Muscle")
    assert info["layout"] == "fold"
    assert info["fold_ids"] == list(range(10)), info["fold_ids"]
    assert int(df["compare_fold"].nunique()) == 10
    assert set(df["compare_fold"].astype(int)) == set(range(10))
    assert prepare.search_fold_budget_sec(10, use_compare_splits=True) == 60
    assert prepare.search_fold_budget_sec(5, use_compare_splits=False) == 12
    print("regression 10-fold default OK", info["fold_ids"], len(df))


def test_mrl_flat() -> None:
    prepare, info, df = _load("dataset/forcompare/Reg_MRL")
    assert info["layout"] == "flat"
    assert info["task_type"] == "regression"
    assert not info["csv_name"].endswith("_explore")
    assert prepare.compare_sampling_on() is False
    counts = df["split"].value_counts().to_dict()
    assert counts["train"] == 76319
    assert counts["val"] == 7600
    assert counts["test"] == 7600
    assert "input_seq" in df.columns
    assert "predicted_target" in df.columns
    print("regression MRL full OK", info["csv_name"], counts)


def main() -> int:
    test_classification_full_flat()
    test_classification_sample_flat()
    test_regression_fold_full()
    test_regression_fold_default_all_ten()
    test_mrl_flat()
    print("smoke_compare_layout: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
