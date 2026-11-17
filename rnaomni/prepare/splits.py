"""Clean, task inference, split assignment, and data cards."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import rnaomni.prepare.route as _route
from rnaomni.paths import DATA_CARD_DIR, SPLIT_DIR
from rnaomni.prepare.constants import (
    ALPHABET,
    CLS_MMSEQS_COVERAGE,
    CLS_MMSEQS_MIN_SEQ_ID,
    CLS_TEST_FRAC,
    CLS_TRAIN_FRAC,
    CLS_VAL_FRAC,
    DATASET_DIR,
    FOLD_RANDOM_STATE,
    FOLD_TEST,
    FOLD_TRAIN,
    FOLD_VAL,
    LABEL_COL,
    MAX_SEQ_LEN,
    N_FOLDS,
    PROJECT_DIR,
    SEQUENCE_COL,
    TEST_FRACTION,
)


def infer_task_type(labels) -> str:
    """
    Infer regression vs classification from predicted_target values.

    Classification if labels are integer-valued and cardinality is small
    relative to n (or labels form contiguous 0..C-1 with C <= 1000).
    """
    arr = np.asarray(labels, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return "regression"
    rounded = np.round(arr)
    if not np.allclose(arr, rounded, equal_nan=False):
        return "regression"
    ints = rounded.astype(np.int64)
    uniq = np.unique(ints)
    n_unique = int(len(uniq))
    n = int(len(ints))
    if n_unique <= 1:
        return "regression"
    if n_unique <= 1000 and (n_unique / max(n, 1)) <= 0.05:
        return "classification"
    if uniq.min() == 0 and uniq.max() == n_unique - 1 and n_unique <= 1000:
        return "classification"
    return "regression"


def classification_num_classes(labels) -> int:
    """
    Infer output head size from predicted_target class ids.

    Expects 0-based integer labels. Returns max(label)+1 so CrossEntropy
    works with the raw ids (gaps are allowed; unused logits are fine).
    Never hardcode a dataset-specific class count (e.g. 88).
    """
    ints = np.asarray(labels, dtype=np.int64).ravel()
    if ints.size == 0:
        raise ValueError("Cannot infer num_classes from empty labels")
    if ints.min() < 0:
        raise ValueError(
            f"Classification labels must be >= 0 for CrossEntropy, got min={ints.min()}"
        )
    n_unique = int(np.unique(ints).size)
    num_classes = int(ints.max()) + 1
    if n_unique != num_classes:
        # Non-contiguous ids: still size the head by max+1 (label values are indices).
        print(
            f"Data: label ids are non-contiguous "
            f"(n_unique={n_unique}, num_classes=max+1={num_classes})"
        )
    return num_classes


def _csv_path(csv_name=None):
    if csv_name is None:
        csv_name = _route.ACTIVE_CSV
    return os.path.join(DATASET_DIR, csv_name)


def _csv_stem(csv_name=None):
    if csv_name is None:
        csv_name = _route.ACTIVE_CSV
    return os.path.splitext(os.path.basename(csv_name))[0]


def _normalize_sequence(seq: str) -> str:
    """Uppercase and map T→U; does not drop illegal bases."""
    return seq.upper().replace("T", "U")


def _is_legal_sequence(seq: str, alphabet: str = ALPHABET) -> bool:
    allowed = set(alphabet)
    return bool(seq) and all(ch in allowed for ch in seq)


def load_dataset(csv_name=None):
    """Load active CSV, validate columns, drop rows with empty sequences."""
    if csv_name is None:
        csv_name = _route.ACTIVE_CSV
    path = _csv_path(csv_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    for col in (SEQUENCE_COL, LABEL_COL):
        if col not in df.columns:
            raise ValueError(f"Missing required column {col!r} in {path}")

    df = df.dropna(subset=[SEQUENCE_COL, LABEL_COL]).copy()
    df[SEQUENCE_COL] = df[SEQUENCE_COL].astype(str)
    df = df[df[SEQUENCE_COL].str.len() > 0].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"No valid rows left after loading {path}")
    return df


def clean_dataset(df, alphabet=ALPHABET, task_type: str | None = None):
    """
    Normalize sequences and collapse identical-sequence rows.

    Order: T→U normalize → illegal bases → drop non-finite labels →
    collapse identical sequences (mean for regression, first/mode for classification).
    Returns (cleaned_df, filter_stats, task_type).
    """
    n_raw = len(df)
    out = df.copy()
    out[SEQUENCE_COL] = out[SEQUENCE_COL].astype(str).map(_normalize_sequence)

    legal_mask = out[SEQUENCE_COL].map(lambda s: _is_legal_sequence(s, alphabet))
    n_illegal = int((~legal_mask).sum())
    out = out.loc[legal_mask].reset_index(drop=True)

    out[LABEL_COL] = pd.to_numeric(out[LABEL_COL], errors="coerce")
    finite_mask = np.isfinite(out[LABEL_COL].to_numpy(dtype=np.float64))
    n_nonfinite = int((~finite_mask).sum())
    out = out.loc[finite_mask].reset_index(drop=True)

    if task_type is None:
        task_type = infer_task_type(out[LABEL_COL].to_numpy())

    n_before_collapse = len(out)
    n_multi_seq = int((out.groupby(SEQUENCE_COL, sort=False).size() > 1).sum())
    agg = {col: "first" for col in out.columns if col not in (SEQUENCE_COL, LABEL_COL)}
    if task_type == "classification":
        # Exact duplicates: keep first label (no conflicting labels expected).
        agg[LABEL_COL] = "first"
    else:
        agg[LABEL_COL] = "mean"
    out = (
        out.groupby(SEQUENCE_COL, sort=False, as_index=False)
        .agg(agg)
        .reset_index(drop=True)
    )
    n_collapsed = n_before_collapse - len(out)

    if task_type == "classification":
        out[LABEL_COL] = out[LABEL_COL].round().astype(np.int64)
    else:
        out[LABEL_COL] = out[LABEL_COL].astype(np.float64)

    stats = {
        "n_raw": n_raw,
        "n_illegal_base": n_illegal,
        "n_nonfinite_label": n_nonfinite,
        "n_seq_groups_averaged": n_multi_seq,
        "n_rows_collapsed": n_collapsed,
        "n_kept": len(out),
        "alphabet": alphabet,
        "task_type": task_type,
    }
    if len(out) == 0:
        raise ValueError("No valid rows left after cleaning")
    return out, stats, task_type


def build_data_card(df, filter_stats, csv_name=None, task_type="regression", extra=None):
    """Basic dataset statistics for the cleaned table."""
    if csv_name is None:
        csv_name = _route.ACTIVE_CSV
    seq_lens = df[SEQUENCE_COL].astype(str).str.len()
    labels = df[LABEL_COL]
    fold_counts = None
    if "fold" in df.columns:
        fold_counts = {
            str(int(k)): int(v)
            for k, v in df["fold"].value_counts().sort_index().items()
        }
    split_counts = None
    if "split" in df.columns:
        split_counts = {
            str(k): int(v) for k, v in df["split"].value_counts().sort_index().items()
        }

    label_info = {
        "min": float(pd.to_numeric(labels, errors="coerce").min()),
        "max": float(pd.to_numeric(labels, errors="coerce").max()),
        "mean": float(pd.to_numeric(labels, errors="coerce").mean()),
        "median": float(pd.to_numeric(labels, errors="coerce").median()),
        "std": float(pd.to_numeric(labels, errors="coerce").std(ddof=0)),
    }
    if task_type == "classification":
        vc = labels.astype(int).value_counts().sort_index()
        label_info.update(
            {
                "num_classes": int(labels.astype(int).max()) + 1,
                "n_unique": int(labels.nunique()),
                "class_count_min": int(vc.min()),
                "class_count_max": int(vc.max()),
                "class_count_median": float(vc.median()),
            }
        )

    card = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "csv_name": csv_name,
        "task_type": task_type,
        "max_seq_len": MAX_SEQ_LEN,
        "alphabet": ALPHABET,
        "sequence_col": SEQUENCE_COL,
        "label_col": LABEL_COL,
        "filters": filter_stats,
        "n_rows": len(df),
        "seq_length": {
            "min": int(seq_lens.min()),
            "max": int(seq_lens.max()),
            "mean": float(seq_lens.mean()),
            "median": float(seq_lens.median()),
            "std": float(seq_lens.std(ddof=0)),
            "gt_max_seq_len": int((seq_lens > MAX_SEQ_LEN).sum()),
        },
        "label": label_info,
        "fold_counts": fold_counts,
        "split_counts": split_counts,
        "n_folds": N_FOLDS if task_type == "regression" else 1,
        "test_fraction": TEST_FRACTION if task_type == "regression" else CLS_TEST_FRAC,
    }
    if extra:
        card.update(extra)
    return card


def write_data_card(card, csv_name=None):
    """Write data card JSON under artifacts/data_cards/."""
    if csv_name is None:
        csv_name = _route.ACTIVE_CSV
    os.makedirs(DATA_CARD_DIR, exist_ok=True)
    path = os.path.join(DATA_CARD_DIR, f"{_csv_stem(csv_name)}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def assign_folds(
    df,
    n_folds=N_FOLDS,
    test_fraction=TEST_FRACTION,
    random_state=FOLD_RANDOM_STATE,
    write_back=True,
    csv_name=None,
):
    """
    Regression: hold out a test set, then assign validation fold ids (0 .. n_folds-1).
    Test rows get fold == FOLD_TEST (-1). Optionally writes columns back to the source CSV.
    Cleaning uses write_back=False so the raw dataset file is not shrunk.
    """
    if csv_name is None:
        csv_name = _route.ACTIVE_CSV
    n = len(df)
    n_test = max(1, int(round(n * test_fraction)))
    n_cv = n - n_test
    if n_cv < n_folds:
        raise ValueError(
            f"Need at least {n_folds} CV rows after {test_fraction:.0%} test holdout, got {n_cv} (n={n})"
        )

    rng = np.random.RandomState(random_state)
    perm = rng.permutation(n)
    cv_idx = perm[n_test:]

    fold_sizes = np.full(n_folds, n_cv // n_folds, dtype=np.int64)
    fold_sizes[: n_cv % n_folds] += 1

    fold_col = np.full(n, FOLD_TEST, dtype=np.int64)
    start = 0
    for fold_id, size in enumerate(fold_sizes):
        fold_col[cv_idx[start : start + size]] = fold_id
        start += size

    out = df.copy()
    out["fold"] = fold_col
    if write_back:
        out.to_csv(_csv_path(csv_name), index=False)
    return out


def assign_classification_splits(df, csv_name=None, random_state=FOLD_RANDOM_STATE):
    """
    Classification: within-class MMseqs clusters -> 70/20/10 train/val/test.
    Encodes fold as train=FOLD_TRAIN(1), val=FOLD_VAL(0), test=FOLD_TEST(-1).
    """
    if csv_name is None:
        csv_name = _route.ACTIVE_CSV
    sys.path.insert(0, os.path.join(PROJECT_DIR, "tools"))
    from mmseqs_cluster import assign_cluster_splits_per_class  # noqa: WPS433

    os.makedirs(SPLIT_DIR, exist_ok=True)
    stem = _csv_stem(csv_name)
    cache_path = os.path.join(
        SPLIT_DIR, f"{stem}_mmseqs{int(CLS_MMSEQS_MIN_SEQ_ID * 100)}_clusters.csv"
    )
    # MMseqs is unreliable on some networked filesystems; keep scratch under /tmp.
    work_root = os.path.join("/tmp", f"rnaomni_mmseqs_{stem}_{os.getpid()}")

    clustered = assign_cluster_splits_per_class(
        df,
        sequence_col=SEQUENCE_COL,
        label_col=LABEL_COL,
        train_frac=CLS_TRAIN_FRAC,
        val_frac=CLS_VAL_FRAC,
        test_frac=CLS_TEST_FRAC,
        min_seq_id=CLS_MMSEQS_MIN_SEQ_ID,
        coverage=CLS_MMSEQS_COVERAGE,
        random_state=random_state,
        cache_path=cache_path,
        work_root=work_root,
    )

    fold = np.full(len(clustered), FOLD_TRAIN, dtype=np.int64)
    fold[clustered["split"].to_numpy() == "val"] = FOLD_VAL
    fold[clustered["split"].to_numpy() == "test"] = FOLD_TEST
    clustered["fold"] = fold

    assign_path = os.path.join(
        SPLIT_DIR, f"{stem}_mmseqs{int(CLS_MMSEQS_MIN_SEQ_ID * 100)}_split.csv"
    )
    clustered[["cluster_id", "split", "fold", LABEL_COL]].to_csv(assign_path, index=False)
    print(f"Data: wrote split assignments -> {assign_path}")
    print(
        f"Data: split counts {clustered['split'].value_counts().to_dict()} "
        f"(train/val/test = {CLS_TRAIN_FRAC:.0%}/{CLS_VAL_FRAC:.0%}/{CLS_TEST_FRAC:.0%})"
    )
    return clustered


def ensure_splits(df, task_type: str, csv_name=None):
    """Assign folds/splits if missing or invalid for the task type."""
    if csv_name is None:
        csv_name = _route.ACTIVE_CSV
    need = False
    if "fold" not in df.columns:
        need = True
    else:
        folds = df["fold"].to_numpy()
        if task_type == "classification":
            # Expect test + val(0) + train(>=1)
            if (folds == FOLD_TEST).sum() == 0 or (folds == FOLD_VAL).sum() == 0:
                need = True
            if folds.max() < FOLD_TRAIN:
                need = True
        else:
            if (folds == FOLD_TEST).sum() == 0:
                need = True
            if folds.min() < FOLD_TEST or folds.max() >= N_FOLDS:
                need = True

    if not need:
        return df

    if task_type == "classification":
        # Prefer reloading a previously written split table (same cleaned row order).
        stem = _csv_stem(csv_name)
        assign_path = os.path.join(
            SPLIT_DIR, f"{stem}_mmseqs{int(CLS_MMSEQS_MIN_SEQ_ID * 100)}_split.csv"
        )
        cluster_path = os.path.join(
            SPLIT_DIR, f"{stem}_mmseqs{int(CLS_MMSEQS_MIN_SEQ_ID * 100)}_clusters.csv"
        )
        if os.path.isfile(assign_path) and os.path.isfile(cluster_path):
            assign = pd.read_csv(assign_path)
            clusters = pd.read_csv(cluster_path)
            if len(assign) == len(df) and len(clusters) == len(df):
                out = df.copy()
                out["cluster_id"] = clusters["cluster_id"].astype(str).to_numpy()
                out["split"] = assign["split"].astype(str).to_numpy()
                out["fold"] = assign["fold"].astype(np.int64).to_numpy()
                print(f"Data: loaded cached classification splits -> {assign_path}")
                return out
        print(
            f"Data: assigning MMseqs cluster splits "
            f"({CLS_TRAIN_FRAC:.0%}/{CLS_VAL_FRAC:.0%}/{CLS_TEST_FRAC:.0%}, "
            f"identity={CLS_MMSEQS_MIN_SEQ_ID})..."
        )
        return assign_classification_splits(df, csv_name=csv_name)

    print(f"Data: assigning test holdout ({TEST_FRACTION:.0%}) + {N_FOLDS}-fold CV...")
    return assign_folds(df, write_back=False, csv_name=csv_name)
